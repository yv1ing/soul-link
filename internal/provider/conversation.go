package provider

import (
	"context"
	"strings"
	"sync"

	"soul-link/internal/model"
)

// ToolExecutor 提供工具同步执行能力
type ToolExecutor interface {
	Execute(call model.ToolCall) model.ToolResult
}

// TextContent 将纯文本包装为 []Content
func TextContent(text model.TextData) []model.Content {
	return []model.Content{{Type: model.ContentTypeText, Text: text}}
}

// ImageInput 描述一张图片输入，支持 URL 或 base64 两种方式
type ImageInput struct {
	URL       model.ImageData // URL 方式，与 Base64 二选一
	Base64    model.ImageData // base64 编码数据，与 URL 二选一
	MediaType string          // base64 时必填，如 "image/png"
}

// UserContent 将若干文本和图片混合包装为 []Content
func UserContent(texts []model.TextData, images []ImageInput) []model.Content {
	contents := make([]model.Content, 0, len(texts)+len(images))
	for _, t := range texts {
		contents = append(contents, model.Content{Type: model.ContentTypeText, Text: t})
	}
	for _, img := range images {
		if img.Base64 != "" {
			contents = append(contents, model.Content{
				Type:      model.ContentTypeImageRaw,
				Image:     img.Base64,
				MediaType: img.MediaType,
			})
		} else if img.URL != "" {
			contents = append(contents, model.Content{
				Type:  model.ContentTypeImageURL,
				Image: img.URL,
			})
		}
	}
	return contents
}

const maxToolIterations = 10

// Conversation 在 Provider 之上维护多轮对话历史
type Conversation struct {
	mu       sync.Mutex
	provider Provider
	history  []model.Message
	compress *compressionConfig
	executor ToolExecutor
	stopPrev context.CancelFunc
}

// ConversationOption 用于配置 Conversation
type ConversationOption func(*Conversation)

// WithToolExecutor 启用内置 tool call loop，工具调用由 exec 自动执行并注入结果
func WithToolExecutor(exec ToolExecutor) ConversationOption {
	return func(c *Conversation) {
		c.executor = exec
	}
}

// NewConversation 创建新的对话实例
func NewConversation(p Provider, opts ...ConversationOption) *Conversation {
	c := &Conversation{provider: p}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// AddSystem 向历史中追加系统消息
func (c *Conversation) AddSystem(text model.TextData) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.history = append(c.history, model.Message{
		Role:     model.MessageRoleSystem,
		Contents: []model.Content{{Type: model.ContentTypeText, Text: text}},
	})
}

// Complete 发起非流式请求
func (c *Conversation) Complete(ctx context.Context, userContents []model.Content, tools []model.ToolDef) ([]model.Content, *model.Usage, error) {
	c.mu.Lock()
	c.cancelPrev()
	ctx, cancel := context.WithCancel(ctx)
	c.stopPrev = cancel
	c.mu.Unlock()
	defer cancel()

	if c.executor == nil || len(tools) == 0 {
		return c.completeOnce(ctx, userContents, tools)
	}

	var (
		totalUsage   model.Usage
		prevThinking []model.Content
	)
	input := userContents

	for range maxToolIterations {
		contents, usage, err := c.completeOnce(ctx, input, tools)
		if err != nil {
			return nil, nil, err
		}
		if usage != nil {
			totalUsage.InputTokens += usage.InputTokens
			totalUsage.OutputTokens += usage.OutputTokens
		}
		input = nil

		var calls []model.ToolCall
		for _, ct := range contents {
			if ct.Type == model.ContentTypeToolCall && ct.ToolCall != nil {
				calls = append(calls, *ct.ToolCall)
			}
		}
		if len(calls) == 0 {
			return append(prevThinking, contents...), &totalUsage, nil
		}

		for _, ct := range contents {
			if ct.Type == model.ContentTypeThinking {
				prevThinking = append(prevThinking, ct)
			}
		}

		results := make([]model.ToolResult, len(calls))
		for i, call := range calls {
			results[i] = c.executor.Execute(call)
		}
		c.InjectToolResults(results...)
	}

	// 超出最大迭代次数，最后一次不带工具强制获取文本回复
	contents, usage, err := c.completeOnce(ctx, nil, nil)
	if err != nil {
		return nil, nil, err
	}
	if usage != nil {
		totalUsage.InputTokens += usage.InputTokens
		totalUsage.OutputTokens += usage.OutputTokens
	}
	return append(prevThinking, contents...), &totalUsage, nil
}

// completeOnce 执行单轮非流式请求并将用户消息和助手回复写入历史
func (c *Conversation) completeOnce(ctx context.Context, userContents []model.Content, tools []model.ToolDef) ([]model.Content, *model.Usage, error) {
	c.mu.Lock()
	base := c.snapshot()
	c.mu.Unlock()

	var userMsg *model.Message
	if len(userContents) > 0 {
		msg := model.Message{Role: model.MessageRoleUser, Contents: userContents}
		userMsg = &msg
	}

	compressed, wasCompressed, err := c.maybeCompress(ctx, base)
	if err != nil {
		return nil, nil, err
	}

	callSnapshot := compressed
	if userMsg != nil {
		callSnapshot = append(callSnapshot, *userMsg)
	}

	contents, usage, err := c.provider.Complete(ctx, callSnapshot, tools)
	if err != nil {
		return nil, nil, err
	}

	c.mu.Lock()
	if wasCompressed {
		c.history = compressed
	}
	if userMsg != nil {
		c.history = append(c.history, *userMsg)
	}
	c.history = append(c.history, model.Message{
		Role:     model.MessageRoleAssistant,
		Contents: contents,
	})
	c.mu.Unlock()

	return cloneContents(contents), usage, nil
}

// Stream 发起流式请求
func (c *Conversation) Stream(ctx context.Context, userContents []model.Content, tools []model.ToolDef) (<-chan Event, error) {
	c.mu.Lock()
	c.cancelPrev()
	ctx, cancel := context.WithCancel(ctx)
	c.stopPrev = cancel
	c.mu.Unlock()

	firstStream, err := c.streamOnce(ctx, userContents, tools)
	if err != nil {
		cancel()
		return nil, err
	}

	if c.executor == nil || len(tools) == 0 {
		return firstStream, nil
	}

	out := make(chan Event, 16)
	go func() {
		defer close(out)
		defer cancel()

		send := func(e Event) bool {
			select {
			case out <- e:
				return true
			case <-ctx.Done():
				return false
			}
		}

		var totalUsage model.Usage
		upstream := firstStream
		for toolIter := 0; toolIter <= maxToolIterations; toolIter++ {
			var toolCalls []model.ToolCall

			for e := range upstream {
				switch e.Type {
				case EventTypeToolCall:
					if e.ToolCall != nil {
						toolCalls = append(toolCalls, *e.ToolCall)
					}
					if !send(e) {
						return
					}
				case EventTypeDone:
					// 累加用量，不转发中间迭代的 Done 事件
					if e.Usage != nil {
						totalUsage.InputTokens += e.Usage.InputTokens
						totalUsage.OutputTokens += e.Usage.OutputTokens
					}
				case EventTypeError:
					send(e)
					return
				default:
					if !send(e) {
						return
					}
				}
			}

			if len(toolCalls) == 0 {
				// 最终回复，发送累计用量
				send(Event{Type: EventTypeDone, Usage: &totalUsage})
				return
			}

			results := make([]model.ToolResult, len(toolCalls))
			for i, call := range toolCalls {
				results[i] = c.executor.Execute(call)
			}
			c.InjectToolResults(results...)

			// 超出最大迭代次数，不带工具强制获取文本回复
			var nextTools []model.ToolDef
			if toolIter+1 < maxToolIterations {
				nextTools = tools
			}

			next, err := c.streamOnce(ctx, nil, nextTools)
			if err != nil {
				send(Event{Type: EventTypeError, Err: err})
				return
			}
			upstream = next
		}
	}()

	return out, nil
}

// streamOnce 执行单轮流式请求，收到 EventTypeDone 后将用户消息和助手回复写入历史
func (c *Conversation) streamOnce(ctx context.Context, userContents []model.Content, tools []model.ToolDef) (<-chan Event, error) {
	c.mu.Lock()
	base := c.snapshot()
	c.mu.Unlock()

	var userMsg *model.Message
	if len(userContents) > 0 {
		msg := model.Message{Role: model.MessageRoleUser, Contents: userContents}
		userMsg = &msg
	}

	compressed, wasCompressed, err := c.maybeCompress(ctx, base)
	if err != nil {
		return nil, err
	}

	callSnapshot := compressed
	if userMsg != nil {
		callSnapshot = append(callSnapshot, *userMsg)
	}

	upstream, err := c.provider.Stream(ctx, callSnapshot, tools)
	if err != nil {
		return nil, err
	}

	ch := make(chan Event, 16)
	go func() {
		defer close(ch)

		send := func(e Event) bool {
			select {
			case ch <- e:
				return true
			case <-ctx.Done():
				return false
			}
		}

		var (
			textBuf        strings.Builder
			thinkingBlocks []model.Content
			toolCallBlocks []model.Content
		)

		commitHistory := func() {
			var contents []model.Content
			contents = append(contents, thinkingBlocks...)
			if textBuf.Len() > 0 {
				contents = append(contents, model.Content{
					Type: model.ContentTypeText,
					Text: model.TextData(textBuf.String()),
				})
			}
			contents = append(contents, toolCallBlocks...)
			c.mu.Lock()
			if wasCompressed {
				c.history = compressed
			}
			if userMsg != nil {
				c.history = append(c.history, *userMsg)
			}
			c.history = append(c.history, model.Message{
				Role:     model.MessageRoleAssistant,
				Contents: contents,
			})
			c.mu.Unlock()
		}

		var committed, hasError bool
		for e := range upstream {
			switch e.Type {
			case EventTypeThinkingDone:
				td := *e.Thinking
				thinkingBlocks = append(thinkingBlocks, model.Content{
					Type:     model.ContentTypeThinking,
					Thinking: &td,
				})
			case EventTypeTextDelta:
				textBuf.WriteString(e.Text)
			case EventTypeToolCall:
				tc := cloneToolCall(e.ToolCall)
				toolCallBlocks = append(toolCallBlocks, model.Content{
					Type:     model.ContentTypeToolCall,
					ToolCall: &tc,
				})
			case EventTypeDone:
				commitHistory()
				committed = true
			case EventTypeError:
				hasError = true
			}
			if !send(e) {
				return
			}
		}

		if !committed && !hasError && (textBuf.Len() > 0 || len(toolCallBlocks) > 0 || len(thinkingBlocks) > 0) {
			commitHistory()
		}
	}()

	return ch, nil
}

// InjectToolResults 将工具执行结果注入历史，供下一轮对话使用
func (c *Conversation) InjectToolResults(results ...model.ToolResult) {
	if len(results) == 0 {
		return
	}
	contents := make([]model.Content, len(results))
	for i, r := range results {
		contents[i] = model.Content{
			Type:       model.ContentTypeToolResult,
			ToolResult: &r,
		}
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.history = append(c.history, model.Message{
		Role:     model.MessageRoleTool,
		Contents: contents,
	})
}

// History 返回当前历史的深拷贝
func (c *Conversation) History() []model.Message {
	c.mu.Lock()
	defer c.mu.Unlock()
	msgs := make([]model.Message, len(c.history))
	for i, msg := range c.history {
		msgs[i] = model.Message{
			Role:     msg.Role,
			Contents: cloneContents(msg.Contents),
		}
	}
	return msgs
}

// Reset 清空对话历史并终止进行中的流式请求
func (c *Conversation) Reset() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.cancelPrev()
	c.history = nil
}

func (c *Conversation) snapshot() []model.Message {
	s := make([]model.Message, len(c.history))
	for i, msg := range c.history {
		s[i] = model.Message{
			Role:     msg.Role,
			Contents: cloneContents(msg.Contents),
		}
	}
	return s
}

// cancelPrev 取消上一次 Stream 产生的 goroutine，必须持有 mu
func (c *Conversation) cancelPrev() {
	if c.stopPrev != nil {
		c.stopPrev()
		c.stopPrev = nil
	}
}

// cloneToolCall 深拷贝 ToolCall，递归拷贝 Arguments 中的嵌套 map/slice
func cloneToolCall(tc *model.ToolCall) model.ToolCall {
	t := *tc
	if t.Arguments != nil {
		t.Arguments = cloneMap(t.Arguments)
	}
	return t
}

func cloneAny(v any) any {
	switch val := v.(type) {
	case map[string]any:
		return cloneMap(val)
	case []any:
		return cloneSlice(val)
	default:
		return v
	}
}

func cloneMap(m map[string]any) map[string]any {
	dst := make(map[string]any, len(m))
	for k, v := range m {
		dst[k] = cloneAny(v)
	}
	return dst
}

func cloneSlice(s []any) []any {
	dst := make([]any, len(s))
	for i, v := range s {
		dst[i] = cloneAny(v)
	}
	return dst
}

// cloneContents 深拷贝 Content 切片，使返回值与 history 内部解耦
func cloneContents(src []model.Content) []model.Content {
	dst := make([]model.Content, len(src))
	for i, c := range src {
		dst[i] = c
		if c.Thinking != nil {
			t := *c.Thinking
			dst[i].Thinking = &t
		}
		if c.ToolCall != nil {
			t := cloneToolCall(c.ToolCall)
			dst[i].ToolCall = &t
		}
		if c.ToolResult != nil {
			t := *c.ToolResult
			dst[i].ToolResult = &t
		}
	}
	return dst
}
