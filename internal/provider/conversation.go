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

const maxToolIterations = 10

// Conversation 在 Provider 之上维护多轮对话历史
type Conversation struct {
	mu       sync.Mutex
	provider Provider
	history  []model.Message
	compress *compressionConfig
	executor ToolExecutor
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
func (c *Conversation) AddSystem(text string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.history = append(c.history, model.Message{
		Role:     model.MessageRoleSystem,
		Contents: []model.Content{{Type: model.ContentTypeText, Text: text}},
	})
}

// Complete 发起非流式请求
func (c *Conversation) Complete(ctx context.Context, userText string, tools []model.ToolSet) ([]model.Content, *model.Usage, error) {
	if c.executor == nil || len(tools) == 0 {
		return c.completeOnce(ctx, userText, tools)
	}

	var totalUsage model.Usage
	text := userText

	for range maxToolIterations {
		contents, usage, err := c.completeOnce(ctx, text, tools)
		if err != nil {
			return nil, nil, err
		}
		if usage != nil {
			totalUsage.InputTokens += usage.InputTokens
			totalUsage.OutputTokens += usage.OutputTokens
		}
		text = ""

		var calls []model.ToolCall
		for _, ct := range contents {
			if ct.Type == model.ContentTypeToolCall && ct.ToolCall != nil {
				calls = append(calls, *ct.ToolCall)
			}
		}
		if len(calls) == 0 {
			return contents, &totalUsage, nil
		}

		results := make([]model.ToolResult, len(calls))
		for i, call := range calls {
			results[i] = c.executor.Execute(call)
		}
		c.InjectToolResults(results...)
	}

	// 超出最大迭代次数，最后一次不带工具强制获取文本回复
	contents, usage, err := c.completeOnce(ctx, "", nil)
	if err != nil {
		return nil, nil, err
	}
	if usage != nil {
		totalUsage.InputTokens += usage.InputTokens
		totalUsage.OutputTokens += usage.OutputTokens
	}
	return contents, &totalUsage, nil
}

// completeOnce 执行单轮非流式请求并将用户消息和助手回复写入历史
func (c *Conversation) completeOnce(ctx context.Context, userText string, tools []model.ToolSet) ([]model.Content, *model.Usage, error) {
	c.mu.Lock()
	base := c.snapshot()
	c.mu.Unlock()

	var userMsg *model.Message
	if userText != "" {
		msg := model.Message{
			Role:     model.MessageRoleUser,
			Contents: []model.Content{{Type: model.ContentTypeText, Text: userText}},
		}
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

	return contents, usage, nil
}

// Stream 发起流式请求
func (c *Conversation) Stream(ctx context.Context, userText string, tools []model.ToolSet) (<-chan Event, error) {
	firstStream, err := c.streamOnce(ctx, userText, tools)
	if err != nil {
		return nil, err
	}

	if c.executor == nil || len(tools) == 0 {
		return firstStream, nil
	}

	out := make(chan Event, 16)
	go func() {
		defer close(out)

		send := func(e Event) bool {
			select {
			case out <- e:
				return true
			case <-ctx.Done():
				return false
			}
		}

		upstream := firstStream
		for iter := range maxToolIterations {
			var toolCalls []model.ToolCall

			for e := range upstream {
				if e.Type == EventTypeToolCall && e.ToolCall != nil {
					toolCalls = append(toolCalls, *e.ToolCall)
				}
				if e.Type == EventTypeError {
					send(e)
					return
				}
				if !send(e) {
					return
				}
			}

			if len(toolCalls) == 0 {
				return
			}

			results := make([]model.ToolResult, len(toolCalls))
			for i, call := range toolCalls {
				results[i] = c.executor.Execute(call)
			}
			c.InjectToolResults(results...)

			// 达到最大迭代次数，最后一次不带工具强制获取文本回复
			if iter+1 >= maxToolIterations {
				final, err := c.streamOnce(ctx, "", nil)
				if err != nil {
					send(Event{Type: EventTypeError, Err: err})
					return
				}
				for e := range final {
					if e.Type == EventTypeError {
						send(e)
						return
					}
					if !send(e) {
						return
					}
				}
				return
			}

			next, err := c.streamOnce(ctx, "", tools)
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
func (c *Conversation) streamOnce(ctx context.Context, userText string, tools []model.ToolSet) (<-chan Event, error) {
	c.mu.Lock()
	base := c.snapshot()
	c.mu.Unlock()

	var userMsg *model.Message
	if userText != "" {
		msg := model.Message{
			Role:     model.MessageRoleUser,
			Contents: []model.Content{{Type: model.ContentTypeText, Text: userText}},
		}
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
					Text: textBuf.String(),
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
				thinkingBlocks = append(thinkingBlocks, model.Content{
					Type:     model.ContentTypeThinking,
					Thinking: e.Thinking,
				})
			case EventTypeTextDelta:
				textBuf.WriteString(e.Text)
			case EventTypeToolCall:
				toolCallBlocks = append(toolCallBlocks, model.Content{
					Type:     model.ContentTypeToolCall,
					ToolCall: e.ToolCall,
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

// History 返回当前历史的副本
func (c *Conversation) History() []model.Message {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.snapshot()
}

// Reset 清空对话历史
func (c *Conversation) Reset() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.history = nil
}

func (c *Conversation) snapshot() []model.Message {
	s := make([]model.Message, len(c.history))
	copy(s, c.history)
	return s
}
