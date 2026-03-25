package provider

import (
	"context"
	"strings"
	"sync"

	"soul-link/internal/model"
)

// Conversation 在 Provider 之上维护多轮对话历史
type Conversation struct {
	mu       sync.Mutex
	provider Provider
	history  []model.Message
	compress *compressionConfig
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
		Contents: []model.Content{{Type: model.ContentTypeTextData, TextData: text}},
	})
}

// Complete 发起非流式请求；成功后将用户消息和助手回复一次性写入历史
func (c *Conversation) Complete(ctx context.Context, userText string, tools []model.ToolSet) ([]model.Content, *model.Usage, error) {
	c.mu.Lock()
	base := c.snapshot()
	c.mu.Unlock()

	var userMsg *model.Message
	if userText != "" {
		msg := model.Message{
			Role:     model.MessageRoleUser,
			Contents: []model.Content{{Type: model.ContentTypeTextData, TextData: userText}},
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

// Stream 发起流式请求；收到 EventTypeDone 后将用户消息和助手回复一次性写入历史
func (c *Conversation) Stream(ctx context.Context, userText string, tools []model.ToolSet) (<-chan Event, error) {
	c.mu.Lock()
	base := c.snapshot()
	c.mu.Unlock()

	var userMsg *model.Message
	if userText != "" {
		msg := model.Message{
			Role:     model.MessageRoleUser,
			Contents: []model.Content{{Type: model.ContentTypeTextData, TextData: userText}},
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
			textBuf   strings.Builder
			toolCalls []model.Content
		)

		for e := range upstream {
			switch e.Type {
			case EventTypeTextDelta:
				textBuf.WriteString(e.Text)
			case EventTypeToolCall:
				toolCalls = append(toolCalls, model.Content{
					Type:     model.ContentTypeToolCall,
					ToolCall: e.ToolCall,
				})
			case EventTypeDone:
				var contents []model.Content
				if textBuf.Len() > 0 {
					contents = append(contents, model.Content{
						Type:     model.ContentTypeTextData,
						TextData: textBuf.String(),
					})
				}
				contents = append(contents, toolCalls...)
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
			if !send(e) {
				return
			}
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

// snapshot 返回当前历史的浅拷贝，调用前必须持有锁
func (c *Conversation) snapshot() []model.Message {
	s := make([]model.Message, len(c.history))
	copy(s, c.history)
	return s
}
