package provider

import (
	"context"
	"encoding/json"
	"fmt"
	"unicode/utf8"

	"soul-link/internal/model"
)

const defaultSummaryPrompt = "Summarize the conversation history above into a concise summary, preserving key information and conclusions. Respond in the same language as the conversation."

// WithCompression 启用历史摘要压缩
func WithCompression(maxTokens, keepRecent int, summaryPrompt model.TextData) ConversationOption {
	return func(c *Conversation) {
		if summaryPrompt == "" {
			summaryPrompt = defaultSummaryPrompt
		}
		c.compress = &compressionConfig{
			maxTokens:     maxTokens,
			keepRecent:    keepRecent,
			summaryPrompt: summaryPrompt,
		}
	}
}

type compressionConfig struct {
	maxTokens     int
	keepRecent    int
	summaryPrompt model.TextData
}

// estimateTokens 估算消息列表的 token 数量（以 rune 数为统一基准）
func estimateTokens(msgs []model.Message) int {
	var n int
	for _, msg := range msgs {
		for _, c := range msg.Contents {
			n += utf8.RuneCountInString(string(c.Text))
			if c.ToolCall != nil {
				if b, err := json.Marshal(c.ToolCall.Arguments); err == nil {
					n += utf8.RuneCountInString(string(b))
				}
			}
			if c.ToolResult != nil {
				n += utf8.RuneCountInString(c.ToolResult.Content)
			}
			if c.Thinking != nil {
				n += utf8.RuneCountInString(c.Thinking.Text)
			}
		}
	}
	return n
}

// maybeCompress 检查 base 是否超出 token 限制，超出时调用 LLM 将旧消息压缩为摘要
func (c *Conversation) maybeCompress(ctx context.Context, base []model.Message) ([]model.Message, bool, error) {
	cfg := c.compress
	if cfg == nil || cfg.maxTokens <= 0 || estimateTokens(base) <= cfg.maxTokens {
		return base, false, nil
	}

	// 分离 system 消息与对话消息
	var sysMsgs, convMsgs []model.Message
	for _, msg := range base {
		if msg.Role == model.MessageRoleSystem {
			sysMsgs = append(sysMsgs, msg)
		} else {
			convMsgs = append(convMsgs, msg)
		}
	}

	if len(convMsgs) <= cfg.keepRecent {
		return base, false, nil
	}

	splitAt := len(convMsgs) - cfg.keepRecent

	for splitAt > 0 && convMsgs[splitAt].Role != model.MessageRoleUser {
		splitAt--
	}

	for splitAt > 0 && convMsgs[splitAt-1].Role == model.MessageRoleTool {
		splitAt--
		for splitAt > 0 && convMsgs[splitAt].Role != model.MessageRoleUser {
			splitAt--
		}
	}

	for splitAt > 0 && convMsgs[splitAt].Role != model.MessageRoleUser {
		splitAt--
	}

	if splitAt == 0 {
		return base, false, nil
	}
	old := convMsgs[:splitAt]
	recent := convMsgs[splitAt:]

	summaryReq := make([]model.Message, 0, len(sysMsgs)+len(old)+1)
	summaryReq = append(summaryReq, sysMsgs...)
	summaryReq = append(summaryReq, old...)
	summaryReq = append(summaryReq, model.Message{
		Role:     model.MessageRoleUser,
		Contents: []model.Content{{Type: model.ContentTypeText, Text: cfg.summaryPrompt}},
	})

	contents, _, err := c.provider.Complete(ctx, summaryReq, nil)
	if err != nil {
		return nil, false, fmt.Errorf("compress history: %w", err)
	}

	var summaryText model.TextData
	for _, ct := range contents {
		if ct.Type == model.ContentTypeText && ct.Text != "" {
			summaryText = ct.Text
			break
		}
	}
	if summaryText == "" {
		return base, false, nil
	}

	summaryUser := model.Message{
		Role:     model.MessageRoleUser,
		Contents: []model.Content{{Type: model.ContentTypeText, Text: "Summary of previous conversation: "}},
	}

	summaryAssistant := model.Message{
		Role:     model.MessageRoleAssistant,
		Contents: []model.Content{{Type: model.ContentTypeText, Text: summaryText}},
	}

	compressed := make([]model.Message, 0, len(sysMsgs)+2+len(recent))
	compressed = append(compressed, sysMsgs...)
	compressed = append(compressed, summaryUser, summaryAssistant)
	compressed = append(compressed, recent...)

	return compressed, true, nil
}
