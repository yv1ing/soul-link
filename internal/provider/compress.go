package provider

import (
	"context"
	"encoding/json"
	"fmt"

	"soul-link/internal/model"
)

const defaultSummaryPrompt = "请将以下对话历史简洁地总结为一段摘要，保留关键信息和结论，使用与对话相同的语言。"

// WithCompression 启用历史摘要压缩。summaryPrompt 为空时使用默认提示词。
func WithCompression(maxTokens, keepRecent int, summaryPrompt string) ConversationOption {
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
	summaryPrompt string
}

// estimateTokens 通过字节数 / 3 估算消息列表的 token 数量。
func estimateTokens(msgs []model.Message) int {
	var n int
	for _, msg := range msgs {
		for _, c := range msg.Contents {
			n += len(c.TextData) / 3
			if c.ToolCall != nil {
				n += len(c.ToolCall.ToolName) / 3
				if b, err := json.Marshal(c.ToolCall.Arguments); err == nil {
					n += len(b) / 3
				}
			}
			if c.ToolResult != nil {
				n += len(c.ToolResult.Content) / 3
			}
		}
	}
	return n
}

// maybeCompress 检查 base 是否超出 token 限制，超出时调用 LLM 将旧消息压缩为摘要。
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

	// 确保 recent 从用户消息开始
	for splitAt > 0 && convMsgs[splitAt].Role != model.MessageRoleUser {
		splitAt--
	}
	// 确保 old 末尾是 assistant（而非 tool），否则摘要请求末尾会出现两条连续 user 消息
	// （tool 角色在各 provider 转换后成为 user 消息，随后摘要提示词也是 user 消息）
	for splitAt > 0 && convMsgs[splitAt-1].Role == model.MessageRoleTool {
		splitAt--
		for splitAt > 0 && convMsgs[splitAt].Role != model.MessageRoleUser {
			splitAt--
		}
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
		Contents: []model.Content{{Type: model.ContentTypeTextData, TextData: cfg.summaryPrompt}},
	})

	contents, _, err := c.provider.Complete(ctx, summaryReq, nil)
	if err != nil {
		return nil, false, fmt.Errorf("compress history: %w", err)
	}

	var summaryText string
	for _, ct := range contents {
		if ct.Type == model.ContentTypeTextData && ct.TextData != "" {
			summaryText = ct.TextData
			break
		}
	}
	if summaryText == "" {
		return base, false, nil
	}

	summaryMsg := model.Message{
		Role:     model.MessageRoleSystem,
		Contents: []model.Content{{Type: model.ContentTypeTextData, TextData: "对话摘要：\n" + summaryText}},
	}

	compressed := make([]model.Message, 0, len(sysMsgs)+1+len(recent))
	compressed = append(compressed, sysMsgs...)
	compressed = append(compressed, summaryMsg)
	compressed = append(compressed, recent...)
	return compressed, true, nil
}
