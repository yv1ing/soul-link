package anthropic

import (
	"context"
	"strings"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"

	"soul-link/internal/model"
	"soul-link/internal/provider"
)

const defaultMaxTokens = int64(4096)

// Provider 封装 Anthropic Messages API 客户端
type Provider struct {
	client    sdk.Client
	model     string
	maxTokens int64
}

// New 创建 Anthropic Provider。maxTokens <= 0 时使用默认值；baseURL 为空时使用 SDK 默认地址。
func New(apiKey, baseURL, modelName string, maxTokens int64, opts ...option.RequestOption) *Provider {
	base := []option.RequestOption{option.WithAPIKey(apiKey)}
	if baseURL != "" {
		base = append(base, option.WithBaseURL(baseURL))
	}

	p := &Provider{
		client:    sdk.NewClient(append(base, opts...)...),
		model:     modelName,
		maxTokens: defaultMaxTokens,
	}
	if maxTokens > 0 {
		p.maxTokens = maxTokens
	}
	return p
}

// Stream 发起流式请求，通过 channel 逐步投递事件，ctx 取消时 goroutine 安全退出
func (p *Provider) Stream(ctx context.Context, messages []model.Message, tools []model.ToolSet) (<-chan provider.Event, error) {
	stream := p.client.Messages.NewStreaming(ctx, p.buildRequest(messages, tools))

	ch := make(chan provider.Event, 16)
	go func() {
		defer close(ch)

		send := func(e provider.Event) bool {
			select {
			case ch <- e:
				return true
			case <-ctx.Done():
				return false
			}
		}

		var (
			toolBuf         strings.Builder
			currentToolID   string
			currentToolName string
			inputTokens     int
		)

		for stream.Next() {
			event := stream.Current()
			switch event.Type {

			case "message_start":
				inputTokens = int(event.AsMessageStart().Message.Usage.InputTokens)

			case "content_block_start":
				cb := event.AsContentBlockStart().ContentBlock
				if cb.Type == "tool_use" {
					currentToolID = cb.ID
					currentToolName = cb.Name
					toolBuf.Reset()
				}

			case "content_block_delta":
				delta := event.AsContentBlockDelta().Delta
				switch delta.Type {
				case "text_delta":
					if !send(provider.Event{Type: provider.EventTypeTextDelta, Text: delta.AsTextDelta().Text}) {
						return
					}
				case "input_json_delta":
					toolBuf.WriteString(delta.AsInputJSONDelta().PartialJSON)
				}

			case "content_block_stop":
				if currentToolID == "" {
					continue
				}
				tc, err := provider.ParseToolCall(currentToolID, currentToolName, toolBuf.String())
				currentToolID, currentToolName = "", ""
				toolBuf.Reset()
				if err != nil {
					send(provider.Event{Type: provider.EventTypeError, Err: err})
					return
				}
				if !send(provider.Event{Type: provider.EventTypeToolCall, ToolCall: tc}) {
					return
				}

			case "message_delta":
				u := event.AsMessageDelta().Usage
				if !send(provider.Event{
					Type:  provider.EventTypeDone,
					Usage: &model.Usage{InputTokens: inputTokens, OutputTokens: int(u.OutputTokens)},
				}) {
					return
				}
				return
			}
		}

		if err := stream.Err(); err != nil {
			send(provider.Event{Type: provider.EventTypeError, Err: err})
		}
	}()

	return ch, nil
}

// Complete 发起非流式请求，返回完整内容列表和用量统计
func (p *Provider) Complete(ctx context.Context, messages []model.Message, tools []model.ToolSet) ([]model.Content, *model.Usage, error) {
	msg, err := p.client.Messages.New(ctx, p.buildRequest(messages, tools))
	if err != nil {
		return nil, nil, err
	}

	contents, err := extractContents(msg.Content)
	if err != nil {
		return nil, nil, err
	}

	usage := &model.Usage{InputTokens: int(msg.Usage.InputTokens), OutputTokens: int(msg.Usage.OutputTokens)}
	return contents, usage, nil
}

// buildRequest 将通用消息和工具定义组装为 Messages API 请求参数
func (p *Provider) buildRequest(messages []model.Message, tools []model.ToolSet) sdk.MessageNewParams {
	msgParams, system := convertMessages(messages)
	req := sdk.MessageNewParams{
		Model:     p.model,
		MaxTokens: p.maxTokens,
		Messages:  msgParams,
		Tools:     convertTools(tools),
	}

	if system != "" {
		req.System = []sdk.TextBlockParam{{Text: system}}
	}
	return req
}

// convertMessages 将通用消息历史转换为 Messages API 输入格式
func convertMessages(messages []model.Message) (params []sdk.MessageParam, system string) {
	for _, msg := range messages {
		switch msg.Role {

		case model.MessageRoleSystem:
			for _, c := range msg.Contents {
				if c.Type == model.ContentTypeTextData {
					if system != "" {
						system += "\n"
					}
					system += c.TextData
				}
			}

		case model.MessageRoleUser:
			var blocks []sdk.ContentBlockParamUnion
			for _, c := range msg.Contents {
				switch c.Type {
				case model.ContentTypeTextData:
					blocks = append(blocks, sdk.NewTextBlock(c.TextData))
				case model.ContentTypeToolResult:
					if c.ToolResult != nil {
						tr := c.ToolResult
						blocks = append(blocks, sdk.NewToolResultBlock(tr.CallID, tr.Content, tr.IsError))
					}
				}
			}
			if len(blocks) > 0 {
				params = append(params, sdk.NewUserMessage(blocks...))
			}

		case model.MessageRoleAssistant:
			var blocks []sdk.ContentBlockParamUnion
			for _, c := range msg.Contents {
				switch c.Type {
				case model.ContentTypeTextData:
					blocks = append(blocks, sdk.NewTextBlock(c.TextData))
				case model.ContentTypeToolCall:
					if c.ToolCall != nil {
						tc := c.ToolCall
						blocks = append(blocks, sdk.NewToolUseBlock(tc.ToolID, tc.Arguments, tc.ToolName))
					}
				}
			}
			if len(blocks) > 0 {
				params = append(params, sdk.NewAssistantMessage(blocks...))
			}

		case model.MessageRoleTool:
			var blocks []sdk.ContentBlockParamUnion
			for _, c := range msg.Contents {
				if c.Type == model.ContentTypeToolResult && c.ToolResult != nil {
					tr := c.ToolResult
					blocks = append(blocks, sdk.NewToolResultBlock(tr.CallID, tr.Content, tr.IsError))
				}
			}

			if len(blocks) == 0 {
				continue
			}

			n := len(params)
			if n > 0 && params[n-1].Role == sdk.MessageParamRoleUser {
				params[n-1].Content = append(params[n-1].Content, blocks...)
			} else {
				params = append(params, sdk.NewUserMessage(blocks...))
			}
		}
	}
	return params, system
}

// convertTools 将通用工具定义转换为 Messages API tool 参数
func convertTools(tools []model.ToolSet) []sdk.ToolUnionParam {
	if len(tools) == 0 {
		return nil
	}

	params := make([]sdk.ToolUnionParam, len(tools))
	for i, t := range tools {
		tp := sdk.ToolParam{
			Name:        t.ToolName,
			Description: sdk.String(t.Description),
			InputSchema: buildInputSchema(t.Parameters),
		}
		params[i] = sdk.ToolUnionParam{OfTool: &tp}
	}
	return params
}

// buildInputSchema 将 JSON Schema map 转换为 Messages API 工具输入模式
// properties 和 required 映射至专用字段，其余字段通过 ExtraFields 透传
func buildInputSchema(params map[string]any) sdk.ToolInputSchemaParam {
	schema := sdk.ToolInputSchemaParam{}
	for k, v := range params {
		switch k {
		case "properties":
			schema.Properties = v
		case "required":
			if list, ok := v.([]any); ok {
				for _, item := range list {
					if s, ok := item.(string); ok {
						schema.Required = append(schema.Required, s)
					}
				}
			}
		default:
			if schema.ExtraFields == nil {
				schema.ExtraFields = make(map[string]any)
			}
			schema.ExtraFields[k] = v
		}
	}
	return schema
}

// extractContents 从 Messages API 响应内容块中提取文本和工具调用内容
func extractContents(blocks []sdk.ContentBlockUnion) ([]model.Content, error) {
	var contents []model.Content
	for _, block := range blocks {
		switch block.Type {
		case "text":
			contents = append(contents, model.Content{
				Type:     model.ContentTypeTextData,
				TextData: block.AsText().Text,
			})
		case "tool_use":
			tu := block.AsToolUse()
			tc, err := provider.ParseToolCall(tu.ID, tu.Name, string(tu.Input))
			if err != nil {
				return nil, err
			}
			contents = append(contents, model.Content{Type: model.ContentTypeToolCall, ToolCall: tc})
		}
	}
	return contents, nil
}
