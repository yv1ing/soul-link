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
	client     sdk.Client
	model      string
	thinking   *model.ThinkingConfig
	maxTokens  int64
	toolChoice string
	reqOpts    []option.RequestOption
}

type Option func(*Provider)

func WithMaxTokens(n int64) Option                           { return func(p *Provider) { p.maxTokens = n } }
func WithToolChoice(choice string) Option                    { return func(p *Provider) { p.toolChoice = choice } }
func WithRequestOptions(opts ...option.RequestOption) Option { return func(p *Provider) { p.reqOpts = append(p.reqOpts, opts...) } }

// New 创建 Anthropic Provider
func New(apiKey, baseURL, modelName string, thinking *model.ThinkingConfig, opts ...Option) *Provider {
	p := &Provider{model: modelName, thinking: thinking, maxTokens: defaultMaxTokens}
	for _, opt := range opts {
		opt(p)
	}

	reqOpts := []option.RequestOption{option.WithAPIKey(apiKey)}
	if baseURL != "" {
		reqOpts = append(reqOpts, option.WithBaseURL(baseURL))
	}
	p.client = sdk.NewClient(append(reqOpts, p.reqOpts...)...)
	return p
}

// Stream 发起流式请求，通过 channel 逐步投递事件
func (p *Provider) Stream(ctx context.Context, messages []model.Message, tools []model.ToolDef) (<-chan provider.Event, error) {
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
			toolBuf          strings.Builder
			thinkingBuf      strings.Builder
			signatureBuf     strings.Builder
			currentToolID    string
			currentToolName  string
			currentBlockType string
			redactedData     string
			usage            *model.Usage
		)

		for stream.Next() {
			event := stream.Current()
			switch event.Type {

			case "message_start":
				usage = &model.Usage{InputTokens: int(event.AsMessageStart().Message.Usage.InputTokens)}

			case "content_block_start":
				cb := event.AsContentBlockStart().ContentBlock
				currentBlockType = cb.Type
				switch cb.Type {
				case "tool_use":
					currentToolID = cb.ID
					currentToolName = cb.Name
					toolBuf.Reset()
				case "thinking":
					thinkingBuf.Reset()
					signatureBuf.Reset()
				case "redacted_thinking":
					redactedData = cb.Data
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
				case "thinking_delta":
					text := delta.AsThinkingDelta().Thinking
					thinkingBuf.WriteString(text)
					if !send(provider.Event{Type: provider.EventTypeThinkingDelta, Text: text}) {
						return
					}
				case "signature_delta":
					signatureBuf.WriteString(delta.AsSignatureDelta().Signature)
				}

			case "content_block_stop":
				switch currentBlockType {
				case "tool_use":
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
				case "thinking":
					td := &model.ThinkingData{Text: thinkingBuf.String(), Signature: signatureBuf.String()}
					if !send(provider.Event{Type: provider.EventTypeThinkingDone, Thinking: td}) {
						return
					}
				case "redacted_thinking":
					td := &model.ThinkingData{Redacted: true, Data: redactedData}
					if !send(provider.Event{Type: provider.EventTypeThinkingDone, Thinking: td}) {
						return
					}
				}
				currentBlockType = ""

			case "message_delta":
				u := event.AsMessageDelta().Usage
				if usage == nil {
					usage = &model.Usage{}
				}
				usage.OutputTokens = int(u.OutputTokens)
			}
		}

		if err := stream.Err(); err != nil {
			send(provider.Event{Type: provider.EventTypeError, Err: err})
		} else {
			send(provider.Event{Type: provider.EventTypeDone, Usage: usage})
		}
	}()

	return ch, nil
}

// Complete 发起非流式请求，返回完整内容列表和用量统计
func (p *Provider) Complete(ctx context.Context, messages []model.Message, tools []model.ToolDef) ([]model.Content, *model.Usage, error) {
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
func (p *Provider) buildRequest(messages []model.Message, tools []model.ToolDef) sdk.MessageNewParams {
	msgParams, system := convertMessages(messages)
	req := sdk.MessageNewParams{
		Model:     p.model,
		MaxTokens: p.maxTokens,
		Messages:  msgParams,
		Tools:     convertTools(tools),
	}

	if system != "" {
		req.System = []sdk.TextBlockParam{{Text: string(system)}}
	}
	if p.thinking != nil && p.thinking.BudgetTokens > 0 {
		req.Thinking = sdk.ThinkingConfigParamOfEnabled(p.thinking.BudgetTokens)
	}
	if p.toolChoice != "" {
		switch p.toolChoice {
		case "auto":
			req.ToolChoice = sdk.ToolChoiceUnionParam{OfAuto: &sdk.ToolChoiceAutoParam{}}
		case "none":
			req.ToolChoice = sdk.ToolChoiceUnionParam{OfNone: &sdk.ToolChoiceNoneParam{}}
		case "any", "required":
			req.ToolChoice = sdk.ToolChoiceUnionParam{OfAny: &sdk.ToolChoiceAnyParam{}}
		default:
			req.ToolChoice = sdk.ToolChoiceParamOfTool(p.toolChoice)
		}
	}

	return req
}

// convertMessages 将通用消息历史转换为 Messages API 输入格式
func convertMessages(messages []model.Message) (params []sdk.MessageParam, system model.TextData) {
	// appendMsg 追加消息，自动合并连续同角色消息，
	// 确保输出始终满足 Anthropic API 的 user/assistant 严格交替要求
	appendMsg := func(role sdk.MessageParamRole, blocks []sdk.ContentBlockParamUnion) {
		if len(blocks) == 0 {
			return
		}
		if n := len(params); n > 0 && params[n-1].Role == role {
			params[n-1].Content = append(params[n-1].Content, blocks...)
		} else {
			params = append(params, sdk.MessageParam{Role: role, Content: blocks})
		}
	}

	for _, msg := range messages {
		switch msg.Role {

		case model.MessageRoleSystem:
			for _, c := range msg.Contents {
				if c.Type == model.ContentTypeText {
					if system != "" {
						system += "\n"
					}
					system += c.Text
				}
			}

		case model.MessageRoleUser:
			var blocks []sdk.ContentBlockParamUnion
			for _, c := range msg.Contents {
				switch c.Type {
				case model.ContentTypeText:
					blocks = append(blocks, sdk.NewTextBlock(string(c.Text)))
				case model.ContentTypeImageURL:
					if c.Image != "" {
						blocks = append(blocks, sdk.NewImageBlock(sdk.URLImageSourceParam{URL: string(c.Image)}))
					}
				case model.ContentTypeImageRaw:
					if c.Image != "" {
						blocks = append(blocks, sdk.NewImageBlock(sdk.Base64ImageSourceParam{
							MediaType: sdk.Base64ImageSourceMediaType(c.MediaType),
							Data:      string(c.Image),
						}))
					}
				}
			}
			appendMsg(sdk.MessageParamRoleUser, blocks)

		case model.MessageRoleAssistant:
			var blocks []sdk.ContentBlockParamUnion
			for _, c := range msg.Contents {
				switch c.Type {
				case model.ContentTypeThinking:
					if c.Thinking != nil {
						if c.Thinking.Redacted {
							blocks = append(blocks, sdk.NewRedactedThinkingBlock(c.Thinking.Data))
						} else {
							blocks = append(blocks, sdk.NewThinkingBlock(c.Thinking.Signature, c.Thinking.Text))
						}
					}
				case model.ContentTypeText:
					blocks = append(blocks, sdk.NewTextBlock(string(c.Text)))
				case model.ContentTypeToolCall:
					if c.ToolCall != nil {
						tc := c.ToolCall
						blocks = append(blocks, sdk.NewToolUseBlock(tc.ID, tc.Arguments, tc.Name))
					}
				}
			}
			appendMsg(sdk.MessageParamRoleAssistant, blocks)

		case model.MessageRoleTool:
			var blocks []sdk.ContentBlockParamUnion
			for _, c := range msg.Contents {
				if c.Type == model.ContentTypeToolResult && c.ToolResult != nil {
					tr := c.ToolResult
					blocks = append(blocks, sdk.NewToolResultBlock(tr.CallID, tr.Content, tr.IsError))
				}
			}
			appendMsg(sdk.MessageParamRoleUser, blocks)
		}
	}

	return params, system
}

// convertTools 将通用工具定义转换为 Messages API tool 参数
func convertTools(tools []model.ToolDef) []sdk.ToolUnionParam {
	if len(tools) == 0 {
		return nil
	}

	params := make([]sdk.ToolUnionParam, len(tools))
	for i, t := range tools {
		tp := sdk.ToolParam{
			Name:        t.Name,
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

// extractContents 从 Messages API 响应内容块中提取文本、工具调用和思考内容
func extractContents(blocks []sdk.ContentBlockUnion) ([]model.Content, error) {
	var contents []model.Content
	for _, block := range blocks {
		switch block.Type {
		case "thinking":
			tb := block.AsThinking()
			contents = append(contents, model.Content{
				Type:     model.ContentTypeThinking,
				Thinking: &model.ThinkingData{Text: tb.Thinking, Signature: tb.Signature},
			})
		case "redacted_thinking":
			rb := block.AsRedactedThinking()
			contents = append(contents, model.Content{
				Type:     model.ContentTypeThinking,
				Thinking: &model.ThinkingData{Data: rb.Data, Redacted: true},
			})
		case "text":
			contents = append(contents, model.Content{
				Type: model.ContentTypeText,
				Text: model.TextData(block.AsText().Text),
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
