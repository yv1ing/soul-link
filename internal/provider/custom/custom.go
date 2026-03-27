package custom

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	sdk "github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/shared"

	"soul-link/internal/model"
	"soul-link/internal/provider"
)

// Provider 封装 OpenAI Chat Completions API 客户端
type Provider struct {
	client     sdk.Client
	model      string
	thinking   *model.ThinkingConfig
	maxTokens  int64
	toolChoice string
	reqOpts    []option.RequestOption
}

type Option func(*Provider)

func WithMaxTokens(n int64) Option                        { return func(p *Provider) { p.maxTokens = n } }
func WithToolChoice(choice string) Option                 { return func(p *Provider) { p.toolChoice = choice } }
func WithRequestOptions(opts ...option.RequestOption) Option { return func(p *Provider) { p.reqOpts = append(p.reqOpts, opts...) } }

// New 创建 Custom Provider
func New(apiKey, baseURL, modelName string, thinking *model.ThinkingConfig, opts ...Option) *Provider {
	p := &Provider{model: modelName, thinking: thinking}
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
	req, err := p.buildRequest(messages, tools)
	if err != nil {
		return nil, err
	}
	req.StreamOptions = sdk.ChatCompletionStreamOptionsParam{
		IncludeUsage: param.NewOpt(true),
	}
	stream := p.client.Chat.Completions.NewStreaming(ctx, req)

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

		type pendingToolCall struct {
			id   string
			name string
			args strings.Builder
		}
		var (
			toolCalls []pendingToolCall
			usage     *model.Usage
		)

		for stream.Next() {
			chunk := stream.Current()

			if len(chunk.Choices) == 0 {
				u := chunk.Usage
				if u.PromptTokens > 0 || u.CompletionTokens > 0 {
					usage = &model.Usage{InputTokens: int(u.PromptTokens), OutputTokens: int(u.CompletionTokens)}
				}
				continue
			}

			delta := chunk.Choices[0].Delta

			if delta.Content != "" {
				if !send(provider.Event{Type: provider.EventTypeTextDelta, Text: delta.Content}) {
					return
				}
			}

			for _, tc := range delta.ToolCalls {
				idx := int(tc.Index)
				for len(toolCalls) <= idx {
					toolCalls = append(toolCalls, pendingToolCall{})
				}
				if tc.ID != "" {
					toolCalls[idx].id = tc.ID
				}
				if tc.Function.Name != "" {
					toolCalls[idx].name = tc.Function.Name
				}
				toolCalls[idx].args.WriteString(tc.Function.Arguments)
			}

			if chunk.Choices[0].FinishReason == "tool_calls" {
				for _, pending := range toolCalls {
					tc, err := provider.ParseToolCall(pending.id, pending.name, pending.args.String())
					if err != nil {
						send(provider.Event{Type: provider.EventTypeError, Err: err})
						return
					}
					if !send(provider.Event{Type: provider.EventTypeToolCall, ToolCall: tc}) {
						return
					}
				}
				toolCalls = nil
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
	req, err := p.buildRequest(messages, tools)
	if err != nil {
		return nil, nil, err
	}

	resp, err := p.client.Chat.Completions.New(ctx, req)
	if err != nil {
		return nil, nil, err
	}

	contents, err := extractContents(resp)
	if err != nil {
		return nil, nil, err
	}

	usage := &model.Usage{InputTokens: int(resp.Usage.PromptTokens), OutputTokens: int(resp.Usage.CompletionTokens)}
	return contents, usage, nil
}

// buildRequest 将通用消息和工具定义组装为 Chat Completions API 请求参数
func (p *Provider) buildRequest(messages []model.Message, tools []model.ToolDef) (sdk.ChatCompletionNewParams, error) {
	msgParams, system, err := convertMessages(messages)
	if err != nil {
		return sdk.ChatCompletionNewParams{}, err
	}

	if system != "" {
		sysMsg := sdk.ChatCompletionMessageParamUnion{
			OfSystem: &sdk.ChatCompletionSystemMessageParam{
				Content: sdk.ChatCompletionSystemMessageParamContentUnion{
					OfString: param.NewOpt(string(system)),
				},
			},
		}
		msgParams = append([]sdk.ChatCompletionMessageParamUnion{sysMsg}, msgParams...)
	}

	req := sdk.ChatCompletionNewParams{
		Model:    shared.ChatModel(p.model),
		Messages: msgParams,
		Tools:    convertTools(tools),
	}

	if p.thinking != nil && p.thinking.Effort != "" {
		req.ReasoningEffort = shared.ReasoningEffort(p.thinking.Effort)
	}
	if p.maxTokens > 0 {
		req.MaxTokens = param.NewOpt(p.maxTokens)
	}
	if p.toolChoice != "" {
		switch p.toolChoice {
		case "auto", "none", "required":
			req.ToolChoice = sdk.ChatCompletionToolChoiceOptionUnionParam{
				OfAuto: param.NewOpt(p.toolChoice),
			}
		default:
			req.ToolChoice = sdk.ChatCompletionToolChoiceOptionParamOfChatCompletionNamedToolChoice(
				sdk.ChatCompletionNamedToolChoiceFunctionParam{Name: p.toolChoice},
			)
		}
	}

	return req, nil
}

// convertMessages 将通用消息历史转换为 Chat Completions API 输入格式
func convertMessages(messages []model.Message) ([]sdk.ChatCompletionMessageParamUnion, model.TextData, error) {
	var (
		system model.TextData
		params []sdk.ChatCompletionMessageParamUnion
	)

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
			var parts []sdk.ChatCompletionContentPartUnionParam
			for _, c := range msg.Contents {
				switch c.Type {
				case model.ContentTypeText:
					parts = append(parts, sdk.ChatCompletionContentPartUnionParam{
						OfText: &sdk.ChatCompletionContentPartTextParam{Text: string(c.Text)},
					})
				case model.ContentTypeImageURL:
					if c.Image != "" {
						parts = append(parts, sdk.ChatCompletionContentPartUnionParam{
							OfImageURL: &sdk.ChatCompletionContentPartImageParam{
								ImageURL: sdk.ChatCompletionContentPartImageImageURLParam{URL: string(c.Image)},
							},
						})
					}
				case model.ContentTypeImageRaw:
					if c.Image != "" {
						dataURI := "data:" + c.MediaType + ";base64," + string(c.Image)
						parts = append(parts, sdk.ChatCompletionContentPartUnionParam{
							OfImageURL: &sdk.ChatCompletionContentPartImageParam{
								ImageURL: sdk.ChatCompletionContentPartImageImageURLParam{URL: dataURI},
							},
						})
					}
				}
			}
			if len(parts) > 0 {
				params = append(params, sdk.ChatCompletionMessageParamUnion{
					OfUser: &sdk.ChatCompletionUserMessageParam{
						Content: sdk.ChatCompletionUserMessageParamContentUnion{
							OfArrayOfContentParts: parts,
						},
					},
				})
			}

		case model.MessageRoleAssistant:
			var (
				pendingText model.TextData
				toolCalls   []sdk.ChatCompletionMessageToolCallParam
			)
			for _, c := range msg.Contents {
				switch c.Type {
				case model.ContentTypeText:
					if pendingText != "" {
						pendingText += "\n"
					}
					pendingText += c.Text
				case model.ContentTypeToolCall:
					if c.ToolCall == nil {
						continue
					}
					tc := c.ToolCall
					argsJSON, err := json.Marshal(tc.Arguments)
					if err != nil {
						return nil, "", fmt.Errorf("marshal tool %q args: %w", tc.Name, err)
					}
					toolCalls = append(toolCalls, sdk.ChatCompletionMessageToolCallParam{
						ID: tc.ID,
						Function: sdk.ChatCompletionMessageToolCallFunctionParam{
							Name:      tc.Name,
							Arguments: string(argsJSON),
						},
					})
				}
			}
			if pendingText != "" || len(toolCalls) > 0 {
				asst := &sdk.ChatCompletionAssistantMessageParam{
					ToolCalls: toolCalls,
				}
				if pendingText != "" {
					asst.Content = sdk.ChatCompletionAssistantMessageParamContentUnion{
						OfString: param.NewOpt(string(pendingText)),
					}
				}
				params = append(params, sdk.ChatCompletionMessageParamUnion{OfAssistant: asst})
			}

		case model.MessageRoleTool:
			for _, c := range msg.Contents {
				if c.Type == model.ContentTypeToolResult && c.ToolResult != nil {
					tr := c.ToolResult
					output := tr.Content
					if tr.IsError {
						output = "[ERROR] " + output
					}
					params = append(params, sdk.ChatCompletionMessageParamUnion{
						OfTool: &sdk.ChatCompletionToolMessageParam{
							ToolCallID: tr.CallID,
							Content: sdk.ChatCompletionToolMessageParamContentUnion{
								OfString: param.NewOpt(output),
							},
						},
					})
				}
			}
		}
	}

	return params, system, nil
}

// convertTools 将通用工具定义转换为 Chat Completions API function tool 参数
func convertTools(tools []model.ToolDef) []sdk.ChatCompletionToolParam {
	if len(tools) == 0 {
		return nil
	}

	params := make([]sdk.ChatCompletionToolParam, len(tools))
	for i, t := range tools {
		params[i] = sdk.ChatCompletionToolParam{
			Function: shared.FunctionDefinitionParam{
				Name:        t.Name,
				Description: param.NewOpt(t.Description),
				Parameters:  shared.FunctionParameters(t.Parameters),
			},
		}
	}
	return params
}

// extractContents 从 Chat Completions API 响应中提取文本和工具调用内容
func extractContents(resp *sdk.ChatCompletion) ([]model.Content, error) {
	if len(resp.Choices) == 0 {
		return nil, nil
	}

	msg := resp.Choices[0].Message
	var contents []model.Content

	if msg.Content != "" {
		contents = append(contents, model.Content{
			Type: model.ContentTypeText,
			Text: model.TextData(msg.Content),
		})
	}

	for _, tc := range msg.ToolCalls {
		parsed, err := provider.ParseToolCall(tc.ID, tc.Function.Name, tc.Function.Arguments)
		if err != nil {
			return nil, err
		}
		contents = append(contents, model.Content{Type: model.ContentTypeToolCall, ToolCall: parsed})
	}

	return contents, nil
}
