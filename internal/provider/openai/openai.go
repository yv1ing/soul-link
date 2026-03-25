package openai

import (
	"context"
	"encoding/json"
	"fmt"

	sdk "github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/responses"
	"github.com/openai/openai-go/shared"

	"soul-link/internal/model"
	"soul-link/internal/provider"
)

// Provider 封装 OpenAI Responses API 客户端
type Provider struct {
	client sdk.Client
	model  string
}

// New 创建 OpenAI Provider，baseURL 为空时使用 SDK 默认地址
func New(apiKey, baseURL, modelName string, opts ...option.RequestOption) *Provider {
	base := []option.RequestOption{option.WithAPIKey(apiKey)}
	if baseURL != "" {
		base = append(base, option.WithBaseURL(baseURL))
	}
	opts = append(base, opts...)
	return &Provider{
		client: sdk.NewClient(opts...),
		model:  modelName,
	}
}

// Stream 发起流式请求，通过 channel 逐步投递事件，ctx 取消时 goroutine 安全退出
func (p *Provider) Stream(ctx context.Context, messages []model.Message, tools []model.ToolSet) (<-chan provider.Event, error) {
	stream := p.client.Responses.NewStreaming(ctx, p.buildRequest(messages, tools))

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

		for stream.Next() {
			event := stream.Current()
			switch event.Type {

			case "response.output_text.delta":
				if !send(provider.Event{
					Type: provider.EventTypeTextDelta,
					Text: event.AsResponseOutputTextDelta().Delta,
				}) {
					return
				}

			case "response.output_item.done":
				item := event.AsResponseOutputItemDone().Item
				if item.Type != "function_call" {
					continue
				}

				fc := item.AsFunctionCall()
				tc, err := parseToolCall(fc.CallID, fc.Name, fc.Arguments)
				if err != nil {
					send(provider.Event{Type: provider.EventTypeError, Err: err})
					return
				}
				if !send(provider.Event{Type: provider.EventTypeToolCall, ToolCall: tc}) {
					return
				}

			case "response.completed":
				u := event.AsResponseCompleted().Response.Usage
				if !send(provider.Event{
					Type:  provider.EventTypeDone,
					Usage: &model.Usage{InputTokens: int(u.InputTokens), OutputTokens: int(u.OutputTokens)},
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
	resp, err := p.client.Responses.New(ctx, p.buildRequest(messages, tools))
	if err != nil {
		return nil, nil, err
	}

	contents, err := extractContents(resp.Output)
	if err != nil {
		return nil, nil, err
	}

	usage := &model.Usage{InputTokens: int(resp.Usage.InputTokens), OutputTokens: int(resp.Usage.OutputTokens)}
	return contents, usage, nil
}

// buildRequest 将通用消息和工具定义组装为 Responses API 请求参数
func (p *Provider) buildRequest(messages []model.Message, tools []model.ToolSet) responses.ResponseNewParams {
	input, instructions := convertMessages(messages)
	req := responses.ResponseNewParams{
		Model: shared.ResponsesModel(p.model),
		Input: responses.ResponseNewParamsInputUnion{OfInputItemList: input},
		Tools: convertTools(tools),
	}

	if instructions != "" {
		req.Instructions = sdk.String(instructions)
	}
	return req
}

// convertMessages 将通用消息历史转换为 Responses API 输入格式
func convertMessages(messages []model.Message) (responses.ResponseInputParam, string) {
	var (
		instructions string
		items        []responses.ResponseInputItemUnionParam
	)
	for _, msg := range messages {
		switch msg.Role {

		case model.MessageRoleSystem:
			for _, c := range msg.Contents {
				if c.Type == model.ContentTypeTextData {
					if instructions != "" {
						instructions += "\n"
					}
					instructions += c.TextData
				}
			}

		case model.MessageRoleUser:
			var (
				text      string
				toolItems []responses.ResponseInputItemUnionParam
			)
			for _, c := range msg.Contents {
				switch c.Type {
				case model.ContentTypeTextData:
					if text != "" {
						text += "\n"
					}
					text += c.TextData
				case model.ContentTypeToolResult:
					if c.ToolResult != nil {
						tr := c.ToolResult
						toolItems = append(toolItems, responses.ResponseInputItemParamOfFunctionCallOutput(tr.CallID, tr.Content))
					}
				}
			}
			if text != "" {
				items = append(items, responses.ResponseInputItemParamOfMessage(text, responses.EasyInputMessageRoleUser))
			}
			items = append(items, toolItems...)

		case model.MessageRoleAssistant:
			var (
				text      string
				toolItems []responses.ResponseInputItemUnionParam
			)
			for _, c := range msg.Contents {
				switch c.Type {
				case model.ContentTypeTextData:
					if text != "" {
						text += "\n"
					}
					text += c.TextData
				case model.ContentTypeToolCall:
					if c.ToolCall == nil {
						continue
					}
					tc := c.ToolCall

					argsJSON, err := json.Marshal(tc.Arguments)
					if err != nil {
						continue
					}
					toolItems = append(toolItems, responses.ResponseInputItemUnionParam{
						OfFunctionCall: &responses.ResponseFunctionToolCallParam{
							CallID:    tc.ToolID,
							Name:      tc.ToolName,
							Arguments: string(argsJSON),
						},
					})
				}
			}

			if text != "" {
				items = append(items, responses.ResponseInputItemParamOfMessage(text, responses.EasyInputMessageRoleAssistant))
			}
			items = append(items, toolItems...)

		case model.MessageRoleTool:
			for _, c := range msg.Contents {
				if c.Type == model.ContentTypeToolResult && c.ToolResult != nil {
					tr := c.ToolResult
					items = append(items, responses.ResponseInputItemParamOfFunctionCallOutput(tr.CallID, tr.Content))
				}
			}
		}
	}
	return responses.ResponseInputParam(items), instructions
}

// convertTools 将通用工具定义转换为 Responses API function tool 参数
func convertTools(tools []model.ToolSet) []responses.ToolUnionParam {
	if len(tools) == 0 {
		return nil
	}

	params := make([]responses.ToolUnionParam, len(tools))
	for i, t := range tools {
		fp := responses.FunctionToolParam{
			Name:        t.ToolName,
			Parameters:  t.Parameters,
			Description: param.NewOpt(t.Description),
			Strict:      param.NewOpt(false),
		}
		params[i] = responses.ToolUnionParam{OfFunction: &fp}
	}
	return params
}

// extractContents 从 Responses API 输出项中提取文本和工具调用内容
func extractContents(items []responses.ResponseOutputItemUnion) ([]model.Content, error) {
	var contents []model.Content
	for _, item := range items {
		switch item.Type {
		case "message":
			for _, part := range item.AsMessage().Content {
				if part.Type == "output_text" {
					contents = append(contents, model.Content{
						Type:     model.ContentTypeTextData,
						TextData: part.AsOutputText().Text,
					})
				}
			}
		case "function_call":
			fc := item.AsFunctionCall()
			tc, err := parseToolCall(fc.CallID, fc.Name, fc.Arguments)
			if err != nil {
				return nil, err
			}
			contents = append(contents, model.Content{Type: model.ContentTypeToolCall, ToolCall: tc})
		}
	}
	return contents, nil
}

// parseToolCall 将工具调用的 JSON 字符串参数解析为结构化 ToolCall
func parseToolCall(id, name, rawJSON string) (*model.ToolCall, error) {
	if rawJSON == "" {
		rawJSON = "{}"
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(rawJSON), &args); err != nil {
		return nil, fmt.Errorf("parse tool %q args: %w", name, err)
	}
	return &model.ToolCall{ToolID: id, ToolName: name, Arguments: args}, nil
}
