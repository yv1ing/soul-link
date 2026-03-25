package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

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
	client   sdk.Client
	model    string
	thinking *model.ThinkingConfig
}

// New 创建 OpenAI Provider
func New(apiKey, baseURL, modelName string, thinking *model.ThinkingConfig, opts ...option.RequestOption) *Provider {
	base := []option.RequestOption{option.WithAPIKey(apiKey)}
	if baseURL != "" {
		base = append(base, option.WithBaseURL(baseURL))
	}

	return &Provider{
		client:   sdk.NewClient(append(base, opts...)...),
		model:    modelName,
		thinking: thinking,
	}
}

// Stream 发起流式请求，通过 channel 逐步投递事件
func (p *Provider) Stream(ctx context.Context, messages []model.Message, tools []model.ToolSet) (<-chan provider.Event, error) {
	req, err := p.buildRequest(messages, tools)
	if err != nil {
		return nil, err
	}
	stream := p.client.Responses.NewStreaming(ctx, req)

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

			case "response.reasoning_summary_text.delta":
				if !send(provider.Event{
					Type: provider.EventTypeThinkingDelta,
					Text: event.AsResponseReasoningSummaryTextDelta().Delta,
				}) {
					return
				}

			case "response.output_item.done":
				item := event.AsResponseOutputItemDone().Item
				switch item.Type {
				case "function_call":
					fc := item.AsFunctionCall()
					tc, err := provider.ParseToolCall(fc.CallID, fc.Name, fc.Arguments)
					if err != nil {
						send(provider.Event{Type: provider.EventTypeError, Err: err})
						return
					}
					if !send(provider.Event{Type: provider.EventTypeToolCall, ToolCall: tc}) {
						return
					}
				case "reasoning":
					ri := item.AsReasoning()
					td := extractReasoningData(ri)
					if !send(provider.Event{Type: provider.EventTypeThinkingDone, Thinking: td}) {
						return
					}
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
	req, err := p.buildRequest(messages, tools)
	if err != nil {
		return nil, nil, err
	}

	resp, err := p.client.Responses.New(ctx, req)
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
func (p *Provider) buildRequest(messages []model.Message, tools []model.ToolSet) (responses.ResponseNewParams, error) {
	input, instructions, err := convertMessages(messages)
	if err != nil {
		return responses.ResponseNewParams{}, err
	}

	req := responses.ResponseNewParams{
		Model: shared.ResponsesModel(p.model),
		Input: responses.ResponseNewParamsInputUnion{OfInputItemList: input},
		Tools: convertTools(tools),
	}

	if instructions != "" {
		req.Instructions = sdk.String(instructions)
	}

	if p.thinking != nil && p.thinking.Effort != "" {
		req.Reasoning = shared.ReasoningParam{
			Effort:  shared.ReasoningEffort(p.thinking.Effort),
			Summary: shared.ReasoningSummaryConcise,
		}
	}

	return req, nil
}

// convertMessages 将通用消息历史转换为 Responses API 输入格式
func convertMessages(messages []model.Message) (responses.ResponseInputParam, string, error) {
	var (
		instructions string
		items        []responses.ResponseInputItemUnionParam
	)

	for _, msg := range messages {
		switch msg.Role {

		case model.MessageRoleSystem:
			for _, c := range msg.Contents {
				if c.Type == model.ContentTypeText {
					if instructions != "" {
						instructions += "\n"
					}
					instructions += c.Text
				}
			}

		case model.MessageRoleUser:
			var text string
			for _, c := range msg.Contents {
				if c.Type == model.ContentTypeText {
					if text != "" {
						text += "\n"
					}
					text += c.Text
				}
			}
			if text != "" {
				items = append(items, responses.ResponseInputItemParamOfMessage(text, responses.EasyInputMessageRoleUser))
			}

		case model.MessageRoleAssistant:
			var pendingText string
			for _, c := range msg.Contents {
				switch c.Type {
				case model.ContentTypeThinking:
					if c.Thinking != nil && c.Thinking.ID != "" {
						if pendingText != "" {
							items = append(items, responses.ResponseInputItemParamOfMessage(pendingText, responses.EasyInputMessageRoleAssistant))
							pendingText = ""
						}
						items = append(items, responses.ResponseInputItemParamOfReasoning(c.Thinking.ID, splitSummaries(c.Thinking.Text)))
					}
				case model.ContentTypeText:
					if pendingText != "" {
						pendingText += "\n"
					}
					pendingText += c.Text
				case model.ContentTypeToolCall:
					if c.ToolCall == nil {
						continue
					}
					if pendingText != "" {
						items = append(items, responses.ResponseInputItemParamOfMessage(pendingText, responses.EasyInputMessageRoleAssistant))
						pendingText = ""
					}
					tc := c.ToolCall
					argsJSON, err := json.Marshal(tc.Arguments)
					if err != nil {
						return nil, "", fmt.Errorf("marshal tool %q args: %w", tc.Name, err)
					}
					items = append(items, responses.ResponseInputItemUnionParam{
						OfFunctionCall: &responses.ResponseFunctionToolCallParam{
							CallID:    tc.ID,
							Name:      tc.Name,
							Arguments: string(argsJSON),
						},
					})
				}
			}

			if pendingText != "" {
				items = append(items, responses.ResponseInputItemParamOfMessage(pendingText, responses.EasyInputMessageRoleAssistant))
			}

		case model.MessageRoleTool:
			for _, c := range msg.Contents {
				if c.Type == model.ContentTypeToolResult && c.ToolResult != nil {
					tr := c.ToolResult
					items = append(items, convertToolResult(tr))
				}
			}
		}
	}

	return responses.ResponseInputParam(items), instructions, nil
}

// convertTools 将通用工具定义转换为 Responses API function tool 参数
func convertTools(tools []model.ToolSet) []responses.ToolUnionParam {
	if len(tools) == 0 {
		return nil
	}

	params := make([]responses.ToolUnionParam, len(tools))
	for i, t := range tools {
		fp := responses.FunctionToolParam{
			Name:        t.Name,
			Parameters:  t.Parameters,
			Description: param.NewOpt(t.Description),
			Strict:      param.NewOpt(false),
		}
		params[i] = responses.ToolUnionParam{OfFunction: &fp}
	}

	return params
}

// convertToolResult 将 ToolResult 转换为 Responses API 的 function_call_output 输入项
func convertToolResult(tr *model.ToolResult) responses.ResponseInputItemUnionParam {
	output := tr.Content
	if tr.IsError {
		output = "[ERROR] " + output
	}

	return responses.ResponseInputItemParamOfFunctionCallOutput(tr.CallID, output)
}

// extractContents 从 Responses API 输出项中提取文本、工具调用和推理内容
func extractContents(items []responses.ResponseOutputItemUnion) ([]model.Content, error) {
	var contents []model.Content
	for _, item := range items {
		switch item.Type {
		case "reasoning":
			ri := item.AsReasoning()
			contents = append(contents, model.Content{
				Type:     model.ContentTypeThinking,
				Thinking: extractReasoningData(ri),
			})
		case "message":
			for _, part := range item.AsMessage().Content {
				if part.Type == "output_text" {
					contents = append(contents, model.Content{
						Type: model.ContentTypeText,
						Text: part.AsOutputText().Text,
					})
				}
			}
		case "function_call":
			fc := item.AsFunctionCall()
			tc, err := provider.ParseToolCall(fc.CallID, fc.Name, fc.Arguments)
			if err != nil {
				return nil, err
			}
			contents = append(contents, model.Content{Type: model.ContentTypeToolCall, ToolCall: tc})
		}
	}

	return contents, nil
}

// extractReasoningData 从 ResponseReasoningItem 中提取推理数据
func extractReasoningData(ri responses.ResponseReasoningItem) *model.ThinkingData {
	var text string
	for _, s := range ri.Summary {
		if text != "" {
			text += "\n"
		}
		text += s.Text
	}

	return &model.ThinkingData{ID: ri.ID, Text: text}
}

// splitSummaries 将文本按行分割为 ResponseReasoningItemSummaryParam 切片
func splitSummaries(text string) []responses.ResponseReasoningItemSummaryParam {
	if text == "" {
		return []responses.ResponseReasoningItemSummaryParam{}
	}

	lines := strings.Split(text, "\n")
	summaries := make([]responses.ResponseReasoningItemSummaryParam, len(lines))
	for i, line := range lines {
		summaries[i] = responses.ResponseReasoningItemSummaryParam{Text: line}
	}

	return summaries
}
