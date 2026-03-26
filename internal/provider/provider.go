package provider

import (
	"context"
	"encoding/json"
	"fmt"

	"soul-link/internal/model"
)

type EventType string

const (
	EventTypeTextDelta     EventType = "text_delta"     // 文本增量
	EventTypeThinkingDelta EventType = "thinking_delta" // 思考文本增量
	EventTypeThinkingDone  EventType = "thinking_done"  // 思考块完成
	EventTypeToolCall      EventType = "tool_call"      // 工具调用完成
	EventTypeDone          EventType = "done"           // 结束
	EventTypeError         EventType = "error"          // 错误
)

type Event struct {
	Type     EventType
	Text     string              // EventTypeTextDelta / EventTypeThinkingDelta
	Thinking *model.ThinkingData // EventTypeThinkingDone
	ToolCall *model.ToolCall     // EventTypeToolCall
	Usage    *model.Usage        // EventTypeDone
	Err      error               // EventTypeError
}

// Provider 定义统一的 LLM 调用接口
type Provider interface {
	Stream(ctx context.Context, messages []model.Message, tools []model.ToolDef) (<-chan Event, error)
	Complete(ctx context.Context, messages []model.Message, tools []model.ToolDef) ([]model.Content, *model.Usage, error)
}

// ParseToolCall 将工具调用的 JSON 字符串参数解析为结构化 ToolCall，供各 Provider 实现共用
func ParseToolCall(id, name, rawJSON string) (*model.ToolCall, error) {
	if rawJSON == "" {
		rawJSON = "{}"
	}

	var args map[string]any
	if err := json.Unmarshal([]byte(rawJSON), &args); err != nil {
		return nil, fmt.Errorf("parse tool %q args: %w", name, err)
	}

	return &model.ToolCall{ID: id, Name: name, Arguments: args}, nil
}
