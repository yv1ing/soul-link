package provider

import (
	"context"

	"soul-link/internal/model"
)

// 事件类型枚举
type EventType string

const (
	EventTypeTextDelta EventType = "text_delta" // 文本增量
	EventTypeToolCall  EventType = "tool_call"  // 工具调用完成
	EventTypeDone      EventType = "done"       // 结束
	EventTypeError     EventType = "error"      // 错误
)

// 流式事件
type Event struct {
	Type     EventType
	Text     string          // EventTypeTextDelta
	ToolCall *model.ToolCall // EventTypeToolCall
	Usage    *model.Usage    // EventTypeDone
	Err      error           // EventTypeError
}

// Provider 定义统一的 LLM 调用接口
type Provider interface {
	Stream(ctx context.Context, messages []model.Message, tools []model.ToolSet) (<-chan Event, error)
	Complete(ctx context.Context, messages []model.Message, tools []model.ToolSet) ([]model.Content, *model.Usage, error)
}
