package model

/**
 * 常量定义
 */

// 内容类型枚举
type ContentType string

const (
	ContentTypeTextData   ContentType = "text_data"
	ContentTypeToolCall   ContentType = "tool_call"
	ContentTypeToolResult ContentType = "tool_result"
)

// 消息角色枚举
type MessageRole string

const (
	MessageRoleSystem    MessageRole = "system"
	MessageRoleAssistant MessageRole = "assistant"
	MessageRoleUser      MessageRole = "user"
	MessageRoleTool      MessageRole = "tool"
)

/**
 * 消息定义
 */

// 内容封装
type Content struct {
	Type ContentType `json:"type"`

	TextData   string      `json:"text_data,omitempty"`
	ToolCall   *ToolCall   `json:"tool_call,omitempty"`
	ToolResult *ToolResult `json:"tool_result,omitempty"`
}

// 消息封装
type Message struct {
	Role MessageRole `json:"role"`

	Contents []Content `json:"contents"`
}
