package model

/**
 * 工具定义
 */

// 函数工具定义
type ToolSet struct {
	Description string         `json:"description"`
	ToolName    string         `json:"tool_name"`
	Parameters  map[string]any `json:"parameters"`
}

// 工具调用请求
type ToolCall struct {
	ToolID    string         `json:"tool_id"`
	ToolName  string         `json:"tool_name"`
	Arguments map[string]any `json:"arguments"`
}

// 工具执行结果
type ToolResult struct {
	CallID  string `json:"call_id"`
	IsError bool   `json:"is_error"`
	Content string `json:"content"`
}
