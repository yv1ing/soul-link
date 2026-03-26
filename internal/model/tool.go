package model

// ToolDef 描述单个函数工具的定义
type ToolDef struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
}

// ToolCall 表示模型发起的工具调用请求
type ToolCall struct {
	ID        string         `json:"id"`
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

// ToolResult 表示工具执行结果
type ToolResult struct {
	CallID  string `json:"call_id"`
	IsError bool   `json:"is_error"`
	Content string `json:"content"`
}
