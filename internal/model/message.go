package model

type ContentType string

const (
	ContentTypeText       ContentType = "text"
	ContentTypeToolCall   ContentType = "tool_call"
	ContentTypeToolResult ContentType = "tool_result"
	ContentTypeThinking   ContentType = "thinking"
)

type MessageRole string

const (
	MessageRoleSystem    MessageRole = "system"
	MessageRoleAssistant MessageRole = "assistant"
	MessageRoleUser      MessageRole = "user"
	MessageRoleTool      MessageRole = "tool"
)

type Content struct {
	Type ContentType `json:"type"`

	Text       string        `json:"text,omitempty"`
	ToolCall   *ToolCall     `json:"tool_call,omitempty"`
	ToolResult *ToolResult   `json:"tool_result,omitempty"`
	Thinking   *ThinkingData `json:"thinking,omitempty"`
}

type ThinkingData struct {
	ID        string `json:"id,omitempty"`        // OpenAI: reasoning item ID
	Text      string `json:"text,omitempty"`      // 思考/推理文本
	Data      string `json:"data,omitempty"`      // Anthropic: redacted block 数据
	Signature string `json:"signature,omitempty"` // Anthropic: thinking block 签名
	Redacted  bool   `json:"redacted,omitempty"`  // 是否为隐匿思考块
}

// Usage 用量统计
type Usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// ThinkingConfig 思考/推理配置
type ThinkingConfig struct {
	BudgetTokens int64  `json:"budget_tokens,omitempty"` // Anthropic: ≥1024; OpenAI: 忽略
	Effort       string `json:"effort,omitempty"`        // OpenAI: "low"/"medium"/"high"; Anthropic: 忽略
}

type Message struct {
	Role MessageRole `json:"role"`

	Contents []Content `json:"contents"`
}
