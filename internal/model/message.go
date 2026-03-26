package model

type ContentType string

const (
	ContentTypeText       ContentType = "text"
	ContentTypeImageURL   ContentType = "image_url"
	ContentTypeImageRaw   ContentType = "image_raw"
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

type TextData string
type ImageData string

type ThinkingData struct {
	ID        string `json:"id,omitempty"`        // OpenAI: reasoning item ID
	Text      string `json:"text,omitempty"`      // Anthropic: 完整思考链文本; OpenAI: reasoning summary 摘要
	Data      string `json:"data,omitempty"`      // Anthropic: redacted block 数据
	Signature string `json:"signature,omitempty"` // Anthropic: thinking block 签名; OpenAI 不使用
	Redacted  bool   `json:"redacted,omitempty"`  // 是否为隐匿思考块
}

type Content struct {
	Type ContentType `json:"type"`

	Text       TextData      `json:"text,omitempty"`
	Image      ImageData     `json:"image,omitempty"`      // ContentTypeImageURL: URL; ContentTypeImageRaw: base64 数据
	MediaType  string        `json:"media_type,omitempty"` // ContentTypeImageRaw 时必填，如 "image/png"
	Thinking   *ThinkingData `json:"thinking,omitempty"`
	ToolCall   *ToolCall     `json:"tool_call,omitempty"`
	ToolResult *ToolResult   `json:"tool_result,omitempty"`
}

type Message struct {
	Role MessageRole `json:"role"`

	Contents []Content `json:"contents"`
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
