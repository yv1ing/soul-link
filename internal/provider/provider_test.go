package provider_test

import (
	"context"
	"testing"
	"time"

	"soul-link/internal/model"
	"soul-link/internal/provider"
	"soul-link/internal/provider/anthropic"
	"soul-link/internal/provider/openai"
)

const (
	openaiKey      = ""
	openaiBase     = ""
	openaiModel    = ""
	anthropicKey   = ""
	anthropicBase  = ""
	anthropicModel = ""
)

// ── 工具定义 ──────────────────────────────────────────────────────────────────

var weatherTool = model.ToolSet{
	ToolName:    "get_weather",
	Description: "获取指定城市的当前天气",
	Parameters: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"city": map[string]any{
				"type":        "string",
				"description": "城市名称",
			},
		},
		"required": []any{"city"},
	},
}

// ── 辅助函数 ──────────────────────────────────────────────────────────────────

func newOpenAI(t *testing.T) provider.Provider {
	t.Helper()
	if openaiKey == "" {
		t.Skip("openaiKey not set")
	}
	return openai.New(openaiKey, openaiBase, openaiModel)
}

func newAnthropic(t *testing.T) provider.Provider {
	t.Helper()
	if anthropicKey == "" {
		t.Skip("anthropicKey not set")
	}
	return anthropic.New(anthropicKey, anthropicBase, anthropicModel)
}

// eachProvider 对每个已配置的 provider 运行子测试
func eachProvider(t *testing.T, f func(t *testing.T, p provider.Provider)) {
	t.Helper()
	for _, tc := range []struct {
		name string
		new  func(*testing.T) provider.Provider
	}{
		{"openai", newOpenAI},
		{"anthropic", newAnthropic},
	} {
		t.Run(tc.name, func(t *testing.T) { f(t, tc.new(t)) })
	}
}

// findToolCall 从内容列表中返回第一个工具调用，未找到则 Fatal
func findToolCall(t *testing.T, contents []model.Content) *model.ToolCall {
	t.Helper()
	for _, c := range contents {
		if c.Type == model.ContentTypeToolCall {
			return c.ToolCall
		}
	}
	t.Fatal("expected tool call in response")
	return nil
}

// drainStream 消费流式 channel，返回收到的工具调用（若有），遇错误则 Fatal
func drainStream(t *testing.T, ch <-chan provider.Event) *model.ToolCall {
	t.Helper()
	var tc *model.ToolCall
	for e := range ch {
		switch e.Type {
		case provider.EventTypeTextDelta:
			t.Logf("[text_delta] %q", e.Text)
		case provider.EventTypeToolCall:
			tc = e.ToolCall
			t.Logf("[tool_call]  %s %v", e.ToolCall.ToolName, e.ToolCall.Arguments)
		case provider.EventTypeDone:
			if e.Usage != nil {
				t.Logf("[done]       input=%d output=%d", e.Usage.InputTokens, e.Usage.OutputTokens)
			}
		case provider.EventTypeError:
			t.Fatalf("[error] %v", e.Err)
		}
	}
	return tc
}

// ── 测试：多轮历史对话 ─────────────────────────────────────────────────────────

func TestConversation_Complete_MultiTurn(t *testing.T) {
	eachProvider(t, func(t *testing.T, p provider.Provider) {
		conv := provider.NewConversation(p)
		conv.AddSystem("你是一个简洁的助手，每次回复不超过两句话。")
		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer cancel()

		contents, _, err := conv.Complete(ctx, "我叫小明。", nil)
		if err != nil {
			t.Fatalf("round 1: %v", err)
		}
		if len(contents) > 0 {
			t.Logf("round 1: %s", contents[0].TextData)
		}

		contents, _, err = conv.Complete(ctx, "我叫什么名字？", nil)
		if err != nil {
			t.Fatalf("round 2: %v", err)
		}
		if len(contents) > 0 {
			t.Logf("round 2: %s", contents[0].TextData)
		}
	})
}

func TestConversation_Stream_MultiTurn(t *testing.T) {
	eachProvider(t, func(t *testing.T, p provider.Provider) {
		conv := provider.NewConversation(p)
		conv.AddSystem("你是一个简洁的助手，每次回复不超过两句话。")
		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer cancel()

		ch, err := conv.Stream(ctx, "我叫小明。", nil)
		if err != nil {
			t.Fatalf("round 1: %v", err)
		}
		drainStream(t, ch)

		ch, err = conv.Stream(ctx, "我叫什么名字？", nil)
		if err != nil {
			t.Fatalf("round 2: %v", err)
		}
		drainStream(t, ch)
	})
}

// ── 测试：工具调用 ─────────────────────────────────────────────────────────────

func TestConversation_Complete_ToolCall(t *testing.T) {
	eachProvider(t, func(t *testing.T, p provider.Provider) {
		conv := provider.NewConversation(p)
		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer cancel()

		contents, _, err := conv.Complete(ctx, "北京现在天气怎么样？", []model.ToolSet{weatherTool})
		if err != nil {
			t.Fatalf("round 1: %v", err)
		}
		tc := findToolCall(t, contents)
		t.Logf("tool call: %s %v", tc.ToolName, tc.Arguments)

		conv.InjectToolResults(model.ToolResult{
			CallID:  tc.ToolID,
			Content: `{"temperature":"25°C","condition":"晴"}`,
		})

		contents, _, err = conv.Complete(ctx, "", []model.ToolSet{weatherTool})
		if err != nil {
			t.Fatalf("round 2: %v", err)
		}
		if len(contents) > 0 {
			t.Logf("final reply: %s", contents[0].TextData)
		}
	})
}

func TestConversation_Stream_ToolCall(t *testing.T) {
	eachProvider(t, func(t *testing.T, p provider.Provider) {
		conv := provider.NewConversation(p)
		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer cancel()

		ch, err := conv.Stream(ctx, "北京现在天气怎么样？", []model.ToolSet{weatherTool})
		if err != nil {
			t.Fatalf("round 1: %v", err)
		}
		tc := drainStream(t, ch)
		if tc == nil {
			t.Fatal("expected tool_call event")
		}

		conv.InjectToolResults(model.ToolResult{
			CallID:  tc.ToolID,
			Content: `{"temperature":"25°C","condition":"晴"}`,
		})

		ch, err = conv.Stream(ctx, "", []model.ToolSet{weatherTool})
		if err != nil {
			t.Fatalf("round 2: %v", err)
		}
		drainStream(t, ch)
	})
}
