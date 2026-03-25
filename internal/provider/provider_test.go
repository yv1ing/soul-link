package provider_test

import (
	"context"
	"testing"
	"time"

	"soul-link/internal/model"
	"soul-link/internal/provider"
	"soul-link/internal/provider/anthropic"
	"soul-link/internal/provider/openai"
	"soul-link/internal/registry"
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

type getWeatherArgs struct {
	City string `json:"city" desc:"城市名称" required:"true"`
}

func getWeather(args getWeatherArgs) (string, error) {
	return `{"temperature":"25°C","condition":"晴"}`, nil
}

func newWeatherRegistry(t *testing.T) *registry.Registry {
	t.Helper()
	r := registry.New()
	if err := r.Register("get_weather", "获取指定城市的当前天气", getWeather); err != nil {
		t.Fatal(err)
	}
	return r
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
	return anthropic.New(anthropicKey, anthropicBase, anthropicModel, 0)
}

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
		r := newWeatherRegistry(t)
		conv := provider.NewConversation(p)
		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer cancel()

		contents, _, err := conv.Complete(ctx, "北京现在天气怎么样？", r.ToolSets())
		if err != nil {
			t.Fatalf("round 1: %v", err)
		}
		tc := findToolCall(t, contents)
		t.Logf("tool call: %s %v", tc.ToolName, tc.Arguments)

		conv.InjectToolResults(r.Execute(*tc))

		contents, _, err = conv.Complete(ctx, "", r.ToolSets())
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
		r := newWeatherRegistry(t)
		conv := provider.NewConversation(p)
		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer cancel()

		ch, err := conv.Stream(ctx, "北京现在天气怎么样？", r.ToolSets())
		if err != nil {
			t.Fatalf("round 1: %v", err)
		}
		tc := drainStream(t, ch)
		if tc == nil {
			t.Fatal("expected tool_call event")
		}

		conv.InjectToolResults(r.Execute(*tc))

		ch, err = conv.Stream(ctx, "", r.ToolSets())
		if err != nil {
			t.Fatalf("round 2: %v", err)
		}
		drainStream(t, ch)
	})
}
