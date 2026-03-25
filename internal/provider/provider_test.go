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

// ── 辅助函数 ──────────────────────────────────────────────────────────────────

func eachProvider(t *testing.T, f func(t *testing.T, p provider.Provider)) {
	t.Helper()
	providers := []struct {
		name string
		new  func() provider.Provider
		skip func() bool
	}{
		{
			name: "openai",
			new:  func() provider.Provider { return openai.New(openaiKey, openaiBase, openaiModel, nil) },
			skip: func() bool { return openaiKey == "" },
		},
		{
			name: "anthropic",
			new:  func() provider.Provider { return anthropic.New(anthropicKey, anthropicBase, anthropicModel, 0, nil) },
			skip: func() bool { return anthropicKey == "" },
		},
		{
			name: "openai/thinking",
			new: func() provider.Provider {
				return openai.New(openaiKey, openaiBase, openaiModel, &model.ThinkingConfig{Effort: "high"})
			},
			skip: func() bool { return openaiKey == "" },
		},
		{
			name: "anthropic/thinking",
			new: func() provider.Provider {
				return anthropic.New(anthropicKey, anthropicBase, anthropicModel, 16000, &model.ThinkingConfig{BudgetTokens: 5000})
			},
			skip: func() bool { return anthropicKey == "" },
		},
	}

	for _, tc := range providers {
		t.Run(tc.name, func(t *testing.T) {
			if tc.skip() {
				t.Skipf("%s key not set", tc.name)
			}
			f(t, tc.new())
		})
	}
}

func drainStream(t *testing.T, ch <-chan provider.Event) {
	t.Helper()
	for e := range ch {
		switch e.Type {
		case provider.EventTypeThinkingDelta:
			t.Logf("[thinking_delta] %q", e.Text)
		case provider.EventTypeThinkingDone:
			t.Logf("[thinking_done]  redacted=%v", e.Thinking.Redacted)
		case provider.EventTypeTextDelta:
			t.Logf("[text_delta] %q", e.Text)
		case provider.EventTypeToolCall:
			t.Logf("[tool_call]  %s %v", e.ToolCall.Name, e.ToolCall.Arguments)
		case provider.EventTypeDone:
			if e.Usage != nil {
				t.Logf("[done]       input=%d output=%d", e.Usage.InputTokens, e.Usage.OutputTokens)
			}
		case provider.EventTypeError:
			t.Fatalf("[error] %v", e.Err)
		}
	}
}

func findContent(contents []model.Content, ct model.ContentType) *model.Content {
	for i := range contents {
		if contents[i].Type == ct {
			return &contents[i]
		}
	}
	return nil
}

func newWeatherRegistry(t *testing.T) *registry.Registry {
	t.Helper()
	r := registry.New()
	if err := r.Register("get_weather", "获取指定城市的当前天气", func(args struct {
		City string `json:"city" desc:"城市名称" required:"true"`
	}) (string, error) {
		return `{"temperature":"25°C","condition":"晴"}`, nil
	}); err != nil {
		t.Fatal(err)
	}
	return r
}

// ── 测试：多轮对话 ──────────────────────────────────────────────────────────────

func TestConversation_Complete_MultiTurn(t *testing.T) {
	eachProvider(t, func(t *testing.T, p provider.Provider) {
		conv := provider.NewConversation(p)
		conv.AddSystem("你是一个简洁的助手，每次回复不超过两句话。")
		ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
		defer cancel()

		contents, _, err := conv.Complete(ctx, "我叫小明。", nil)
		if err != nil {
			t.Fatalf("round 1: %v", err)
		}
		if c := findContent(contents, model.ContentTypeThinking); c != nil {
			t.Logf("round 1 [thinking] redacted=%v text=%q", c.Thinking.Redacted, c.Thinking.Text)
		}
		if c := findContent(contents, model.ContentTypeText); c != nil {
			t.Logf("round 1: %s", c.Text)
		}

		contents, _, err = conv.Complete(ctx, "我叫什么名字？", nil)
		if err != nil {
			t.Fatalf("round 2: %v", err)
		}
		if c := findContent(contents, model.ContentTypeThinking); c != nil {
			t.Logf("round 2 [thinking] redacted=%v text=%q", c.Thinking.Redacted, c.Thinking.Text)
		}
		if c := findContent(contents, model.ContentTypeText); c != nil {
			t.Logf("round 2: %s", c.Text)
		}
	})
}

func TestConversation_Stream_MultiTurn(t *testing.T) {
	eachProvider(t, func(t *testing.T, p provider.Provider) {
		conv := provider.NewConversation(p)
		conv.AddSystem("你是一个简洁的助手，每次回复不超过两句话。")
		ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
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

// ── 测试：工具调用 ──────────────────────────────────────────────────

func TestConversation_Complete_ToolCall(t *testing.T) {
	eachProvider(t, func(t *testing.T, p provider.Provider) {
		r := newWeatherRegistry(t)
		conv := provider.NewConversation(p, provider.WithToolExecutor(r))
		ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
		defer cancel()

		contents, _, err := conv.Complete(ctx, "北京现在天气怎么样？", r.ToolSets())
		if err != nil {
			t.Fatalf("complete: %v", err)
		}
		if c := findContent(contents, model.ContentTypeThinking); c != nil {
			t.Logf("[thinking] redacted=%v text=%q", c.Thinking.Redacted, c.Thinking.Text)
		}
		c := findContent(contents, model.ContentTypeText)
		if c == nil {
			t.Fatal("expected text reply after tool loop")
		}
		t.Logf("reply: %s", c.Text)
	})
}

func TestConversation_Stream_ToolCall(t *testing.T) {
	eachProvider(t, func(t *testing.T, p provider.Provider) {
		r := newWeatherRegistry(t)
		conv := provider.NewConversation(p, provider.WithToolExecutor(r))
		ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
		defer cancel()

		ch, err := conv.Stream(ctx, "北京现在天气怎么样？", r.ToolSets())
		if err != nil {
			t.Fatalf("stream: %v", err)
		}
		drainStream(t, ch)
	})
}
