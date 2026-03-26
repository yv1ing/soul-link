package registry

import (
	"fmt"

	"soul-link/internal/model"
)

type Registry struct {
	tools map[string]*funcTool
	names []string // 保持注册顺序，使 ToolDefs 输出稳定
}

func New() *Registry {
	return &Registry{tools: make(map[string]*funcTool)}
}

func (r *Registry) Register(name, description string, fn any) error {
	ft, err := newFuncTool(name, description, fn)
	if err != nil {
		return err
	}
	if _, exists := r.tools[name]; !exists {
		r.names = append(r.names, name)
	}
	r.tools[name] = ft
	return nil
}

func (r *Registry) ToolDefs() []model.ToolDef {
	defs := make([]model.ToolDef, 0, len(r.names))
	for _, name := range r.names {
		defs = append(defs, r.tools[name].def)
	}
	return defs
}

func (r *Registry) Execute(call model.ToolCall) model.ToolResult {
	ft, ok := r.tools[call.Name]
	if !ok {
		return model.ToolResult{
			CallID:  call.ID,
			IsError: true,
			Content: fmt.Sprintf("unknown tool: %q", call.Name),
		}
	}
	result, err := ft.handler(call.Arguments)
	if err != nil {
		return model.ToolResult{CallID: call.ID, IsError: true, Content: err.Error()}
	}
	return model.ToolResult{CallID: call.ID, Content: result}
}
