package registry

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"

	"soul-link/internal/model"
)

type funcTool struct {
	set     model.ToolSet
	handler func(map[string]any) (string, error)
}

var errorType = reflect.TypeOf((*error)(nil)).Elem()

func newFuncTool(name, description string, fn any) (*funcTool, error) {
	fnType := reflect.TypeOf(fn)
	if fnType == nil || fnType.Kind() != reflect.Func {
		return nil, fmt.Errorf("registry: fn must be a function")
	}
	if fnType.NumIn() != 1 || fnType.In(0).Kind() != reflect.Struct {
		return nil, fmt.Errorf("registry: fn must accept exactly one struct argument")
	}
	if fnType.NumOut() != 2 || fnType.Out(0).Kind() != reflect.String || !fnType.Out(1).Implements(errorType) {
		return nil, fmt.Errorf("registry: fn must return (string, error)")
	}

	argType := fnType.In(0)
	props, required := schemaFromStruct(argType)

	params := map[string]any{
		"type":       "object",
		"properties": props,
	}
	if len(required) > 0 {
		params["required"] = required
	}

	fnVal := reflect.ValueOf(fn)
	return &funcTool{
		set: model.ToolSet{
			ToolName:    name,
			Description: description,
			Parameters:  params,
		},
		handler: func(args map[string]any) (string, error) {
			raw, err := json.Marshal(args)
			if err != nil {
				return "", fmt.Errorf("tool %q: marshal args: %w", name, err)
			}
			argPtr := reflect.New(argType)
			if err := json.Unmarshal(raw, argPtr.Interface()); err != nil {
				return "", fmt.Errorf("tool %q: unmarshal args: %w", name, err)
			}
			out := fnVal.Call([]reflect.Value{argPtr.Elem()})
			if !out[1].IsNil() {
				return "", out[1].Interface().(error)
			}
			return out[0].String(), nil
		},
	}, nil
}

func schemaFromStruct(t reflect.Type) (map[string]any, []any) {
	props := make(map[string]any)
	var required []any

	for i := range t.NumField() {
		field := t.Field(i)

		name := strings.Split(field.Tag.Get("json"), ",")[0]
		if name == "" || name == "-" {
			name = strings.ToLower(field.Name)
		}

		prop := map[string]any{"type": jsonType(field.Type)}
		if desc := field.Tag.Get("desc"); desc != "" {
			prop["description"] = desc
		}
		if enum := field.Tag.Get("enum"); enum != "" {
			parts := strings.Split(enum, ",")
			vals := make([]any, len(parts))
			for i, p := range parts {
				vals[i] = strings.TrimSpace(p)
			}
			prop["enum"] = vals
		}

		props[name] = prop
		if field.Tag.Get("required") == "true" {
			required = append(required, name)
		}
	}

	return props, required
}

func jsonType(t reflect.Type) string {
	switch t.Kind() {
	case reflect.String:
		return "string"
	case reflect.Bool:
		return "boolean"
	case reflect.Float32, reflect.Float64:
		return "number"
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return "integer"
	case reflect.Slice, reflect.Array:
		return "array"
	default:
		return "object"
	}
}
