package openai

import (
	"context"
	"testing"

	"github.com/Ingenimax/agent-sdk-go/pkg/interfaces"
	"github.com/Ingenimax/agent-sdk-go/pkg/logging"
	"github.com/openai/openai-go/v2/responses"
)

func TestRequiresResponsesAPI(t *testing.T) {
	tests := []struct {
		name      string
		model     string
		params    *interfaces.GenerateOptions
		toolCount int
		want      bool
	}{
		{"reasoning model with reasoning and tools", "gpt-5-mini", &interfaces.GenerateOptions{LLMConfig: &interfaces.LLMConfig{Reasoning: "low"}}, 2, true},
		{"reasoning model no reasoning effort", "gpt-5-mini", &interfaces.GenerateOptions{LLMConfig: &interfaces.LLMConfig{}}, 2, false},
		{"reasoning model no tools", "gpt-5-mini", &interfaces.GenerateOptions{LLMConfig: &interfaces.LLMConfig{Reasoning: "high"}}, 0, false},
		{"non-reasoning model with tools", "gpt-4o-mini", &interfaces.GenerateOptions{LLMConfig: &interfaces.LLMConfig{Reasoning: "low"}}, 2, false},
		{"o3 reasoning with tools", "o3-mini", &interfaces.GenerateOptions{LLMConfig: &interfaces.LLMConfig{Reasoning: "medium"}}, 1, true},
		{"file inputs on non-reasoning model no tools", "gpt-4o-mini", &interfaces.GenerateOptions{FileInputs: []interfaces.FileInput{{FileID: "file-1"}}}, 0, true},
		{"file inputs on reasoning model with tools", "gpt-5-mini", &interfaces.GenerateOptions{FileInputs: []interfaces.FileInput{{FileID: "file-1"}}}, 2, true},
		{"code execution without files", "gpt-4o-mini", &interfaces.GenerateOptions{EnableCodeExecution: true}, 0, true},
		{"nil params", "gpt-4o-mini", nil, 0, false},
		{"nil llm config", "gpt-5-mini", &interfaces.GenerateOptions{}, 2, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := requiresResponsesAPI(tt.model, tt.params, tt.toolCount); got != tt.want {
				t.Errorf("requiresResponsesAPI(%q,%+v,%d) = %v, want %v", tt.model, tt.params, tt.toolCount, got, tt.want)
			}
		})
	}
}

func TestValidateOpenAIStreamingOptions(t *testing.T) {
	tests := []struct {
		name            string
		params          *interfaces.GenerateOptions
		useResponsesAPI bool
		wantErr         bool
	}{
		{"no options", &interfaces.GenerateOptions{}, false, false},
		{"nil params", nil, false, false},
		{"responses api flag without files rejected", &interfaces.GenerateOptions{}, true, true},
		{"valid file input allowed", &interfaces.GenerateOptions{FileInputs: []interfaces.FileInput{{FileID: "file-1"}}}, false, false},
		{"valid file input allowed with responses api flag", &interfaces.GenerateOptions{FileInputs: []interfaces.FileInput{{FileID: "file-1"}}}, true, false},
		{"code execution without files allowed", &interfaces.GenerateOptions{EnableCodeExecution: true}, false, false},
		{"file input with no source rejected", &interfaces.GenerateOptions{FileInputs: []interfaces.FileInput{{}}}, false, true},
		{"file input with two sources rejected", &interfaces.GenerateOptions{FileInputs: []interfaces.FileInput{{FileID: "file-1", FileURL: "https://example.com/f.pdf"}}}, false, true},
		{"file data without filename rejected", &interfaces.GenerateOptions{FileInputs: []interfaces.FileInput{{FileData: "data:text/csv;base64,QQ=="}}}, false, true},
		{"code execution with url file rejected", &interfaces.GenerateOptions{EnableCodeExecution: true, FileInputs: []interfaces.FileInput{{FileURL: "https://example.com/f.csv"}}}, false, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateOpenAIStreamingOptions(tt.params, tt.useResponsesAPI)
			if (err != nil) != tt.wantErr {
				t.Errorf("validateOpenAIStreamingOptions() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestBuildResponseInput(t *testing.T) {
	c := &OpenAIClient{logger: logging.New()}

	tests := []struct {
		name     string
		prompt   string
		memory   interfaces.Memory
		expected int
	}{
		{
			name:     "no memory yields single user item",
			prompt:   "Hello",
			memory:   nil,
			expected: 1,
		},
		{
			name:   "user and assistant text preserved",
			prompt: "Continue",
			memory: &mockMemory{messages: []interfaces.Message{
				{Role: interfaces.MessageRoleUser, Content: "Hi"},
				{Role: interfaces.MessageRoleAssistant, Content: "Hello!"},
			}},
			expected: 2,
		},
		{
			name:   "system message preserved",
			prompt: "Continue",
			memory: &mockMemory{messages: []interfaces.Message{
				{Role: interfaces.MessageRoleSystem, Content: "System"},
				{Role: interfaces.MessageRoleUser, Content: "Hi"},
				{Role: interfaces.MessageRoleAssistant, Content: "Hello!"},
			}},
			expected: 3,
		},
		{
			name:   "tool role dropped, assistant text kept",
			prompt: "What's next?",
			memory: &mockMemory{messages: []interfaces.Message{
				{Role: interfaces.MessageRoleUser, Content: "Get weather"},
				{Role: interfaces.MessageRoleAssistant, Content: "I'll check", ToolCalls: []interfaces.ToolCall{
					{ID: "call_1", Name: "get_weather", Arguments: `{}`},
				}},
				{Role: interfaces.MessageRoleTool, Content: "Sunny", ToolCallID: "call_1"},
			}},
			expected: 2,
		},
		{
			name:   "empty assistant content dropped",
			prompt: "x",
			memory: &mockMemory{messages: []interfaces.Message{
				{Role: interfaces.MessageRoleUser, Content: "Hi"},
				{Role: interfaces.MessageRoleAssistant, Content: ""},
			}},
			expected: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			items := c.buildResponseInput(context.Background(), tt.prompt, tt.memory, nil)
			if len(items) != tt.expected {
				t.Errorf("expected %d input items, got %d", tt.expected, len(items))
			}
		})
	}
}

// countInputFileParts returns the number of input_file content parts on the
// item if it is a user message with a content list, or 0 otherwise.
func countInputFileParts(item responses.ResponseInputItemUnionParam) int {
	if item.OfMessage == nil {
		return 0
	}
	count := 0
	for _, part := range item.OfMessage.Content.OfInputItemContentList {
		if part.OfInputFile != nil {
			count++
		}
	}
	return count
}

func TestBuildResponseInputFileAttachment(t *testing.T) {
	c := &OpenAIClient{logger: logging.New()}
	files := []interfaces.FileInput{{FileID: "file-1"}, {FileID: "file-2"}}

	t.Run("no memory attaches files to prompt message", func(t *testing.T) {
		items := c.buildResponseInput(context.Background(), "Analyze this", nil, files)
		if len(items) != 1 {
			t.Fatalf("expected 1 input item, got %d", len(items))
		}
		if got := countInputFileParts(items[0]); got != 2 {
			t.Errorf("expected 2 input_file parts, got %d", got)
		}
	})

	t.Run("no memory no files keeps plain text message", func(t *testing.T) {
		items := c.buildResponseInput(context.Background(), "Hello", nil, nil)
		if len(items) != 1 {
			t.Fatalf("expected 1 input item, got %d", len(items))
		}
		if got := countInputFileParts(items[0]); got != 0 {
			t.Errorf("expected 0 input_file parts, got %d", got)
		}
		if items[0].OfMessage == nil || !items[0].OfMessage.Content.OfString.Valid() {
			t.Error("expected plain string user message when no files are attached")
		}
	})

	t.Run("memory attaches files to last user message", func(t *testing.T) {
		memory := &mockMemory{messages: []interfaces.Message{
			{Role: interfaces.MessageRoleUser, Content: "First question"},
			{Role: interfaces.MessageRoleAssistant, Content: "First answer"},
			{Role: interfaces.MessageRoleUser, Content: "Analyze the file"},
		}}
		items := c.buildResponseInput(context.Background(), "Analyze the file", memory, files)
		if len(items) != 3 {
			t.Fatalf("expected 3 input items, got %d", len(items))
		}
		if got := countInputFileParts(items[0]); got != 0 {
			t.Errorf("expected no input_file parts on first user message, got %d", got)
		}
		if got := countInputFileParts(items[2]); got != 2 {
			t.Errorf("expected 2 input_file parts on last user message, got %d", got)
		}
	})

	t.Run("memory without user message appends prompt with files", func(t *testing.T) {
		memory := &mockMemory{messages: []interfaces.Message{
			{Role: interfaces.MessageRoleSystem, Content: "System"},
		}}
		items := c.buildResponseInput(context.Background(), "Analyze the file", memory, files)
		if len(items) != 2 {
			t.Fatalf("expected 2 input items, got %d", len(items))
		}
		if got := countInputFileParts(items[1]); got != 2 {
			t.Errorf("expected 2 input_file parts on appended message, got %d", got)
		}
	})
}
