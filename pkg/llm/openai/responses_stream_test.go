package openai_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Ingenimax/agent-sdk-go/pkg/interfaces"
	openai_client "github.com/Ingenimax/agent-sdk-go/pkg/llm/openai"
	"github.com/Ingenimax/agent-sdk-go/pkg/logging"
)

// newResponsesSSEServer returns a test server that answers POST /responses with
// a minimal SSE stream (one text delta, then response.completed) and records
// each decoded request body.
func newResponsesSSEServer(t *testing.T, requests *[]map[string]interface{}) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/responses" {
			t.Errorf("expected /responses endpoint, got %s", r.URL.Path)
			http.Error(w, "wrong endpoint", http.StatusNotFound)
			return
		}
		var reqBody map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&reqBody); err != nil {
			t.Errorf("failed to decode request body: %v", err)
			http.Error(w, "bad body", http.StatusBadRequest)
			return
		}
		*requests = append(*requests, reqBody)

		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)
		writeEvent := func(event, data string) {
			fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, data)
			flusher.Flush()
		}
		writeEvent("response.output_text.delta", `{"type":"response.output_text.delta","delta":"Hello"}`)
		writeEvent("response.completed", `{"type":"response.completed","response":{`+
			`"id":"resp_1","object":"response","created_at":0,"model":"gpt-4o-mini","status":"completed",`+
			`"output":[{"type":"message","id":"msg_1","role":"assistant","status":"completed",`+
			`"content":[{"type":"output_text","text":"Hello","annotations":[]}]}],`+
			`"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}`)
	}))
}

// drainStream collects all events, returning accumulated content and the first
// error event, if any.
func drainStream(ch <-chan interfaces.StreamEvent) (string, error) {
	var content strings.Builder
	var firstErr error
	for ev := range ch {
		if ev.Type == interfaces.StreamEventContentDelta {
			content.WriteString(ev.Content)
		}
		if ev.Type == interfaces.StreamEventError && firstErr == nil {
			firstErr = ev.Error
		}
	}
	return content.String(), firstErr
}

// requestTools returns the "tools" array of a captured request, or nil.
func requestTools(reqBody map[string]interface{}) []interface{} {
	tools, _ := reqBody["tools"].([]interface{})
	return tools
}

func TestGenerateStreamWithFileInputUsesResponsesAPI(t *testing.T) {
	var requests []map[string]interface{}
	server := newResponsesSSEServer(t, &requests)
	defer server.Close()

	client := openai_client.NewClient("test-key",
		openai_client.WithBaseURL(server.URL),
		openai_client.WithModel("gpt-4o-mini"),
		openai_client.WithLogger(logging.New()),
	)

	ch, err := client.GenerateStream(context.Background(), "Summarize this file",
		openai_client.WithFileID("file_123"),
	)
	if err != nil {
		t.Fatalf("GenerateStream failed: %v", err)
	}
	content, streamErr := drainStream(ch)
	if streamErr != nil {
		t.Fatalf("unexpected stream error: %v", streamErr)
	}
	if content != "Hello" {
		t.Errorf("expected streamed content %q, got %q", "Hello", content)
	}

	if len(requests) != 1 {
		t.Fatalf("expected 1 request, got %d", len(requests))
	}
	reqBody := requests[0]

	// Without code execution the file attaches as readable input_file content
	// on the user turn.
	input := reqBody["input"].([]interface{})
	message := input[len(input)-1].(map[string]interface{})
	parts, ok := message["content"].([]interface{})
	if !ok {
		t.Fatalf("expected content-part list on user message, got %v", message["content"])
	}
	foundFile := false
	for _, p := range parts {
		part := p.(map[string]interface{})
		if part["type"] == "input_file" && part["file_id"] == "file_123" {
			foundFile = true
		}
	}
	if !foundFile {
		t.Errorf("expected input_file part with file_123, got %v", parts)
	}

	if tools := requestTools(reqBody); len(tools) != 0 {
		t.Errorf("expected no tools without code execution, got %v", tools)
	}

	// Non-reasoning model: sampling params apply, reasoning must not be sent.
	if _, hasReasoning := reqBody["reasoning"]; hasReasoning {
		t.Errorf("expected no reasoning param for non-reasoning model, got %v", reqBody["reasoning"])
	}
	if _, hasTemp := reqBody["temperature"]; !hasTemp {
		t.Error("expected temperature to be set for non-reasoning model")
	}
}

func TestGenerateStreamWithCodeExecutionMountsFiles(t *testing.T) {
	var requests []map[string]interface{}
	server := newResponsesSSEServer(t, &requests)
	defer server.Close()

	client := openai_client.NewClient("test-key",
		openai_client.WithBaseURL(server.URL),
		openai_client.WithModel("gpt-4o-mini"),
		openai_client.WithLogger(logging.New()),
	)

	ch, err := client.GenerateStream(context.Background(), "Top 3 trends in this spreadsheet?",
		openai_client.WithFileID("file_123"),
		openai_client.WithCodeExecution(),
	)
	if err != nil {
		t.Fatalf("GenerateStream failed: %v", err)
	}
	if _, streamErr := drainStream(ch); streamErr != nil {
		t.Fatalf("unexpected stream error: %v", streamErr)
	}

	if len(requests) != 1 {
		t.Fatalf("expected 1 request, got %d", len(requests))
	}
	reqBody := requests[0]

	// Code execution mounts files in the container, so the user turn is plain text.
	input := reqBody["input"].([]interface{})
	message := input[len(input)-1].(map[string]interface{})
	if _, ok := message["content"].(string); !ok {
		t.Fatalf("expected plain string user content with code execution, got %v", message["content"])
	}

	tools := requestTools(reqBody)
	if len(tools) != 1 {
		t.Fatalf("expected exactly one tool, got %v", tools)
	}
	tool := tools[0].(map[string]interface{})
	if tool["type"] != "code_interpreter" {
		t.Fatalf("expected code_interpreter tool, got %v", tool["type"])
	}
	container := tool["container"].(map[string]interface{})
	fileIDs := container["file_ids"].([]interface{})
	if len(fileIDs) != 1 || fileIDs[0] != "file_123" {
		t.Errorf("expected file_123 mounted in container, got %v", fileIDs)
	}
}

func TestGenerateWithToolsStreamCombinesFunctionAndCodeExecutionTools(t *testing.T) {
	var requests []map[string]interface{}
	server := newResponsesSSEServer(t, &requests)
	defer server.Close()

	client := openai_client.NewClient("test-key",
		openai_client.WithBaseURL(server.URL),
		openai_client.WithModel("gpt-4o-mini"),
		openai_client.WithLogger(logging.New()),
	)

	tools := []interfaces.Tool{&mockTool{name: "get_data", description: "Fetches data"}}
	ch, err := client.GenerateWithToolsStream(context.Background(), "Analyze the file", tools,
		openai_client.WithFileID("file_123"),
		openai_client.WithCodeExecution(),
	)
	if err != nil {
		t.Fatalf("GenerateWithToolsStream failed: %v", err)
	}
	if _, streamErr := drainStream(ch); streamErr != nil {
		t.Fatalf("unexpected stream error: %v", streamErr)
	}

	if len(requests) != 1 {
		t.Fatalf("expected 1 request, got %d", len(requests))
	}

	toolTypes := map[string]bool{}
	var codeTool map[string]interface{}
	for _, tl := range requestTools(requests[0]) {
		tool := tl.(map[string]interface{})
		typ, _ := tool["type"].(string)
		toolTypes[typ] = true
		if typ == "code_interpreter" {
			codeTool = tool
		}
	}
	if !toolTypes["function"] || !toolTypes["code_interpreter"] {
		t.Fatalf("expected function and code_interpreter tools, got %v", toolTypes)
	}
	container := codeTool["container"].(map[string]interface{})
	fileIDs := container["file_ids"].([]interface{})
	if len(fileIDs) != 1 || fileIDs[0] != "file_123" {
		t.Errorf("expected file_123 mounted in container, got %v", fileIDs)
	}
}
