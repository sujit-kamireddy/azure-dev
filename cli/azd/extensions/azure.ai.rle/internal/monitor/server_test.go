// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package monitor

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"azure.ai.rle/internal/rollouts"
	"azure.ai.rle/internal/ui"
)

const testHost = "127.0.0.1:12345"

func TestHandlerAuthorizationAndRoutes(t *testing.T) {
	snapshot := rollouts.Snapshot{
		Response: json.RawMessage(`{"rollout_id":"abc","reward":1,"result":"<script>alert(1)</script>"}`),
		Source:   "Saved local result",
	}
	handler, err := newHandler(source{snapshot: &snapshot}, testHost)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name, method, path, host, origin string
		status                           int
	}{
		{name: "public shell", method: "GET", path: "/", status: 200},
		{name: "rollout data", method: "GET", path: "/api/rollout", status: 200},
		{name: "foreign host", method: "GET", path: "/api/rollout", host: "evil.test", status: 403},
		{name: "foreign origin", method: "GET", path: "/api/rollout", origin: "https://evil.test", status: 403},
		{name: "read only", method: "POST", path: "/api/rollout", status: 405},
		{name: "unknown route", method: "GET", path: "/api/other", status: 404},
		{name: "no directories", method: "GET", path: "/web/", status: 404},
	} {
		t.Run(tc.name, func(t *testing.T) {
			request := httptest.NewRequest(tc.method, "http://"+testHost+tc.path, nil)
			if tc.host != "" {
				request.Host = tc.host
			}
			request.Header.Set("Origin", tc.origin)
			recorder := httptest.NewRecorder()
			handler.ServeHTTP(recorder, request)
			if recorder.Code != tc.status {
				t.Fatalf("expected %d, got %d: %s", tc.status, recorder.Code, recorder.Body.String())
			}
			if recorder.Header().Get("Cache-Control") != "no-store" ||
				recorder.Header().Get("Content-Security-Policy") == "" {
				t.Fatal("missing privacy/security headers")
			}
			if tc.name == "rollout data" {
				var got rollouts.Snapshot
				if err := json.Unmarshal(recorder.Body.Bytes(), &got); err != nil {
					t.Fatal(err)
				}
				if !strings.Contains(string(got.Response), "reward") {
					t.Fatal("missing rollout response")
				}
			} else if strings.Contains(recorder.Body.String(), "alert(1)") {
				t.Fatal("result leaked outside the data route")
			}
		})
	}
}

func TestModuleAssetsUseJavaScriptMIMEType(t *testing.T) {
	handler, err := newHandler(source{snapshot: &rollouts.Snapshot{Response: json.RawMessage(`{}`)}}, testHost)
	if err != nil {
		t.Fatal(err)
	}
	modules, err := fs.Glob(Assets, "web/*.mjs")
	if err != nil || len(modules) == 0 {
		t.Fatalf("missing embedded modules: %v", err)
	}
	for _, module := range modules {
		request := httptest.NewRequest("GET", "http://"+testHost+"/"+strings.TrimPrefix(module, "web/"), nil)
		recorder := httptest.NewRecorder()
		handler.ServeHTTP(recorder, request)
		if recorder.Code != http.StatusOK || !strings.HasPrefix(recorder.Header().Get("Content-Type"), "text/javascript") {
			t.Fatalf("module %s is not served as JavaScript: %d %v", module, recorder.Code, recorder.Header())
		}
	}
}

type staticReader struct{}

func (staticReader) Get(context.Context, string) (rollouts.Snapshot, error) {
	return rollouts.Snapshot{Response: json.RawMessage(`{"rollout_id":"abc","reward":1}`)}, nil
}

type readyWriter struct {
	ready chan string
}

func (w readyWriter) Write(data []byte) (int, error) {
	w.ready <- string(data)
	return len(data), nil
}

func TestRunServesAndStops(t *testing.T) {
	for _, noBrowser := range []bool{true, false} {
		t.Run(fmt.Sprint(noBrowser), func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			oldOpen := ui.OpenBrowser
			t.Cleanup(func() { ui.OpenBrowser = oldOpen })
			opened := make(chan string, 1)
			ui.OpenBrowser = func(link string) error {
				opened <- link
				return errors.New("browser failed with secret-url")
			}
			ready := make(chan string, 1)
			done := make(chan error, 1)
			var diagnostics bytes.Buffer
			go func() { done <- Run(ctx, staticReader{}, "abc", noBrowser, readyWriter{ready}, &diagnostics) }()
			var output string
			select {
			case output = <-ready:
			case <-time.After(10 * time.Second):
				t.Fatal("monitor did not start")
			}
			link := strings.TrimPrefix(strings.Split(output, "\n")[0], "Rollout monitor: ")
			if strings.Contains(link, "#") || strings.Contains(link, "?") {
				t.Fatal("printed URL contains unexpected query or fragment")
			}
			client := &http.Client{Timeout: 5 * time.Second}
			response, err := client.Get(link)
			if err != nil {
				t.Fatal(err)
			}
			_ = response.Body.Close()
			if response.StatusCode != 200 {
				t.Fatalf("monitor not responsive: %d", response.StatusCode)
			}
			if !noBrowser {
				select {
				case openedLink := <-opened:
					if openedLink != link {
						t.Fatalf("automatic browser link %q does not match printed link %q", openedLink, link)
					}
				case <-time.After(5 * time.Second):
					t.Fatal("browser not opened")
				}
			}
			cancel()
			select {
			case err := <-done:
				if err != nil {
					t.Fatal(err)
				}
			case <-time.After(10 * time.Second):
				t.Fatal("monitor did not stop")
			}
			if noBrowser && len(opened) != 0 {
				t.Fatal("--no-browser opened a browser")
			}
			if strings.Contains(diagnostics.String(), "secret-url") {
				t.Fatal("browser error disclosed its URL")
			}
			if !noBrowser && !strings.Contains(diagnostics.String(), "Warning:") {
				t.Fatal("browser failure was not reported")
			}
		})
	}
}

type stubJobSource struct {
	entries  []rollouts.Entry
	requests []string
	err      error
}

func (s *stubJobSource) List(context.Context) ([]rollouts.Entry, error) { return s.entries, s.err }

func (s *stubJobSource) Get(_ context.Context, rolloutID string) (rollouts.Snapshot, error) {
	s.requests = append(s.requests, rolloutID)
	if s.err != nil {
		return rollouts.Snapshot{}, s.err
	}
	return rollouts.Snapshot{
		Response: json.RawMessage(`{"rollout_id":"` + rolloutID + `","reward":1}`),
		Source:   "Training run artifacts",
	}, nil
}

func jobHandler(t *testing.T, stub *stubJobSource) http.Handler {
	t.Helper()
	handler, err := newHandler(source{jobID: "ftjob-1", entries: stub.entries, reader: stub}, testHost)
	if err != nil {
		t.Fatal(err)
	}
	return handler
}

func jobRequest(t *testing.T, handler http.Handler, path string) *httptest.ResponseRecorder {
	t.Helper()
	request := httptest.NewRequest("GET", "http://"+testHost+path, nil)
	request.Host = testHost
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, request)
	return recorder
}

func TestJobModeListsAndFetchesOnDemand(t *testing.T) {
	success := true
	reward := 0.75
	stub := &stubJobSource{entries: []rollouts.Entry{
		{RolloutID: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", Sequence: 1, Split: "validation",
			Reward: &reward, Success: &success},
		{RolloutID: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", Sequence: 2, Split: "train"},
	}}
	handler := jobHandler(t, stub)

	index := jobRequest(t, handler, "/api/rollouts")
	if index.Code != 200 {
		t.Fatalf("list status = %d, want 200", index.Code)
	}
	var listed struct {
		JobID string           `json:"job_id"`
		Data  []rollouts.Entry `json:"data"`
	}
	if err := json.Unmarshal(index.Body.Bytes(), &listed); err != nil {
		t.Fatal(err)
	}
	if listed.JobID != "ftjob-1" || len(listed.Data) != 2 {
		t.Fatalf("list = %+v, want job ftjob-1 with 2 entries", listed)
	}
	// The index must stay a summary: a run records thousands of large responses.
	if len(stub.requests) != 0 {
		t.Fatalf("listing fetched %v, want no rollout bodies", stub.requests)
	}

	body := jobRequest(t, handler, "/api/rollout?id=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
	if body.Code != 200 {
		t.Fatalf("fetch status = %d, want 200", body.Code)
	}
	if len(stub.requests) != 1 || stub.requests[0] != "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" {
		t.Fatalf("fetched %v, want the opened rollout only", stub.requests)
	}
}

func TestJobModeRejectsRolloutsOutsideTheJob(t *testing.T) {
	stub := &stubJobSource{entries: []rollouts.Entry{{RolloutID: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}}}
	handler := jobHandler(t, stub)
	response := jobRequest(t, handler, "/api/rollout?id=cccccccccccccccccccccccccccccccc")
	if response.Code != 404 {
		t.Fatalf("status = %d, want 404 for a rollout this job did not record", response.Code)
	}
	if len(stub.requests) != 0 {
		t.Fatalf("fetched %v, want no call for an unlisted rollout", stub.requests)
	}
}

func TestJobModeRequiresAnID(t *testing.T) {
	handler := jobHandler(t, &stubJobSource{entries: []rollouts.Entry{}})
	if response := jobRequest(t, handler, "/api/rollout"); response.Code != 400 {
		t.Fatalf("status = %d, want 400 when no rollout is named", response.Code)
	}
}

func TestSingleRolloutModeHasNoIndex(t *testing.T) {
	handler, err := newHandler(source{snapshot: &rollouts.Snapshot{Response: json.RawMessage(`{}`)}}, testHost)
	if err != nil {
		t.Fatal(err)
	}
	// The page reads 404 as "there is no set to browse" and shows the saved rollout.
	if response := jobRequest(t, handler, "/api/rollouts"); response.Code != 404 {
		t.Fatalf("status = %d, want 404 without a job", response.Code)
	}
	if response := jobRequest(t, handler, "/api/rollout"); response.Code != 200 {
		t.Fatalf("status = %d, want the saved rollout", response.Code)
	}
}

func TestRunJobSurfacesListFailure(t *testing.T) {
	stub := &stubJobSource{err: errors.New("service unavailable")}
	err := RunJob(context.Background(), stub, stub, "ftjob-1", true, io.Discard, io.Discard)
	if err == nil || !strings.Contains(err.Error(), "service unavailable") {
		t.Fatalf("err = %v, want the listing failure", err)
	}
}
