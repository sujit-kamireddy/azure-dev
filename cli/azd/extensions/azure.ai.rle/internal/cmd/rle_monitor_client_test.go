// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package cmd

import (
	"compress/gzip"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"azure.ai.rle/internal/monitor"
)

const realMonitorJobID = "ftjob-7898670d3ae8453e89e591f9"

func stubAPIJobMonitor(t *testing.T) {
	t.Helper()
	oldRun, oldRle := runAPIJobMonitor, createRleClient
	t.Cleanup(func() { runAPIJobMonitor, createRleClient = oldRun, oldRle })
	createRleClient = func(endpoint string) (*rleClient, error) {
		return newRleClientWithCredential(endpoint, &testTokenCredential{}), nil
	}
	runAPIJobMonitor = func(context.Context, monitor.JobSource, string, bool, io.Writer, io.Writer) error {
		return nil
	}
}

// TestTrainAndMonitorUseTheSameAPIBackendRegardlessOfEndpoint is the regression
// test for the single unified mode: train's own post-submission monitor and a
// separate `azd ai rle monitor --job-id` both read Config/Metrics/Rollouts from
// the real RLE service, whether or not RLE_TRAIN_ENDPOINT/--endpoint is set.
// Only job creation and status ever go through a facade endpoint.
func TestTrainAndMonitorUseTheSameAPIBackendRegardlessOfEndpoint(t *testing.T) {
	for _, env := range []string{"", "https://facade.example.com"} {
		t.Run("env="+env, func(t *testing.T) {
			action, _ := stubbedTrain(t, t.Context(), &rleTrainFlags{noBrowser: true})
			t.Setenv(rleTrainEndpointEnvVar, env)
			stubAPIJobMonitor(t)
			calls := 0
			runAPIJobMonitor = func(
				ctx context.Context, source monitor.JobSource, id string, noBrowser bool, out, errOut io.Writer,
			) error {
				calls++
				s, ok := source.(*rleJobSource)
				if !ok || id != realMonitorJobID || !noBrowser || s.jobID != id ||
					s.project != "project" || s.rle.baseUrl != "https://account.services.ai.azure.com/api/projects/project" {
					t.Fatalf("unexpected monitor source: %#v", source)
				}
				return nil
			}
			oldStream := followTrainingRunFunc
			t.Cleanup(func() { followTrainingRunFunc = oldStream })
			followTrainingRunFunc = func(context.Context, string, string, string, string, io.Writer) (string, error) {
				return "succeeded", nil
			}
			if err := action.Run(); err != nil {
				t.Fatal(err)
			}
			command := newMonitorCommand()
			command.SetOut(io.Discard)
			command.SetErr(io.Discard)
			command.SetArgs([]string{"--job-id", realMonitorJobID, "--no-browser"})
			if err := command.Execute(); err != nil {
				t.Fatal(err)
			}
			if calls != 2 {
				t.Fatalf("monitor started %d times, want 2", calls)
			}
		})
	}
}

func TestRleMonitorClientRoutesAndScopes(t *testing.T) {
	credential := &testTokenCredential{}
	rle := newRleClientWithCredential("https://account.services.ai.azure.com/api/projects/project", credential)
	ftCred := &testTokenCredential{}
	ft := newFinetuneClientWithCredential("https://account.openai.azure.com", ftCred)
	s := &rleJobSource{rle: rle, ft: ft, jobID: realMonitorJobID, project: "project"}
	rle.httpClient.Transport = roundTripFunc(func(req *http.Request) (*http.Response, error) {
		if req.Method != "GET" || req.URL.Query().Get("api-version") != foundryAPIVersion {
			t.Fatalf("unexpected request %s", req.URL)
		}
		body := `{"job_id":"` + realMonitorJobID + `","model_name":"sample"}`
		switch {
		case strings.HasSuffix(req.URL.Path, "/metrics"):
			if req.URL.Query().Get("lastStep") != "7" || req.URL.Query().Get("continuationToken") != "a+/=" {
				t.Fatal("metric cursor was not preserved")
			}
			body = `{"data":[{"step":8,"custom":1}],"nextContinuationToken":"opaque"}`
		case strings.HasSuffix(req.URL.Path, "/rollouts"):
			query := req.URL.Query()
			if query.Has("lastSequence") || query.Has("after") || query.Has("createdAfter") ||
				query.Get("limit") != "100" || query.Get("continuationToken") != "t+/=" || len(query) != 3 ||
				!strings.Contains(req.URL.RawQuery, "continuationToken=t%2B%2F%3D") {
				t.Fatalf("unexpected rollout query %s", req.URL.RawQuery)
			}
			body = `{"data":[{"rollout_id":"` + monitorTestID + `","job_id":"` + realMonitorJobID +
				`","status":"completed","success":false}]}`
		}
		return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(body)), Header: http.Header{}}, nil
	})
	ft.httpClient.Transport = roundTripFunc(func(req *http.Request) (*http.Response, error) {
		if req.URL.Path != finetuneJobsPath+"/"+realMonitorJobID || req.Header.Get("azureai-project") != "project" {
			t.Fatalf("unexpected job status request: %s", req.URL)
		}
		return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(
			`{"id":"` + realMonitorJobID + `","status":"running"}`)), Header: http.Header{}}, nil
	})
	if _, err := s.Config(t.Context()); err != nil {
		t.Fatal(err)
	}
	page, err := s.Metrics(t.Context(), 7, "a+/=")
	if err != nil || page.Next != "opaque" {
		t.Fatalf("%+v %v", page, err)
	}
	entries, err := s.Rollouts(t.Context(), "t+/=")
	if err != nil || entries.Data[0].Success == nil || *entries.Data[0].Success {
		t.Fatalf("%+v %v", entries, err)
	}
	if status, err := s.Status(t.Context()); err != nil || status != "running" {
		t.Fatalf("%s %v", status, err)
	}
	for _, scope := range credential.scopes {
		if scope != foundryTokenScope {
			t.Fatal("RLE used fine-tuning auth scope")
		}
	}
	if len(ftCred.scopes) != 1 || ftCred.scopes[0] != finetuneTokenScope {
		t.Fatal("wrong status auth scope")
	}
}

func TestRleMonitorResultGzipAndUnavailable(t *testing.T) {
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Encoding", "gzip")
		w.Header().Set("Content-Type", "application/json")
		zip := gzip.NewWriter(w)
		defer zip.Close()
		_, _ = io.WriteString(zip, `{"rollout_id":"`+monitorTestID+`","reward":0.5,"rollout":{"turns":[]}}`)
	}))
	defer server.Close()
	rle := newRleClientWithCredential(server.URL, &testTokenCredential{})
	rle.httpClient = server.Client()
	s := &rleJobSource{rle: rle, jobID: realMonitorJobID}
	snapshot, err := s.Result(t.Context(), monitor.JobRollout{RolloutID: monitorTestID})
	if err != nil || !strings.Contains(string(snapshot.Response), `"reward":0.5`) {
		t.Fatalf("%+v %v", snapshot, err)
	}
}

func TestMonitorReadLimitsAndCredentialNonDisclosure(t *testing.T) {
	auth := func(context.Context) (string, error) { return "Bearer test", nil }
	for _, tc := range []struct {
		name   string
		status int
		body   string
		want   string
	}{
		{"throttled", 429, `{"code":"TooManyRequests","message":"sig=secret"}`, "TooManyRequests"},
		{"malformed", 200, `{`, "decode"},
		{"oversized", 200, `{"data":"larger than cap"}`, "size limit"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			client := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				return &http.Response{StatusCode: tc.status, Body: io.NopCloser(strings.NewReader(tc.body)),
					Header: http.Header{"Retry-After": {"60"}}}, nil
			})}
			var target json.RawMessage
			err := readMonitorJSON(t.Context(), client, "https://example.com", "/read?sig=secret",
				auth, nil, &target, 8)
			if err == nil || !strings.Contains(err.Error(), tc.want) || strings.Contains(err.Error(), "secret") {
				t.Fatalf("unexpected error: %v", err)
			}
			if tc.status == 429 {
				readErr, ok := errors.AsType[*monitor.ReadError](err)
				if !ok || readErr.RetryAfter != time.Minute {
					t.Fatal("Retry-After lost")
				}
			}
		})
	}
	var target any
	err := readMonitorJSON(t.Context(), http.DefaultClient, "https://user:password@example.com", "/read?sig=secret",
		auth, nil, &target, 100)
	if err == nil || strings.Contains(err.Error(), "password") || strings.Contains(err.Error(), "secret") {
		t.Fatalf("credential-bearing endpoint not rejected safely: %v", err)
	}
}

func TestMonitorErrorKeepsCorrelationIDs(t *testing.T) {
	auth := func(context.Context) (string, error) { return "******", nil }
	for _, tc := range []struct {
		name, body, header, code, operation, request string
	}{
		{"nested", `{"error":{"code":"ServiceError","message":"sig=secret"},` +
			`"correlation":{"operation":"573804dcdf39af6a10d06775fc7cfc92","request":"abc-123"}}`,
			"0f4c-77aa", "ServiceError", "573804dcdf39af6a10d06775fc7cfc92", "0f4c-77aa"},
		{"unsafe", `{"error":{"code":"bad code!"},"correlation":{"operation":"x y"}}`,
			"bad\nvalue", "Internal Server Error", "", ""},
		{"body request", `{"correlation":{"request":"abc-123"}}`, "", "Internal Server Error", "", "abc-123"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			client := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				header := http.Header{}
				if tc.header != "" {
					header.Set("x-ms-request-id", tc.header)
				}
				return &http.Response{StatusCode: 500, Body: io.NopCloser(strings.NewReader(tc.body)),
					Header: header}, nil
			})}
			var target any
			err := readMonitorJSON(t.Context(), client, "https://example.com", "/read", auth, nil, &target, 100)
			readErr, ok := errors.AsType[*monitor.ReadError](err)
			if !ok || readErr.Code != tc.code || readErr.Operation != tc.operation || readErr.Request != tc.request {
				t.Fatalf("unexpected error: %#v", err)
			}
			if strings.Contains(err.Error(), "secret") ||
				(tc.operation != "" && !strings.Contains(err.Error(), "operation ID "+tc.operation)) {
				t.Fatalf("unexpected message: %v", err)
			}
		})
	}
}

func TestRealMonitorFailureDoesNotResubmit(t *testing.T) {
	action, output := stubbedTrain(t, t.Context(), &rleTrainFlags{noBrowser: true})
	stubAPIJobMonitor(t)
	calls := 0
	runAPIJobMonitor = func(context.Context, monitor.JobSource, string, bool, io.Writer, io.Writer) error {
		calls++
		return errors.New("listener unavailable")
	}
	originalFollow := followTrainingRunFunc
	followTrainingRunFunc = func(context.Context, string, string, string, string, io.Writer) (string, error) {
		return "succeeded", nil
	}
	t.Cleanup(func() { followTrainingRunFunc = originalFollow })

	// A dashboard that fails to start is reported, not fatal: the job was
	// already accepted and is running on the service regardless.
	if err := action.Run(); err != nil {
		t.Fatal(err)
	}
	if calls != 1 {
		t.Fatalf("job monitor started %d times, want 1", calls)
	}
	if !strings.Contains(output.String(), "The job monitor did not start: listener unavailable") {
		t.Fatalf("output = %q, want the monitor failure reported", output.String())
	}
}
