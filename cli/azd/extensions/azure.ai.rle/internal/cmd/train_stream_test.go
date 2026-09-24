// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package cmd

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
)

func chunkFrame(name string, mode string, offset int64, payload string) trainStreamFrame {
	return trainStreamFrame{
		Type:     "chunk",
		Name:     name,
		Mode:     mode,
		Offset:   offset,
		Encoding: "base64",
		Data:     base64.StdEncoding.EncodeToString([]byte(payload)),
	}
}

func TestTrainStreamURLUpgradesTheScheme(t *testing.T) {
	cases := map[string]string{
		"https://host":       "wss://host/openai/v1/fine_tuning/jobs/ftjob-1/stream",
		"https://host/":      "wss://host/openai/v1/fine_tuning/jobs/ftjob-1/stream",
		"http://127.0.0.1:8": "ws://127.0.0.1:8/openai/v1/fine_tuning/jobs/ftjob-1/stream",
		"https://host/base":  "wss://host/base/openai/v1/fine_tuning/jobs/ftjob-1/stream",
	}
	for endpoint, expected := range cases {
		actual, err := trainStreamURL(endpoint, "ftjob-1", "")
		if err != nil {
			t.Fatalf("%s: %v", endpoint, err)
		}
		if actual != expected {
			t.Fatalf("%s -> %s, wanted %s", endpoint, actual, expected)
		}
	}
}

// An https endpoint must never be downgraded: the bearer token rides in the
// handshake headers.
func TestTrainStreamURLRefusesAnUnknownScheme(t *testing.T) {
	if _, err := trainStreamURL("ftp://host", "ftjob-1", ""); err == nil {
		t.Fatal("expected a non-http scheme to be refused")
	}
}

func TestTrainStreamURLCarriesResumeState(t *testing.T) {
	actual, err := trainStreamURL("https://host", "ftjob-1", `{"offsets":{"logs.log":12}}`)
	if err != nil {
		t.Fatalf("trainStreamURL: %v", err)
	}
	if !strings.Contains(actual, "resume=") || !strings.Contains(actual, "logs.log") {
		t.Fatalf("resume state missing from %s", actual)
	}
}

func TestMirrorWritesAppendedAndReplacedArtifacts(t *testing.T) {
	mirror, err := newRunMirror(t.TempDir(), "ftjob-1")
	if err != nil {
		t.Fatalf("newRunMirror: %v", err)
	}

	if err := mirror.apply(chunkFrame("metrics.jsonl", "append", 0, "{\"step\":1}\n")); err != nil {
		t.Fatalf("apply: %v", err)
	}
	if err := mirror.apply(chunkFrame("metrics.jsonl", "append", 11, "{\"step\":2}\n")); err != nil {
		t.Fatalf("apply: %v", err)
	}
	if err := mirror.apply(chunkFrame("run_meta.json", "replace", 0, `{"a":1}`)); err != nil {
		t.Fatalf("apply: %v", err)
	}
	if err := mirror.apply(chunkFrame("run_meta.json", "replace", 0, `{"a":2}`)); err != nil {
		t.Fatalf("apply: %v", err)
	}

	metrics, err := os.ReadFile(filepath.Join(mirror.directory, "metrics.jsonl"))
	if err != nil {
		t.Fatalf("read metrics: %v", err)
	}
	if string(metrics) != "{\"step\":1}\n{\"step\":2}\n" {
		t.Fatalf("appended artifact is %q", metrics)
	}

	meta, err := os.ReadFile(filepath.Join(mirror.directory, "run_meta.json"))
	if err != nil {
		t.Fatalf("read run_meta: %v", err)
	}
	if string(meta) != `{"a":2}` {
		t.Fatalf("replaced artifact is %q", meta)
	}
}

// The artifact name becomes a path on the caller's disk. A service that could
// choose it could write anywhere the caller can.
func TestMirrorIgnoresArtifactsItDoesNotExpect(t *testing.T) {
	root := t.TempDir()
	mirror, err := newRunMirror(root, "ftjob-1")
	if err != nil {
		t.Fatalf("newRunMirror: %v", err)
	}

	for _, name := range []string{
		"../../escaped.txt",
		"/etc/passwd",
		"nested/thing.json",
		"unexpected.bin",
	} {
		if err := mirror.apply(chunkFrame(name, "append", 0, "nope")); err != nil {
			t.Fatalf("apply(%s): %v", name, err)
		}
	}

	var written []string
	err = filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err == nil && info != nil && !info.IsDir() {
			written = append(written, path)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walk: %v", err)
	}
	if len(written) != 0 {
		t.Fatalf("unexpected artifact names were written: %v", written)
	}
}

func TestMirrorRejectsAnUnknownEncoding(t *testing.T) {
	mirror, err := newRunMirror(t.TempDir(), "ftjob-1")
	if err != nil {
		t.Fatalf("newRunMirror: %v", err)
	}
	frame := chunkFrame("logs.log", "append", 0, "hello")
	frame.Encoding = "rot13"
	if err := mirror.apply(frame); err == nil {
		t.Fatal("expected an unsupported encoding to be refused")
	}
}

func TestMirrorResumesFromWhatIsAlreadyOnDisk(t *testing.T) {
	root := t.TempDir()
	directory := filepath.Join(root, "rle-harness", "ftjob-1")
	if err := os.MkdirAll(directory, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(directory, "logs.log"), []byte("already\n"), 0o644); err != nil {
		t.Fatalf("seed: %v", err)
	}

	mirror, err := newRunMirror(root, "ftjob-1")
	if err != nil {
		t.Fatalf("newRunMirror: %v", err)
	}
	if mirror.offsets["logs.log"] != int64(len("already\n")) {
		t.Fatalf("expected the existing file length to be adopted, got %d", mirror.offsets["logs.log"])
	}
	resume := mirror.resumeQuery()
	if !strings.Contains(resume, "logs.log") {
		t.Fatalf("resume query does not mention the file: %s", resume)
	}
}

func TestMirrorLandsInTheLayoutTheDashboardDiscovers(t *testing.T) {
	// dashboard_server.py treats any directory holding metrics.jsonl or
	// config.json as a run, and labels it by its path under the root.
	root := t.TempDir()
	mirror, err := newRunMirror(root, "ftjob-1")
	if err != nil {
		t.Fatalf("newRunMirror: %v", err)
	}
	relative, err := filepath.Rel(root, mirror.directory)
	if err != nil {
		t.Fatalf("rel: %v", err)
	}
	if relative != filepath.Join("rle-harness", "ftjob-1") {
		t.Fatalf("run directory is %s", relative)
	}
}

// followTrainingRun against a real WebSocket server, so the frame shapes are
// exercised over the wire rather than asserted in isolation.
func TestFollowTrainingRunMirrorsAndStops(t *testing.T) {
	upgrader := websocket.Upgrader{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer test" {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		connection, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer connection.Close()
		frames := []any{
			map[string]any{"type": "hello", "job_id": "ftjob-1", "status": "running"},
			chunkFrame("metrics.jsonl", "append", 0, "{\"step\":1}\n"),
			chunkFrame("run_meta.json", "replace", 0, `{"recipe":"rle-harness"}`),
			map[string]any{"type": "status", "status": "succeeded"},
			map[string]any{"type": "end", "status": "succeeded"},
		}
		for _, frame := range frames {
			payload, _ := json.Marshal(frame)
			if err := connection.WriteMessage(websocket.TextMessage, payload); err != nil {
				return
			}
		}
	}))
	defer server.Close()

	root := t.TempDir()
	var out bytes.Buffer
	status, err := followTrainingRun(
		context.Background(), server.URL, "Bearer test", "ftjob-1", root, &out)
	if err != nil {
		t.Fatalf("followTrainingRun: %v", err)
	}
	if status != "succeeded" {
		t.Fatalf("status is %q", status)
	}

	directory := filepath.Join(root, "rle-harness", "ftjob-1")
	metrics, err := os.ReadFile(filepath.Join(directory, "metrics.jsonl"))
	if err != nil {
		t.Fatalf("read mirrored metrics: %v", err)
	}
	if string(metrics) != "{\"step\":1}\n" {
		t.Fatalf("mirrored metrics are %q", metrics)
	}
	if _, err := os.ReadFile(filepath.Join(directory, "run_meta.json")); err != nil {
		t.Fatalf("read mirrored run_meta: %v", err)
	}
	if !strings.Contains(out.String(), "dashboard_server.py") {
		t.Fatalf("expected the output to say how to view the run, got %q", out.String())
	}
}

func TestFollowTrainingRunReportsAnUpgradeRefusal(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte(`{"error":{"message":"No such job."}}`))
	}))
	defer server.Close()

	var out bytes.Buffer
	_, err := followTrainingRun(
		context.Background(), server.URL, "Bearer test", "ftjob-missing", t.TempDir(), &out)
	if err == nil {
		t.Fatal("expected a refused upgrade to be an error")
	}
	if !strings.Contains(err.Error(), "No such job") {
		t.Fatalf("the service's reason was lost: %v", err)
	}
}

func TestDefaultLogsRootPrefersTheDashboardEnvironmentVariable(t *testing.T) {
	t.Setenv("LOOM_LOGS_ROOT", "/tmp/somewhere-else")
	if defaultLogsRoot() != "/tmp/somewhere-else" {
		t.Fatalf("defaultLogsRoot is %s", defaultLogsRoot())
	}
	t.Setenv("LOOM_LOGS_ROOT", "")
	if !strings.HasSuffix(defaultLogsRoot(), "loom-runs") {
		t.Fatalf("defaultLogsRoot is %s", defaultLogsRoot())
	}
}

// A run is mostly silence: the service sends a data frame only when an artifact
// changes, and a rollout can take minutes. The connection is held open by pings,
// which must count as liveness or a healthy run is cut off part way through.
func TestFollowTrainingRunSurvivesAnIdleConnection(t *testing.T) {
	original := trainStreamReadTimeout
	trainStreamReadTimeout = 300 * time.Millisecond
	t.Cleanup(func() { trainStreamReadTimeout = original })

	upgrader := websocket.Upgrader{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		connection, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer connection.Close()

		hello, _ := json.Marshal(map[string]any{
			"type": "hello", "job_id": "ftjob-1", "status": "running",
		})
		if err := connection.WriteMessage(websocket.TextMessage, hello); err != nil {
			return
		}

		// Nothing but pings for well over the read timeout, the way the service
		// behaves while a rollout is running.
		deadline := time.Now().Add(900 * time.Millisecond)
		for time.Now().Before(deadline) {
			err := connection.WriteControl(
				websocket.PingMessage, nil, time.Now().Add(time.Second))
			if err != nil {
				return
			}
			time.Sleep(100 * time.Millisecond)
		}

		end, _ := json.Marshal(map[string]any{"type": "end", "status": "succeeded"})
		_ = connection.WriteMessage(websocket.TextMessage, end)
	}))
	defer server.Close()

	status, err := followTrainingRun(
		context.Background(), server.URL, "******", "ftjob-1", t.TempDir(), io.Discard)
	if err != nil {
		t.Fatalf("an idle connection ended the stream: %v", err)
	}
	if status != "succeeded" {
		t.Fatalf("status is %q, want the status that arrived after the idle gap", status)
	}
}
