// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package cmd

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/azure/azure-dev/cli/azd/pkg/azdext"
	"github.com/gorilla/websocket"
)

const (
	trainStreamHandshakeTimeout = 30 * time.Second
	// The service sends a heartbeat every 30s and a poll result every second, so
	// silence for this long means the connection is dead rather than idle.
	trainStreamReadTimeout = 90 * time.Second
)

// trainStreamArtifacts is what the local dashboard reads. Frames naming anything
// else are ignored: the artifact name becomes a path on the caller's disk, and a
// service that could choose that path could write anywhere the caller can.
var trainStreamArtifacts = map[string]bool{
	"metrics.jsonl": true,
	"config.json":   true,
	"logs.log":      true,
	"run_meta.json": true,
	"console.log":   true,
}

type trainStreamFrame struct {
	Type     string          `json:"type"`
	JobID    string          `json:"job_id"`
	Status   string          `json:"status"`
	Name     string          `json:"name"`
	Mode     string          `json:"mode"`
	Offset   int64           `json:"offset"`
	Encoding string          `json:"encoding"`
	Data     string          `json:"data"`
	Digest   string          `json:"digest"`
	Error    json.RawMessage `json:"error"`
}

// runMirror keeps a local copy of one run's artifacts, in the layout the
// cookbook dashboard discovers: <root>/rle-harness/<job id>/, matching where
// orchestration.py writes when the recipe runs locally. The dashboard cannot
// tell the difference, which is the point -- there is no second dashboard to
// keep in step with the first.
type runMirror struct {
	directory string
	offsets   map[string]int64
	digests   map[string]string
}

func newRunMirror(logsRoot string, jobID string) (*runMirror, error) {
	directory := filepath.Join(logsRoot, "rle-harness", jobID)
	if err := os.MkdirAll(directory, 0o755); err != nil {
		return nil, fmt.Errorf("create local run directory %s: %w", directory, err)
	}
	mirror := &runMirror{
		directory: directory,
		offsets:   map[string]int64{},
		digests:   map[string]string{},
	}
	mirror.adoptExistingFiles()
	return mirror, nil
}

// adoptExistingFiles lets a re-run of --follow against the same job resume
// rather than re-download. Only appended files can be resumed by length;
// replaced files are re-sent, which costs nothing because they are small.
func (m *runMirror) adoptExistingFiles() {
	for name := range trainStreamArtifacts {
		info, err := os.Stat(filepath.Join(m.directory, name))
		if err == nil && info.Mode().IsRegular() {
			m.offsets[name] = info.Size()
		}
	}
}

func (m *runMirror) resumeQuery() string {
	if len(m.offsets) == 0 && len(m.digests) == 0 {
		return ""
	}
	payload := map[string]any{}
	if len(m.offsets) > 0 {
		payload["offsets"] = m.offsets
	}
	if len(m.digests) > 0 {
		payload["digests"] = m.digests
	}
	encoded, err := json.Marshal(payload)
	if err != nil {
		return ""
	}
	return string(encoded)
}

func (m *runMirror) apply(frame trainStreamFrame) error {
	if !trainStreamArtifacts[frame.Name] {
		return nil
	}
	if frame.Encoding != "base64" {
		return fmt.Errorf("artifact %s arrived in unsupported encoding %q", frame.Name, frame.Encoding)
	}
	payload, err := base64.StdEncoding.DecodeString(frame.Data)
	if err != nil {
		return fmt.Errorf("decode artifact %s: %w", frame.Name, err)
	}
	path := filepath.Join(m.directory, frame.Name)

	if frame.Mode == "replace" {
		// Written whole and atomically: the dashboard polls these files and
		// would otherwise read a half-written JSON document and log a parse
		// error the user cannot act on.
		temporary := path + ".partial"
		if err := os.WriteFile(temporary, payload, 0o644); err != nil {
			return fmt.Errorf("write artifact %s: %w", frame.Name, err)
		}
		if err := os.Rename(temporary, path); err != nil {
			return fmt.Errorf("replace artifact %s: %w", frame.Name, err)
		}
		m.digests[frame.Name] = frame.Digest
		return nil
	}

	file, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return fmt.Errorf("open artifact %s: %w", frame.Name, err)
	}
	defer file.Close()

	// WriteAt rather than append: the frame carries the offset its bytes belong
	// at, so a resumed stream lands them in the right place even if the local
	// file is longer than the service thinks.
	if _, err := file.WriteAt(payload, frame.Offset); err != nil {
		return fmt.Errorf("write artifact %s: %w", frame.Name, err)
	}
	if end := frame.Offset + int64(len(payload)); end > m.offsets[frame.Name] {
		m.offsets[frame.Name] = end
	}
	return nil
}

// dialTrainStream is the dial seam. gorilla's Dialer does not run through an
// http.Client's transport, so tests replace this rather than a round tripper.
var dialTrainStream = func(
	ctx context.Context,
	endpoint string,
	headers http.Header,
) (*websocket.Conn, *http.Response, error) {
	dialer := &websocket.Dialer{
		HandshakeTimeout: trainStreamHandshakeTimeout,
		Proxy:            http.ProxyFromEnvironment,
	}
	return dialer.DialContext(ctx, endpoint, headers)
}

// trainStreamURL turns the fine-tuning endpoint into the WebSocket address of a
// job's stream, preserving the scheme's security: https becomes wss, http (only
// ever a local facade) becomes ws.
func trainStreamURL(endpoint string, jobID string, resume string) (string, error) {
	parsed, err := url.Parse(strings.TrimRight(endpoint, "/"))
	if err != nil {
		return "", fmt.Errorf("parse endpoint %q: %w", endpoint, err)
	}
	switch parsed.Scheme {
	case "https", "wss":
		parsed.Scheme = "wss"
	case "http", "ws":
		parsed.Scheme = "ws"
	default:
		return "", fmt.Errorf("endpoint %q must be http or https", endpoint)
	}
	parsed.Path = strings.TrimRight(parsed.Path, "/") + "/openai/v1/fine_tuning/jobs/" +
		url.PathEscape(jobID) + "/stream"
	if resume != "" {
		query := parsed.Query()
		query.Set("resume", resume)
		parsed.RawQuery = query.Encode()
	}
	return parsed.String(), nil
}

// followTrainingRun mirrors a running job's artifacts into logsRoot until the job
// ends, reporting progress on out. It returns the job's terminal status.
func followTrainingRun(
	ctx context.Context,
	endpoint string,
	authorization string,
	jobID string,
	logsRoot string,
	out io.Writer,
) (string, error) {
	mirror, err := newRunMirror(logsRoot, jobID)
	if err != nil {
		return "", err
	}

	address, err := trainStreamURL(endpoint, jobID, mirror.resumeQuery())
	if err != nil {
		return "", err
	}

	headers := http.Header{}
	headers.Set("Authorization", authorization)

	connection, response, err := dialTrainStream(ctx, address, headers)
	if err != nil {
		return "", newTrainStreamDialError(err, response)
	}
	defer connection.Close()

	fmt.Fprintf(out, "Streaming run artifacts to %s\n", mirror.directory)
	fmt.Fprintf(out, "View them with: python dashboard_server.py --root %s\n", logsRoot)

	seen := map[string]bool{}
	status := ""
	for {
		if err := connection.SetReadDeadline(time.Now().Add(trainStreamReadTimeout)); err != nil {
			return status, fmt.Errorf("set stream read deadline: %w", err)
		}
		messageType, payload, err := connection.ReadMessage()
		if err != nil {
			if websocket.IsCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) {
				return status, nil
			}
			return status, fmt.Errorf("read from the run stream: %w", err)
		}
		if messageType != websocket.TextMessage {
			continue
		}

		var frame trainStreamFrame
		if err := json.Unmarshal(payload, &frame); err != nil {
			return status, fmt.Errorf("decode a run stream frame: %w", err)
		}

		switch frame.Type {
		case "hello":
			status = frame.Status
		case "chunk":
			if err := mirror.apply(frame); err != nil {
				return status, err
			}
			if !seen[frame.Name] {
				seen[frame.Name] = true
				fmt.Fprintf(out, "  receiving %s\n", frame.Name)
			}
		case "status":
			status = frame.Status
			fmt.Fprintf(out, "Job is %s.\n", frame.Status)
		case "end":
			status = frame.Status
			reportStreamCompletion(out, frame, mirror)
			return status, nil
		}
	}
}

func reportStreamCompletion(out io.Writer, frame trainStreamFrame, mirror *runMirror) {
	names := make([]string, 0, len(mirror.offsets)+len(mirror.digests))
	for name := range mirror.offsets {
		names = append(names, name)
	}
	for name := range mirror.digests {
		if _, already := mirror.offsets[name]; !already {
			names = append(names, name)
		}
	}
	sort.Strings(names)

	fmt.Fprintf(out, "Job finished with status %s.\n", frame.Status)
	if len(names) > 0 {
		fmt.Fprintf(out, "Mirrored %s to %s\n", strings.Join(names, ", "), mirror.directory)
	}
	if len(frame.Error) > 0 && string(frame.Error) != "null" {
		fmt.Fprintf(out, "Error: %s\n", string(frame.Error))
	}
}

func newTrainStreamDialError(cause error, response *http.Response) error {
	suggestion := "Check that the fine-tuning endpoint supports run streaming and that you " +
		"can reach it. The job is unaffected; re-attach with --follow or poll it with " +
		"azd ai rle jobs."
	if response != nil {
		body, _ := io.ReadAll(io.LimitReader(response.Body, 2048))
		response.Body.Close()
		return &azdext.LocalError{
			Message: fmt.Sprintf(
				"Could not open the run stream (HTTP %d): %s", response.StatusCode, strings.TrimSpace(string(body))),
			Code:       "rle_train_stream_unavailable",
			Category:   azdext.LocalErrorCategoryUser,
			Suggestion: suggestion,
		}
	}
	return &azdext.LocalError{
		Message:    fmt.Sprintf("Could not open the run stream: %v", cause),
		Code:       "rle_train_stream_unavailable",
		Category:   azdext.LocalErrorCategoryUser,
		Suggestion: suggestion,
	}
}

// defaultLogsRoot matches the roots the cookbook dashboard scans, so a followed
// run shows up in the dropdown without anyone passing a path.
func defaultLogsRoot() string {
	if root := strings.TrimSpace(os.Getenv("LOOM_LOGS_ROOT")); root != "" {
		return root
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return filepath.Join(os.TempDir(), "loom-runs")
	}
	return filepath.Join(home, "loom-runs")
}
