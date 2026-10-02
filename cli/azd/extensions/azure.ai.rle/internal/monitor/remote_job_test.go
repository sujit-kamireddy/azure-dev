// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package monitor

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"azure.ai.rle/internal/rollouts"
)

const remoteTestJob = "ftjob-7898670d3ae8453e89e591f9"
const remoteTestRollout = "3c27c30f5fba261c3a7a3e856b4e1388"

var remoteTestTime = time.Date(2100, 1, 1, 0, 0, 0, 0, time.UTC)

type memoryJobSource struct {
	configCalls, metricCalls, listCalls, detailCalls, resultCalls, statusCalls int

	configErr, resultErr error
	status               string
	entry                JobRollout
	metrics              func(int64, string) (JobPage[json.RawMessage], error)
	rollouts             func(int64, string) (JobPage[JobRollout], error)
}

func newMemoryJobSource() *memoryJobSource {
	return &memoryJobSource{
		status: "running",
		entry: JobRollout{
			RolloutID: remoteTestRollout, JobID: remoteTestJob, Sequence: new(int64(0)),
			Status: "running", Success: new(false),
		},
	}
}

func (s *memoryJobSource) Config(context.Context) (json.RawMessage, error) {
	s.configCalls++
	return json.RawMessage(`{"job_id":"` + remoteTestJob + `","max_steps":60}`), s.configErr
}

func (s *memoryJobSource) Metrics(_ context.Context, after int64, token string) (JobPage[json.RawMessage], error) {
	s.metricCalls++
	if s.metrics != nil {
		return s.metrics(after, token)
	}
	return JobPage[json.RawMessage]{Data: []json.RawMessage{json.RawMessage(`{"step":0,"optim/lr":0.0003}`)}}, nil
}

func (s *memoryJobSource) Rollouts(_ context.Context, after int64, token string) (JobPage[JobRollout], error) {
	s.listCalls++
	if s.rollouts != nil {
		return s.rollouts(after, token)
	}
	return JobPage[JobRollout]{Data: []JobRollout{s.entry}}, nil
}

func (s *memoryJobSource) Detail(context.Context, string) (JobRollout, error) {
	s.detailCalls++
	return s.entry, nil
}

func (s *memoryJobSource) Result(context.Context, JobRollout) (rollouts.Snapshot, error) {
	s.resultCalls++
	return rollouts.Snapshot{Response: json.RawMessage(`{"rollout_id":"` + remoteTestRollout + `","reward":1}`)}, s.resultErr
}

func (s *memoryJobSource) Status(context.Context) (string, error) {
	s.statusCalls++
	return s.status, nil
}

func remoteRequest(t *testing.T, handler http.Handler, method, path string) *httptest.ResponseRecorder {
	t.Helper()
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, httptest.NewRequest(method, "http://"+testHost+path, nil))
	return recorder
}

func TestRemoteEmptyRolloutListIsArray(t *testing.T) {
	j := newRemoteJob(newMemoryJobSource(), remoteTestJob)
	handler, err := newHandler(source{remote: j, jobID: remoteTestJob}, testHost)
	if err != nil {
		t.Fatal(err)
	}
	response := remoteRequest(t, handler, "GET", "/api/rollouts")
	if response.Code != 200 || !strings.Contains(response.Body.String(), `"data":[]`) {
		t.Fatalf("empty rollout list must serialize as []: %d %s", response.Code, response.Body)
	}
}

func TestRemoteMonitorPollsSummariesButNeverResultsUntilOpened(t *testing.T) {
	reader := newMemoryJobSource()
	j := newRemoteJob(reader, remoteTestJob)
	handler, err := newHandler(source{remote: j, jobID: remoteTestJob}, testHost)
	if err != nil {
		t.Fatal(err)
	}
	now := remoteTestTime
	j.refresh(t.Context(), now, false)
	for range 3 {
		for _, path := range []string{"/api/run", "/api/run/metrics", "/api/rollouts"} {
			if response := remoteRequest(t, handler, "GET", path); response.Code != 200 {
				t.Fatalf("%s: %d %s", path, response.Code, response.Body)
			}
		}
	}
	if reader.resultCalls != 0 || reader.configCalls != 1 || reader.metricCalls != 1 {
		t.Fatalf("browser reads triggered remote calls: %+v", reader)
	}
	// The next full reconciliation picks up the status change.
	reader.entry.Status, reader.entry.ResultAvailable = "completed", true
	reader.entry.Reward = new(0.5)
	j.refresh(t.Context(), now.Add(15*time.Second), false)
	response := remoteRequest(t, handler, "GET", "/api/rollouts")
	if !strings.Contains(response.Body.String(), `"status":"completed"`) ||
		!strings.Contains(response.Body.String(), `"success":false`) {
		t.Fatal(response.Body.String())
	}
	if reader.resultCalls != 0 || reader.detailCalls != 0 {
		t.Fatal("polling read rollout details or results")
	}
	response = remoteRequest(t, handler, "GET", "/api/rollout?id="+remoteTestRollout)
	if response.Code != 200 || reader.resultCalls != 1 {
		t.Fatalf("click failed: %d %s", response.Code, response.Body)
	}
	reader.resultErr = &ReadError{Status: 404, Code: "RolloutResultNotFound"}
	response = remoteRequest(t, handler, "GET", "/api/rollout?id="+remoteTestRollout)
	if !strings.Contains(response.Body.String(), `"unavailable":true`) {
		t.Fatal(response.Body.String())
	}
	reader.entry.JobID = "another-job"
	response = remoteRequest(t, handler, "GET", "/api/rollout?id="+remoteTestRollout)
	if response.Code != http.StatusNotFound || reader.resultCalls != 2 {
		t.Fatal("cross-job result exposed")
	}
}

func TestRemoteMonitorRegistrationAndSchedule(t *testing.T) {
	reader := newMemoryJobSource()
	reader.configErr = &ReadError{Status: 404, Code: "JobNotFound"}
	j := newRemoteJob(reader, remoteTestJob)
	now := remoteTestTime
	j.refresh(t.Context(), now, false)
	if j.config != nil || !strings.Contains(j.states["config"].Error, "Waiting for job registration") {
		t.Fatal("registration delay not represented")
	}
	j.refresh(t.Context(), now.Add(time.Second), false)
	if reader.configCalls != 1 || reader.metricCalls != 1 {
		t.Fatal("polled before due")
	}
	reader.configErr = nil
	j.refresh(t.Context(), now.Add(5*time.Second), false)
	if reader.configCalls != 2 || reader.metricCalls != 2 || reader.statusCalls != 1 {
		t.Fatalf("unexpected cadence: %+v", reader)
	}
	j.refresh(t.Context(), now.Add(10*time.Second), false)
	j.refresh(t.Context(), now.Add(15*time.Second), false)
	if reader.configCalls != 2 || reader.statusCalls != 2 || reader.detailCalls != 0 {
		t.Fatalf("unexpected cadence: %+v", reader)
	}
}

func TestRemoteMetricPagingKeepsWatermarkFixedAndRecovers(t *testing.T) {
	reader := newMemoryJobSource()
	fail := true
	reader.metrics = func(after int64, token string) (JobPage[json.RawMessage], error) {
		if after != -1 {
			t.Fatalf("watermark moved inside cycle: %d", after)
		}
		if token == "" {
			return JobPage[json.RawMessage]{Data: []json.RawMessage{json.RawMessage(`{"step":3}`)}, Next: "next"}, nil
		}
		if fail {
			return JobPage[json.RawMessage]{}, errors.New("temporary read failure")
		}
		return JobPage[json.RawMessage]{Data: []json.RawMessage{json.RawMessage(`{"step":7}`)}}, nil
	}
	j := newRemoteJob(reader, remoteTestJob)
	now := remoteTestTime
	if err := j.readMetrics(t.Context(), now); err == nil {
		t.Fatal("expected failure")
	}
	if j.metricCursor.watermark != -1 || len(j.metrics) != 1 {
		t.Fatal("cursor committed early")
	}
	fail = false
	if err := j.readMetrics(t.Context(), now); err != nil {
		t.Fatal(err)
	}
	if j.metricCursor.watermark != 7 || len(j.metrics) != 2 {
		t.Fatal("retry skipped or duplicated rows")
	}
	reader.metrics = func(after int64, token string) (JobPage[json.RawMessage], error) {
		if after != -1 {
			t.Fatal("reconciliation must include late lower steps")
		}
		return JobPage[json.RawMessage]{Data: []json.RawMessage{json.RawMessage(`{"step":1}`)}}, nil
	}
	if err := j.readMetrics(t.Context(), now.Add(time.Minute)); err != nil {
		t.Fatal(err)
	}
	if len(j.metrics) != 3 || j.metricCursor.watermark != 7 {
		t.Fatal("late step lost")
	}
}

func TestRemotePagingEmptyPagesRepeatedTokensAndSameSequence(t *testing.T) {
	reader := newMemoryJobSource()
	reader.rollouts = func(after int64, token string) (JobPage[JobRollout], error) {
		if after != -1 {
			t.Fatal("sequence advanced between pages")
		}
		switch token {
		case "":
			return JobPage[JobRollout]{Next: "empty"}, nil
		case "empty":
			return JobPage[JobRollout]{Data: []JobRollout{reader.entry}, Next: "second"}, nil
		default:
			entry := reader.entry
			entry.RolloutID = "4c27c30f5fba261c3a7a3e856b4e1388"
			return JobPage[JobRollout]{Data: []JobRollout{entry}}, nil
		}
	}
	j := newRemoteJob(reader, remoteTestJob)
	if err := j.readRollouts(t.Context(), time.Now()); err != nil {
		t.Fatal(err)
	}
	if len(j.entries) != 2 {
		t.Fatal("same-sequence row skipped")
	}
	reader.metrics = func(int64, string) (JobPage[json.RawMessage], error) {
		return JobPage[json.RawMessage]{Next: "repeat"}, nil
	}
	if err := j.readMetrics(t.Context(), time.Now()); err == nil {
		t.Fatal("expected repeated-token error")
	}
}

func TestRemoteMonitorBackoffAndBoundedMetrics(t *testing.T) {
	j := newRemoteJob(newMemoryJobSource(), remoteTestJob)
	now := remoteTestTime
	calls := 0
	read := func() error {
		calls++
		return &ReadError{Status: 429, RetryAfter: time.Minute}
	}
	j.poll(t.Context(), now, "metrics", 5*time.Second, read)
	j.poll(t.Context(), now.Add(30*time.Second), "metrics", 5*time.Second, read)
	if calls != 1 || j.states["metrics"].Error == "" {
		t.Fatal("Retry-After ignored")
	}
	j.refresh(t.Context(), now.Add(30*time.Second), true)
	j.poll(t.Context(), now.Add(30*time.Second), "metrics", 5*time.Second, read)
	if calls != 1 {
		t.Fatal("manual refresh ignored Retry-After")
	}
	for step := range int64(maxRunMetricLines + 10) {
		j.metrics[step] = json.RawMessage(`{"step":0}`)
	}
	j.trimMetrics()
	if len(j.metrics) != maxRunMetricLines || !j.limited || j.metrics[0] != nil || j.metricCursor.floor != 9 {
		t.Fatal("history not bounded")
	}
}

func TestRemoteRefreshRetriesFailedReads(t *testing.T) {
	j := newRemoteJob(newMemoryJobSource(), remoteTestJob)
	now := remoteTestTime
	calls := 0
	read := func() error {
		calls++
		return errors.New("temporary read failure")
	}
	for range 3 {
		j.poll(t.Context(), now, "metrics", 5*time.Second, read)
		j.mu.Lock()
		state := j.states["metrics"]
		state.Next = state.RetryAt // What a manual refresh does.
		j.states["metrics"] = state
		j.mu.Unlock()
	}
	if calls != 3 {
		t.Fatalf("manual refresh did not retry a failed read: %d calls", calls)
	}
}

func TestRemoteFeatureDisabledStopsPolling(t *testing.T) {
	reader := newMemoryJobSource()
	reader.configErr = &ReadError{Status: 403, Code: "FeatureDisabled"}
	j := newRemoteJob(reader, remoteTestJob)
	now := remoteTestTime
	j.refresh(t.Context(), now, false)
	if j.paused != pausedDisabled || !strings.Contains(j.states["config"].Error, "not enabled") {
		t.Fatalf("feature gate not reported: %q %+v", j.paused, j.states["config"])
	}
	calls := reader.configCalls
	j.refresh(t.Context(), now.Add(time.Hour), false)
	if reader.configCalls != calls {
		t.Fatal("disabled feature was polled again")
	}
	j.refresh(t.Context(), now.Add(time.Hour), true)
	if reader.configCalls == calls {
		t.Fatal("manual refresh did not retry")
	}
}

func TestRemoteReconciliationStartsAtRetainedWindow(t *testing.T) {
	reader := newMemoryJobSource()
	var starts []int64
	reader.rollouts = func(after int64, token string) (JobPage[JobRollout], error) {
		starts = append(starts, after)
		var page JobPage[JobRollout]
		for sequence := range int64(remoteRolloutLimit + 5) {
			if sequence > after {
				page.Data = append(page.Data, JobRollout{
					RolloutID: strconv.FormatInt(sequence, 10), JobID: remoteTestJob, Sequence: new(sequence),
				})
			}
		}
		return page, nil
	}
	j := newRemoteJob(reader, remoteTestJob)
	now := remoteTestTime
	for _, at := range []time.Duration{0, time.Second, 15 * time.Second} {
		if err := j.readRollouts(t.Context(), now.Add(at)); err != nil {
			t.Fatal(err)
		}
	}
	// Full, incremental from the watermark, then full again from the oldest retained row.
	want := []int64{-1, remoteRolloutLimit + 4, 4}
	if len(starts) != len(want) || starts[0] != want[0] || starts[1] != want[1] || starts[2] != want[2] {
		t.Fatalf("unexpected reconciliation starts %v, want %v", starts, want)
	}
	if _, kept := j.entries["4"]; len(j.entries) != remoteRolloutLimit || !j.limited || kept {
		t.Fatal("rollout cache not bounded to the newest rows")
	}
}

func TestRemoteCompletionPausesAndRefreshResumes(t *testing.T) {
	reader := newMemoryJobSource()
	reader.status = "succeeded"
	reader.entry.Status, reader.entry.ResultAvailable = "completed", true
	j := newRemoteJob(reader, remoteTestJob)
	now := remoteTestTime
	j.refresh(t.Context(), now, false)
	j.refresh(t.Context(), now.Add(time.Minute), false)
	if j.paused != pausedCompleted {
		t.Fatal("terminal settling and reconciliation did not pause polling")
	}
	calls := reader.metricCalls
	j.refresh(t.Context(), now.Add(2*time.Minute), false)
	if reader.metricCalls != calls {
		t.Fatal("paused monitor still polled")
	}
	j.refresh(t.Context(), now.Add(2*time.Minute), true)
	if reader.metricCalls == calls || j.paused != "" {
		t.Fatal("manual refresh did not resume")
	}
}

func TestRemoteConcurrentBrowserReads(t *testing.T) {
	j := newRemoteJob(newMemoryJobSource(), remoteTestJob)
	handler, err := newHandler(source{remote: j, jobID: remoteTestJob}, testHost)
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	wg.Go(func() {
		for range 20 {
			j.refresh(t.Context(), time.Now(), true)
		}
	})
	for range 3 {
		wg.Go(func() {
			for range 20 {
				remoteRequest(t, handler, "GET", "/api/run/metrics")
			}
		})
	}
	wg.Wait()
}
