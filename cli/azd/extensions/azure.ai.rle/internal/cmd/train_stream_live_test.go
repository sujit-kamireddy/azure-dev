// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package cmd

import (
	"context"
	"io"
	"os"
	"testing"
	"time"
)

// TestLiveRunStreamSurvivesAnIdleGap follows a real running job for longer than
// the read deadline, to prove against the service that a run with nothing to
// report is not mistaken for a dead connection. The unit test alongside it
// simulates the pings; this one checks the service actually sends them.
//
// Skipped unless RLE_LIVE_JOB names a job that is currently running:
//
//	RLE_LIVE_JOB=ftjob-... \
//	RLE_LIVE_ENDPOINT=https://... \
//	RLE_LIVE_TOKEN=$(az account get-access-token \
//	  --resource https://cognitiveservices.azure.com --query accessToken -o tsv) \
//	go test ./internal/cmd/ -run TestLiveRunStreamSurvivesAnIdleGap -v -timeout 5m
func TestLiveRunStreamSurvivesAnIdleGap(t *testing.T) {
	job := os.Getenv("RLE_LIVE_JOB")
	if job == "" {
		t.Skip("set RLE_LIVE_JOB to a running job to check the stream against the service")
	}

	type outcome struct {
		status string
		err    error
	}
	// gorilla's ReadMessage does not observe a context, so the stream is watched
	// from here rather than cancelled. The goroutine outlives the test; the
	// process is about to exit, and a live probe should not fail on teardown.
	finished := make(chan outcome, 1)
	started := time.Now()
	go func() {
		status, err := followTrainingRun(
			context.Background(),
			os.Getenv("RLE_LIVE_ENDPOINT"),
			"Bearer "+os.Getenv("RLE_LIVE_TOKEN"),
			job,
			t.TempDir(),
			io.Discard,
		)
		finished <- outcome{status, err}
	}()

	watchFor := trainStreamReadTimeout + 30*time.Second
	select {
	case result := <-finished:
		if result.err != nil {
			t.Fatalf("the stream ended after %s, well inside a running job: %v",
				time.Since(started).Round(time.Second), result.err)
		}
		t.Logf("the job reached %q after %s, before the idle gap could be observed",
			result.status, time.Since(started).Round(time.Second))
	case <-time.After(watchFor):
		t.Logf("the stream is still open after %s, past the %s read deadline",
			time.Since(started).Round(time.Second), trainStreamReadTimeout)
	}
}
