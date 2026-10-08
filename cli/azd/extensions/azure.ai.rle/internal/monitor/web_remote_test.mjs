// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import test from "node:test";
import assert from "node:assert/strict";
import {
  remoteRolloutState, remoteRunNotices, rolloutAttemptSessions, runFacts, runCharts, fetchSnapshot,
  unavailableRolloutFacts, unavailableRolloutMessage,
} from "./web/data.mjs";

test("remote execution state comes from metadata, not the task verdict", () => {
  assert.equal(remoteRolloutState({ status: "completed", success: false }), "completed");
  assert.equal(remoteRolloutState({ status: "running", success: true }), "running");
  assert.equal(remoteRolloutState({ success: true }), "unknown");
});

test("retry attempts are separated by each rollout's training session", () => {
  const attempts = rolloutAttemptSessions([
    { rollout_id: "first", session_id: "session-initial" },
    { rollout_id: "second", session_id: "session-initial" },
    { rollout_id: "third", session_id: "session-retry-1" },
    { rollout_id: "fourth", session_id: "session-retry-2" },
  ]);
  assert.deepEqual([...attempts], [
    ["first", "session-initial"], ["second", "session-initial"],
    ["third", "session-retry-1"], ["fourth", "session-retry-2"],
  ]);
  assert.equal(rolloutAttemptSessions([{ rollout_id: "only", session_id: "one-session" }]).size, 0);
  assert.equal(rolloutAttemptSessions([{ rollout_id: "unknown" }, { rollout_id: "retry", session_id: "known" }]).size, 0);
});

test("registered config does not require a facade run_meta file", () => {
  const facts = runFacts({ config: { model_name: "Qwen/Qwen3-32B", learning_rate: 0.0003, max_steps: 60 } });
  assert.ok(facts.model.some(([key, value]) => key === "Base model" && value === "Qwen/Qwen3-32B"));
});

test("remote notices surface pending, stale, bounded and paused states", () => {
  const notes = remoteRunNotices({
    run: { config: null, meta: { status: "running" } },
    states: { metrics: { error: "HTTP 403", updated_at: "2026-10-01T00:00:00Z" } },
    limited: true, paused: "completed",
  }).join(" ");
  assert.match(notes, /registration/);
  assert.match(notes, /HTTP 403/);
  assert.match(notes, /Last successful read/);
  assert.match(notes, /not the full run/);
  assert.match(notes, /paused after completion/);
  assert.match(remoteRunNotices({ run: { config: {} }, paused: "disabled" }).join(" "), /polling stopped/);
});

test("sparse real-service metrics keep their reported steps", () => {
  const chart = runCharts([{ step: 9, "env/all/reward/total": 0.5 }, { step: 15, "optim/lr": 0.001 }])
    .find((entry) => entry.id === "reward");
  assert.deepEqual(chart.series[0].points, [{ step: 9, value: 0.5 }]);
});

test("selected rollout fetch accepts unavailable metadata without inventing a graph", async () => {
  const calls = [];
  const payload = { unavailable: true, metadata: { status: "running" } };
  const result = await fetchSnapshot(async (url) => {
    calls.push(url);
    return { ok: true, json: async () => payload };
  }, "abc");
  assert.deepEqual(calls, ["/api/rollout?id=abc"]);
  assert.equal(result, payload);
  assert.equal(result.response, undefined);
});

test("an unavailable rollout explains its state and summarizes service metadata", () => {
  const failed = {
    rollout_id: "77ddddcf0bd44cc99e5642c3dd022e4d", job_id: "ftjob-1", environment_name: "math_rl",
    environment_version: "1.0.8", created_at_utc: "2026-10-02T22:05:18Z", status: "failed",
    latency_s: 22.1011194, checkpoint_id: "step0", session_id: "session_1a74b5e9", has_graph: false,
  };
  assert.match(unavailableRolloutMessage(failed), /failed before producing a result/);
  assert.match(unavailableRolloutMessage({ status: "running" }), /still running/);
  assert.match(unavailableRolloutMessage({ status: "completed" }), /not retained or has expired/);
  assert.match(unavailableRolloutMessage({}), /not available/);
  assert.deepEqual(unavailableRolloutFacts(failed), [
    ["State", "Failed"], ["Environment", "math_rl 1.0.8"], ["Checkpoint", "step0"], ["Latency", "22.1s"],
    ["Created", "2026-10-02T22:05:18Z"], ["Session", "session_1a74b5e9"], ["Job", "ftjob-1"],
  ]);
  assert.deepEqual(unavailableRolloutFacts({ status: "expired", reward: 0 }), [["State", "expired"], ["Reward", "0.000"]]);
});