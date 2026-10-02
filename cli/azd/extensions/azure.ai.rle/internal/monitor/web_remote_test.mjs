// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import test from "node:test";
import assert from "node:assert/strict";
import { remoteRolloutState, remoteRunNotices, runFacts, runCharts, fetchSnapshot } from "./web/data.mjs";

test("remote execution state comes from metadata, not the task verdict", () => {
  assert.equal(remoteRolloutState({ status: "completed", success: false }), "completed");
  assert.equal(remoteRolloutState({ status: "running", success: true }), "running");
  assert.equal(remoteRolloutState({ success: true }), "unknown");
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
