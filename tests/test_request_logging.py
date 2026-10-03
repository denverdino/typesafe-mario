import hashlib
import json
from dataclasses import replace

import pytest
from test_control_semantics import MovingEnv
from test_policy import Provider, policy

from typesafe_mario.policy import HeuristicPolicy
from typesafe_mario.runner import run_episode


def read_log(folder):
    return [json.loads(line) for line in next(folder.glob("*.jsonl")).read_text().splitlines()]


def run(folder, monkeypatch, p, env=None):
    env = env or MovingEnv()
    monkeypatch.setattr("typesafe_mario.runner.create_mario_env", lambda *a, **k: env)
    run_episode(
        env_id="test",
        policy=p,
        frames_per_decision=8,
        max_decisions=2,
        seed=123,
        artifacts_dir=folder,
        display="none",
    )
    return env, read_log(folder)


def test_requests_are_logged_before_sdk_and_executed_state_is_exact(tmp_path, monkeypatch):
    sent = []

    class InspectingProvider(Provider):
        def system_one(self, **request):
            sent.append(json.loads(json.dumps(request)))
            rows = read_log(tmp_path)
            records = [r for r in rows if r["record_type"] == "policy_request"]
            assert records, "Request evidence must be flushed before invoking the SDK"
            assert records[-1]["request"]["payload"] == request
            self.choice = next(iter(request["questions"]["next_action"]["criteria"]))
            return super().system_one(**request)

    _, rows = run(tmp_path, monkeypatch, policy(InspectingProvider()))
    requests = [r for r in rows if r["record_type"] == "policy_request"]
    decisions = [r for r in rows if r["record_type"] == "decision"]
    assert len(requests) == len(decisions) == len(sent) == 2
    outcomes = [r for r in rows if r["record_type"] == "request_outcome"]
    assert [r["status"] for r in outcomes] == ["received", "applied", "received", "applied"]
    assert requests[0]["request_id"] != requests[1]["request_id"]
    for request, decision, payload in zip(requests, decisions, sent, strict=True):
        assert request["request"]["payload"] == payload
        canonical = json.dumps(payload, allow_nan=False, separators=(",", ":"))
        assert request["request"]["sha256"] == hashlib.sha256(canonical.encode()).hexdigest()
        assert decision["state"] == payload["state"]
        assert decision["request_id"] == request["request_id"]
        assert decision["execution"]["frames"] == 8
        assert decision["action"] in payload["questions"]["next_action"]["criteria"]
        assert decision["execution"]["apply_at_frame"] == request["timing"]["apply_at_frame"]


def test_failed_api_retains_request_without_executing_buttons(tmp_path, monkeypatch):
    class FailingProvider(Provider):
        def system_one(self, **request):
            raise RuntimeError("provider unavailable")

    env = MovingEnv()
    with pytest.raises(RuntimeError, match="provider unavailable"):
        run(tmp_path, monkeypatch, policy(FailingProvider()), env)
    rows = read_log(tmp_path)
    assert rows[0]["record_type"] == "policy_request"
    assert rows[0]["request"]["payload"]["questions"]["next_action"]
    assert rows[-1]["reason"] == "policy_error"
    assert any(r.get("status") == "failed" and r["error_type"] == "RuntimeError" for r in rows)
    assert not env.actions


def test_terminal_discard_keeps_pending_request_evidence(tmp_path, monkeypatch):
    class AllowedProvider(Provider):
        def system_one(self, **request):
            self.choice = next(iter(request["questions"]["next_action"]["criteria"]))
            return super().system_one(**request)

    env, rows = run(tmp_path, monkeypatch, policy(AllowedProvider()), MovingEnv(3))
    requests = [r for r in rows if r["record_type"] == "policy_request"]
    decisions = [r for r in rows if r["record_type"] == "decision"]
    assert len(requests) == 2
    assert len(decisions) == 1
    assert requests[1]["request_id"] != decisions[0]["request_id"]
    assert len(env.actions) == 3
    assert rows[-1]["reason"] == "death"
    discarded = [r for r in rows if r.get("status") == "discarded"]
    assert discarded[0]["request_id"] == requests[1]["request_id"]
    assert discarded[0]["reason"] == "death"


def test_wrong_application_frame_is_rejected_before_execution(tmp_path, monkeypatch):
    class StalePolicy(HeuristicPolicy):
        def prepare(self, snapshot, actions):
            return super().prepare(replace(snapshot, frame_index=snapshot.frame_index + 1), actions)

    env = MovingEnv()
    with pytest.raises(ValueError, match="application frame"):
        run(tmp_path, monkeypatch, StalePolicy(), env)
    assert not env.actions
    assert read_log(tmp_path)[-1]["reason"] == "decision_contract_error"


def test_failed_first_frame_is_not_reported_as_an_applied_request(tmp_path, monkeypatch):
    class BrokenEnv(MovingEnv):
        def step(self, action):
            raise RuntimeError("emulator step failed")

    class AllowedProvider(Provider):
        def system_one(self, **request):
            self.choice = next(iter(request["questions"]["next_action"]["criteria"]))
            return super().system_one(**request)

    with pytest.raises(RuntimeError, match="emulator step failed"):
        run(tmp_path, monkeypatch, policy(AllowedProvider()), BrokenEnv())
    rows = read_log(tmp_path)
    assert not any(r.get("status") == "applied" for r in rows)
    assert rows[-1]["reason"] == "execution_error"
    decision = next(r for r in rows if r["record_type"] == "decision")
    assert decision["execution"]["frames"] == 0
    assert any(
        r.get("status") == "discarded" and r["request_id"] == decision["request_id"] for r in rows
    )
    from typesafe_mario.analysis import summarize_run

    summary = summarize_run(next(tmp_path.glob("*.jsonl")))["episodes"][0]
    assert decision["request_id"] in summary["unapplied_request_ids"]
