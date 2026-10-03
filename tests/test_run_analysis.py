import json

from typesafe_mario.cli import main


def test_analysis_cli_handles_two_episodes_and_unapplied_requests(tmp_path, capsys):
    path = tmp_path / "run.jsonl"
    rows = []
    for episode, x in (("first", 672), ("second", 992)):
        base = {"schema_version": 2, "episode_id": episode, "run_config": {"seed": 123}}
        rows += [
            dict(base, record_type="policy_request", request_id=episode + ":0"),
            dict(base, record_type="policy_request", request_id=episode + ":1"),
            dict(
                base,
                record_type="decision",
                decision=0,
                request_id=episode + ":0",
                action="right",
                latency_ms=350,
                execution={"frames": 2},
                state={"risk_control": {"candidate_actions": ["right"], "status": "filtered"}},
                result_state={"player": {"x": x}, "episode": {"best_progress": x}},
            ),
            dict(
                base,
                record_type="episode_end",
                reason="death",
                frames=2,
                result_state={
                    "player": {"x": x},
                    "episode": {"best_progress": x, "stage_clear": False},
                },
            ),
        ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert main(["analyze-log", str(path), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)["runs"][0]
    assert result["cleared_episodes"] == 0
    assert [e["furthest_x"] for e in result["episodes"]] == [672, 992]
    assert result["episodes"][0]["unapplied_request_ids"] == ["first:1"]
    assert result["episodes"][1]["single_candidate_decisions"] == 1


def test_incomplete_episode_is_not_reported_as_a_death_or_clear(tmp_path, capsys):
    path = tmp_path / "interrupted.jsonl"
    path.write_text(json.dumps({"episode_id": "e", "record_type": "policy_request"}) + "\n")
    main(["analyze-log", str(path), "--json"])
    episode = json.loads(capsys.readouterr().out)["runs"][0]["episodes"][0]
    assert episode["outcome"] == "incomplete"
    assert episode["prediction_comparisons"]["executed_cycle"]["samples"] == 0


def test_interrupted_cycle_preserves_explicit_application_evidence(tmp_path, capsys):
    path = tmp_path / "interrupted-after-first-frame.jsonl"
    rows = [
        {"episode_id": "e", "record_type": "policy_request", "request_id": "e:0"},
        {
            "episode_id": "e",
            "record_type": "request_outcome",
            "request_id": "e:0",
            "status": "applied",
        },
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    main(["analyze-log", str(path), "--json"])
    episode = json.loads(capsys.readouterr().out)["runs"][0]["episodes"][0]
    assert episode["outcome"] == "incomplete"
    assert episode["unapplied_request_ids"] == []
    assert episode["unfinalized_applied_request_ids"] == ["e:0"]
