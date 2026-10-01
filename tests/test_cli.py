from unittest.mock import patch

import pytest

from typesafe_mario.cli import build_parser, main


def test_removed_millisecond_option_is_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["play", "--api-run-ms", "100"])


def test_frame_cycle_is_forwarded_without_a_second_timing_control():
    with (
        patch("typesafe_mario.cli.HeuristicPolicy"),
        patch("typesafe_mario.cli.run_episode") as run,
    ):
        assert main(["play", "--policy", "heuristic", "--frames-per-decision", "4"]) == 0
    assert run.call_args.kwargs["frames_per_decision"] == 4
    assert "api_run_ms" not in run.call_args.kwargs


def test_resume_checkpoint_options_are_forwarded():
    from pathlib import Path

    with (
        patch("typesafe_mario.cli.HeuristicPolicy"),
        patch("typesafe_mario.cli.run_episode") as run,
    ):
        assert (
            main(
                [
                    "play",
                    "--policy",
                    "heuristic",
                    "--resume-log",
                    "run.jsonl",
                    "--resume-decision",
                    "152",
                    "--resume-episode",
                    "episode-a",
                ]
            )
            == 0
        )
    assert run.call_args.kwargs["resume_log"] == Path("run.jsonl")
    assert run.call_args.kwargs["resume_decision"] == 152
    assert run.call_args.kwargs["resume_episode"] == "episode-a"
