import argparse
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from arena import ArenaStats
from compare_checkpoints import (
    Score,
    compare_checkpoints,
    load_compare_config,
    merge_compare_args,
    paired_score,
    parse_external_agents,
    recommend,
    write_report,
)


class CompareCheckpointsTests(unittest.TestCase):
    def test_load_compare_config(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "compare.yaml"
            path.write_text(
                "preset: custom\ngames: 4\nmax_plies: 50\nlog_interval: 2\n"
                "seeds: 1,2\nopening_random_plies: 0,4\n"
                "external_agents: material,alphabeta\n"
                "promote_threshold: 0.55\nreject_threshold: 0.49\nbucket_floor: 0.44\n"
                "report: none\nmaterial_config: none\nalphabeta_config: none\ndevice: cpu\n",
                encoding="utf-8",
            )

            config = load_compare_config(path)

        self.assertEqual(config.games, 4)
        self.assertEqual(config.preset, "custom")
        self.assertEqual(config.seeds, "1,2")
        self.assertEqual(config.opening_random_plies, "0,4")
        self.assertEqual(config.device, "cpu")
        self.assertAlmostEqual(config.promote_threshold, 0.55)

    def test_quick_preset_overrides_config_defaults(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "compare.yaml"
            path.write_text(
                "preset: custom\ngames: 200\nmax_plies: 500\nlog_interval: 100\n"
                "seeds: 2026\nopening_random_plies: 4\nexternal_agents: material,alphabeta\n"
                "promote_threshold: 0.53\nreject_threshold: 0.50\nbucket_floor: 0.45\n"
                "report: none\nmaterial_config: none\nalphabeta_config: none\ndevice: none\n",
                encoding="utf-8",
            )
            args = merge_compare_args(
                argparse.Namespace(
                    config=str(path),
                    candidate="candidate.pt",
                    baseline="baseline.pt",
                    preset="quick",
                    games=None,
                    max_plies=None,
                    log_interval=None,
                    seeds=None,
                    opening_random_plies=None,
                    external_agents=None,
                    promote_threshold=None,
                    reject_threshold=None,
                    bucket_floor=None,
                    report=None,
                    material_config=None,
                    alphabeta_config=None,
                    device=None,
                )
            )

        self.assertEqual(args.preset, "quick")
        self.assertEqual(args.games, 50)
        self.assertEqual(args.seeds, "2028")
        self.assertEqual(args.opening_random_plies, "4")
        self.assertEqual(args.external_agents, "")

    def test_parse_external_agents_rejects_supervised(self):
        self.assertEqual(parse_external_agents("random,material"), ["random", "material"])
        with self.assertRaises(ValueError):
            parse_external_agents("supervised")

    def test_paired_score_aggregates_color_swap(self):
        model_black = ArenaStats(games=10, black_wins=6, white_wins=3, max_plies_reached=1, total_plies=100)
        opponent_black = ArenaStats(games=10, black_wins=4, white_wins=5, draws=1, total_plies=110)

        score = paired_score(model_black, opponent_black)

        self.assertEqual(score.candidate_wins, 11)
        self.assertEqual(score.baseline_wins, 7)
        self.assertEqual(score.max_plies, 1)
        self.assertEqual(score.games, 20)
        self.assertAlmostEqual(score.candidate_win_rate, 11 / 18)

    def test_recommendation_thresholds(self):
        args = argparse.Namespace(promote_threshold=0.53, reject_threshold=0.50, bucket_floor=0.45)
        promote_score = Score(candidate_wins=60, baseline_wins=40)
        weak_bucket = {"score": {"candidate_wins": 40, "baseline_wins": 60}}

        self.assertEqual(recommend(promote_score, [], args), "promote")
        self.assertEqual(recommend(Score(candidate_wins=51, baseline_wins=49), [], args), "inconclusive")
        self.assertEqual(recommend(Score(candidate_wins=49, baseline_wins=51), [], args), "reject")
        self.assertEqual(recommend(promote_score, [weak_bucket], args), "inconclusive")

    def test_compare_checkpoints_writes_expected_report_shape(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            candidate = tmp_path / "candidate.pt"
            baseline = tmp_path / "baseline.pt"
            report_path = tmp_path / "report.json"
            candidate.write_text("candidate", encoding="utf-8")
            baseline.write_text("baseline", encoding="utf-8")
            args = argparse.Namespace(
                candidate=str(candidate),
                baseline=str(baseline),
                preset="custom",
                games=2,
                max_plies=20,
                log_interval=2,
                seeds="1",
                opening_random_plies="0",
                external_agents="random",
                promote_threshold=0.53,
                reject_threshold=0.50,
                bucket_floor=0.45,
                material_config=None,
                alphabeta_config=None,
                device="cpu",
            )
            with patch("compare_checkpoints.run_arena", side_effect=fake_run_arena):
                report = compare_checkpoints(args)
            write_report(report_path, report)
            loaded = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertIn("main", loaded)
        self.assertIn("external", loaded)
        self.assertIn("random", loaded["external"])
        self.assertEqual(loaded["main"]["overall"]["games"], 4)
        self.assertEqual(loaded["settings"]["preset"], "custom")


def fake_run_arena(**kwargs):
    black = kwargs["black_name"]
    white = kwargs["white_name"]
    black_checkpoint = kwargs.get("black_checkpoint_path")
    white_checkpoint = kwargs.get("white_checkpoint_path")
    games = int(kwargs["games"])
    if black == "supervised" and white == "supervised":
        if black_checkpoint and "candidate" in black_checkpoint:
            return ArenaStats(games=games, black_wins=2, white_wins=0, total_plies=20)
        return ArenaStats(games=games, black_wins=0, white_wins=2, total_plies=20)
    if black == "supervised":
        if black_checkpoint and "candidate" in black_checkpoint:
            return ArenaStats(games=games, black_wins=2, white_wins=0, total_plies=20)
        return ArenaStats(games=games, black_wins=1, white_wins=1, total_plies=20)
    if white == "supervised":
        if white_checkpoint and "candidate" in white_checkpoint:
            return ArenaStats(games=games, black_wins=0, white_wins=2, total_plies=20)
        return ArenaStats(games=games, black_wins=1, white_wins=1, total_plies=20)
    raise AssertionError("unexpected fake arena call")


if __name__ == "__main__":
    unittest.main()
