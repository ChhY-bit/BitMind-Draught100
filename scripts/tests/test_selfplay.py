import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from selfplay import (
    BLACK_SIDE,
    BLACK_WIN,
    WHITE_SIDE,
    WHITE_WIN,
    generate_dataset,
    load_selfplay_config,
    next_game_id,
    parse_args,
    resolve_output_path,
    value_target_for_side,
)


class SelfPlayTests(unittest.TestCase):
    def test_value_target_for_side(self):
        self.assertEqual(value_target_for_side(BLACK_WIN, BLACK_SIDE), 1)
        self.assertEqual(value_target_for_side(BLACK_WIN, WHITE_SIDE), -1)
        self.assertEqual(value_target_for_side(WHITE_WIN, BLACK_SIDE), -1)
        self.assertEqual(value_target_for_side(WHITE_WIN, WHITE_SIDE), 1)
        self.assertEqual(value_target_for_side(0, BLACK_SIDE, maxed=True), 0)

    def test_generate_dataset_writes_jsonl_samples(self):
        with TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "selfplay.jsonl"

            stats = generate_dataset(
                black_name="random",
                white_name="random",
                games=1,
                max_plies=6,
                output_path=str(output),
                log_interval=1,
                seed=1,
            )

            lines = output.read_text(encoding="utf-8").splitlines()

        self.assertEqual(stats.games, 1)
        self.assertEqual(stats.samples, len(lines))
        self.assertGreater(len(lines), 0)

        sample = json.loads(lines[0])
        self.assertEqual(sample["schema_version"], 1)
        self.assertEqual(sample["game_id"], 0)
        self.assertEqual(sample["ply"], 0)
        self.assertIn(sample["side_to_move"], {BLACK_SIDE, WHITE_SIDE})
        self.assertEqual(len(sample["raw_squares"]), 50)
        self.assertEqual(len(sample["observation"]), 6)
        self.assertEqual(len(sample["action_mask"]), 512)
        self.assertIn(sample["selected_action_id"], sample["legal_action_ids"])
        self.assertIn("final_result", sample)
        self.assertIn(sample["value_target"], {-1, 0, 1})
        self.assertIn("selected_move", sample)

    def test_generate_dataset_append_continues_game_id(self):
        with TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "selfplay.jsonl"
            generate_dataset(
                black_name="random",
                white_name="random",
                games=1,
                max_plies=4,
                output_path=str(output),
                log_interval=1,
                seed=1,
            )
            first_lines = output.read_text(encoding="utf-8").splitlines()

            generate_dataset(
                black_name="random",
                white_name="random",
                games=1,
                max_plies=4,
                output_path=str(output),
                log_interval=1,
                seed=2,
                append=True,
            )
            lines = output.read_text(encoding="utf-8").splitlines()
            next_id = next_game_id(output)

            self.assertGreater(len(lines), len(first_lines))
            game_ids = [json.loads(line)["game_id"] for line in lines]
            self.assertEqual(min(game_ids), 0)
            self.assertEqual(max(game_ids), 1)
            self.assertEqual(next_id, 2)

    def test_auto_output_name_avoids_existing_file(self):
        with TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "selfplay.jsonl"
            output.write_text("existing\n", encoding="utf-8")

            resolved = resolve_output_path(output, append=False, auto_output_name=True)
            generate_dataset(
                black_name="random",
                white_name="random",
                games=1,
                max_plies=4,
                output_path=str(output),
                log_interval=1,
                seed=1,
                auto_output_name=True,
            )

            original_lines = output.read_text(encoding="utf-8").splitlines()

            self.assertEqual(resolved.name, "selfplay_001.jsonl")
            self.assertEqual(original_lines, ["existing"])
            self.assertTrue(resolved.exists())

    def test_load_selfplay_config_from_yaml(self):
        with TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "selfplay.yaml"
            config_path.write_text(
                "black: random\nwhite: material\ngames: 2\nmax_plies: 12\n"
                "output: data/custom.jsonl\nlog_interval: 2\nseed: 9\n"
                "material_config: none\nalphabeta_config: none\n"
                "black_checkpoint: none\nwhite_checkpoint: none\ndevice: cpu\n"
                "append: 0\nauto_output_name: 1\n",
                encoding="utf-8",
            )

            config = load_selfplay_config(config_path)

        self.assertEqual(config.black, "random")
        self.assertEqual(config.white, "material")
        self.assertEqual(config.games, 2)
        self.assertEqual(config.max_plies, 12)
        self.assertEqual(config.output, "data/custom.jsonl")
        self.assertEqual(config.device, "cpu")

    def test_parse_args_merges_config_and_cli_overrides(self):
        with TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "selfplay.yaml"
            output_path = Path(tmp_dir) / "out.jsonl"
            config_path.write_text(
                f"black: random\nwhite: random\ngames: 2\nmax_plies: 12\n"
                f"output: {output_path}\nlog_interval: 2\nseed: 9\n"
                "material_config: none\nalphabeta_config: none\n"
                "black_checkpoint: none\nwhite_checkpoint: none\ndevice: none\n"
                "append: 0\nauto_output_name: 1\n",
                encoding="utf-8",
            )

            args = self.parse_with_argv(
                "--config",
                str(config_path),
                "--games",
                "3",
                "--append",
            )

        self.assertEqual(args.games, 3)
        self.assertTrue(args.append)
        self.assertFalse(args.auto_output_name)

    def parse_with_argv(self, *args):
        old_argv = sys.argv
        sys.argv = ["selfplay.py", *args]
        try:
            return parse_args()
        finally:
            sys.argv = old_argv


if __name__ == "__main__":
    unittest.main()
