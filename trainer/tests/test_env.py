import unittest

from env import BLACK_SIDE, WHITE_SIDE, encode_observation, perspective_square_id, square_to_coord
from env import StandardDraughtsEnv


def channel_sum(observation, channel):
    return sum(sum(row) for row in observation[channel])


class StandardDraughtsEnvTests(unittest.TestCase):
    def test_reset_observation_shape_and_initial_planes(self):
        env = StandardDraughtsEnv()
        observation = env.reset()

        self.assertEqual(len(observation), 6)
        self.assertEqual(len(observation[0]), 10)
        self.assertEqual(len(observation[0][0]), 10)
        self.assertEqual(channel_sum(observation, 0), 20)
        self.assertEqual(channel_sum(observation, 1), 0)
        self.assertEqual(channel_sum(observation, 2), 20)
        self.assertEqual(channel_sum(observation, 3), 0)
        self.assertEqual(channel_sum(observation, 4), 50)
        self.assertEqual(channel_sum(observation, 5), 0)

    def test_action_mask_matches_legal_actions(self):
        env = StandardDraughtsEnv()
        env.reset()
        mask = env.action_mask()
        legal_actions = env.legal_actions()

        self.assertEqual(len(mask), env.max_actions)
        self.assertEqual(sum(mask), len(legal_actions))
        self.assertEqual(len(legal_actions), 9)
        self.assertEqual(mask[:9], [1] * 9)
        self.assertEqual(mask[9], 0)

    def test_step_returns_standard_tuple(self):
        env = StandardDraughtsEnv()
        env.reset()
        observation, reward, done, info = env.step(0)

        self.assertEqual(reward, 0)
        self.assertFalse(done)
        self.assertEqual(info["side_to_move_name"], "white")
        self.assertGreater(info["legal_move_count"], 0)
        self.assertEqual(channel_sum(observation, 0), 20)
        self.assertEqual(channel_sum(observation, 2), 20)
        self.assertEqual(channel_sum(observation, 5), 0)

    def test_clone_replays_history_without_mutating_original(self):
        env = StandardDraughtsEnv()
        env.reset()
        env.step(0)
        cloned = env.clone()

        self.assertEqual(env.action_history(), cloned.action_history())
        self.assertEqual(env.raw_observation(), cloned.raw_observation())

        cloned.step(0)
        self.assertEqual(len(env.action_history()), 1)
        self.assertEqual(len(cloned.action_history()), 2)

    def test_square_to_coord_uses_project_numbering(self):
        self.assertEqual(square_to_coord(1), (0, 1))
        self.assertEqual(square_to_coord(6), (1, 0))
        self.assertEqual(square_to_coord(50), (9, 8))

    def test_perspective_square_id_rotates_white_side(self):
        self.assertEqual(perspective_square_id(1, BLACK_SIDE), 1)
        self.assertEqual(perspective_square_id(1, WHITE_SIDE), 50)
        self.assertEqual(perspective_square_id(50, WHITE_SIDE), 1)

    def test_encode_observation_uses_current_side_perspective(self):
        black_to_move = {"side_to_move": BLACK_SIDE, "squares": [0] * 50}
        black_to_move["squares"][15] = 1
        black_to_move["squares"][30] = 3
        white_to_move = {"side_to_move": WHITE_SIDE, "squares": [0] * 50}
        white_to_move["squares"][34] = 3
        white_to_move["squares"][19] = 1

        self.assertEqual(encode_observation(black_to_move), encode_observation(white_to_move))


if __name__ == "__main__":
    unittest.main()
