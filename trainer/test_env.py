import unittest

from env import StandardDraughtsEnv, square_to_coord


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
        self.assertEqual(channel_sum(observation, 5), 50)

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


if __name__ == "__main__":
    unittest.main()
