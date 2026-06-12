import unittest

from draughts_env import DRAUGHTS_GAME_ONGOING, DraughtsEnv


class DraughtsInterfaceTests(unittest.TestCase):
    def test_reset_and_initial_moves(self):
        env = DraughtsEnv()
        observation = env.observation()
        moves = env.legal_moves()

        self.assertEqual(observation["side_to_move_name"], "black")
        self.assertEqual(observation["game_result"], DRAUGHTS_GAME_ONGOING)
        self.assertEqual(observation["legal_move_count"], 9)
        self.assertEqual(len(moves), 9)
        self.assertEqual(moves[0].path, (16, 21))

    def test_step_uses_action_id_and_refreshes_state(self):
        env = DraughtsEnv()
        observation, info = env.step(0)

        self.assertEqual(observation["side_to_move_name"], "white")
        self.assertEqual(info["done"], 0)
        self.assertEqual(info["reward"], 0)
        self.assertEqual(observation["squares"][15], 0)
        self.assertEqual(observation["squares"][20], 1)
        self.assertGreater(observation["legal_move_count"], 0)

    def test_set_state_refreshes_legal_moves(self):
        env = DraughtsEnv()
        squares = [0] * 50
        squares[15] = 1
        squares[25] = 3

        observation = env.set_state(squares, 0)
        moves = env.legal_moves()

        self.assertEqual(observation["side_to_move_name"], "black")
        self.assertEqual(observation["squares"], squares)
        self.assertEqual(observation["legal_move_count"], len(moves))
        self.assertGreater(len(moves), 0)

    def test_invalid_action_raises(self):
        env = DraughtsEnv()

        with self.assertRaises(RuntimeError):
            env.step(999)

    def test_clone_copies_environment_independently(self):
        env = DraughtsEnv()
        env.step(0)
        cloned = env.clone()

        self.assertEqual(env.observation(), cloned.observation())
        cloned.step(0)
        self.assertNotEqual(env.observation(), cloned.observation())


if __name__ == "__main__":
    unittest.main()
