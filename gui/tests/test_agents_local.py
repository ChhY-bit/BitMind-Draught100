from __future__ import annotations

import unittest
from unittest.mock import patch

from gui.agents_local import MCTSPlayer, SupervisedPlayer, available_devices


class DeviceTests(unittest.TestCase):
    def test_cpu_only_environment(self):
        with patch("gui.agents_local.torch.cuda.is_available", return_value=False):
            self.assertEqual(available_devices(), ["cpu"])

    def test_all_visible_cuda_devices_are_offered(self):
        with patch("gui.agents_local.torch.cuda.is_available", return_value=True), \
             patch("gui.agents_local.torch.cuda.device_count", return_value=2):
            self.assertEqual(available_devices(), ["cpu", "cuda:0", "cuda:1"])

    def test_gui_adapters_pass_selected_device_to_agents(self):
        for player_type, agent_name in (
            (SupervisedPlayer, "SupervisedAgent"), (MCTSPlayer, "MCTSAgent")
        ):
            with self.subTest(player=player_type.__name__):
                with patch(f"gui.agents_local.{agent_name}") as agent:
                    player_type("test", "model.pt", device="cuda:1")
                self.assertEqual(agent.call_args.kwargs["device"], "cuda:1")


if __name__ == "__main__":
    unittest.main()
