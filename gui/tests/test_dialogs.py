from __future__ import annotations

import unittest
from datetime import date
from unittest.mock import Mock, patch

from gui.app import DraughtsApp
from gui.i18n import _tr_static


class Variable:
    def __init__(self, value=None):
        self.value = value
        self.traces = []

    def get(self):
        return self.value

    def set(self, value):
        self.value = value
        for callback in self.traces:
            callback()

    def trace_add(self, _mode, callback):
        self.traces.append(callback)


class DialogTests(unittest.TestCase):
    def setUp(self):
        self.app = Mock()
        self.app._cjk_family = "test font"
        self.app._tr = lambda key, **kwargs: _tr_static(key, "zh").format(**kwargs)
        self.app.controller.players = {}

    def test_forfeit_confirm_cancel_and_window_close(self):
        for choice, expected in (("confirm", True), ("cancel", False), ("close", False)):
            with self.subTest(choice=choice):
                buttons = []

                def make_button(*_args, **kwargs):
                    button = Mock()
                    button.invoke.side_effect = kwargs["command"]
                    buttons.append(button)
                    return button

                with patch("gui.app.tk.Toplevel") as top, \
                     patch("gui.app.ttk.Frame"), patch("gui.app.tk.Label"), \
                     patch("gui.app.font.Font"), \
                     patch("gui.app.tk.Button", side_effect=make_button):
                    def choose(_dialog):
                        if choice == "close":
                            top.return_value.protocol.call_args.args[1]()
                        else:
                            buttons[0 if choice == "confirm" else 1].invoke()

                    self.app.wait_window.side_effect = choose
                    result = DraughtsApp.show_forfeit_dialog(self.app, "opponent error", 0)
                self.assertEqual(result, expected)
                self.app.controller.forfeit.assert_not_called()

    def test_device_selection_and_cpu_fp16_state_reach_new_game(self):
        combos, checks, buttons = [], [], []

        def widget_factory(collection):
            def create(*_args, **kwargs):
                widget = Mock()
                widget.options = kwargs.copy()
                widget.configure.side_effect = lambda **values: widget.options.update(values)
                collection.append(widget)
                return widget
            return create

        def interact(_dialog):
            black, _white, device = combos
            black.options["textvariable"].set(self.app._tr("kind_mcts"))
            self.assertEqual(checks[0].options["state"], "disabled")
            device.options["textvariable"].set("cuda:1")
            self.assertEqual(checks[0].options["state"], "normal")
            checks[0].options["variable"].set(True)
            device.options["textvariable"].set("cpu")
            self.assertFalse(checks[0].options["variable"].get())
            self.assertEqual(checks[0].options["state"], "disabled")
            device.options["textvariable"].set("cuda:1")
            buttons[0].options["command"]()

        self.app.ask_checkpoint_path.return_value = "model.pt"
        self.app.ask_mcts_config_path.return_value = None
        self.app.wait_window.side_effect = interact
        with patch("gui.app.tk.Toplevel"), patch("gui.app.ttk.Frame"), \
             patch("gui.app.ttk.Label"), \
             patch("gui.app.ttk.Entry"), \
             patch("gui.app.tk.StringVar", side_effect=Variable), \
             patch("gui.app.tk.BooleanVar", side_effect=Variable), \
             patch("gui.app.ttk.Combobox", side_effect=widget_factory(combos)), \
             patch("gui.app.ttk.Checkbutton", side_effect=widget_factory(checks)), \
             patch("gui.app.ttk.Button", side_effect=widget_factory(buttons)), \
             patch("gui.app.available_devices", return_value=["cpu", "cuda:0", "cuda:1"]):
            DraughtsApp.show_mode_dialog(self.app, "human", "human")
        arguments = self.app.controller.start_new_game.call_args
        self.assertEqual(arguments.args[:2], ("mcts", "human"))
        self.assertEqual(arguments.kwargs["device"], "cuda:1")
        self.assertFalse(arguments.kwargs["force_fp16"])
        self.assertEqual(arguments.kwargs["first_player"], "先手方")
        self.assertEqual(arguments.kwargs["second_player"], "后手方")

    def test_export_uses_snapshot_even_if_autoplay_continues_in_file_chooser(self):
        from draughts.notation import GameRecord
        from draughts_env import Move

        record = GameRecord()
        snapshot = record.snapshot()
        self.app.controller.record_snapshot.return_value = snapshot

        def choose_file(**_kwargs):
            record.moves.append(Move(0, (20, 24), ()))
            return "record.txt"

        with patch("gui.app.filedialog.asksaveasfilename", side_effect=choose_file), \
             patch.object(snapshot, "write") as write:
            DraughtsApp._on_export_notation(self.app)
        write.assert_called_once_with("record.txt")
        self.assertFalse(snapshot.moves)

    def test_export_cancel_does_not_write_and_errors_are_reported(self):
        snapshot = self.app.controller.record_snapshot.return_value
        snapshot.played_on = date(2026, 10, 18)
        with patch("gui.app.filedialog.asksaveasfilename", return_value=""):
            DraughtsApp._on_export_notation(self.app)
        snapshot.write.assert_not_called()
        snapshot.write.side_effect = OSError("disk error")
        with patch("gui.app.filedialog.asksaveasfilename", return_value="record.txt"), \
             patch("gui.app.messagebox.showerror") as error:
            DraughtsApp._on_export_notation(self.app)
        error.assert_called_once()


if __name__ == "__main__":
    unittest.main()
