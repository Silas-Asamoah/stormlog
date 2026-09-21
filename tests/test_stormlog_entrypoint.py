import contextlib
import importlib
import io
import unittest
from unittest import mock

import stormlog.entrypoint as entrypoint


class StormlogEntrypointTests(unittest.TestCase):
    def test_no_args_launches_tui(self) -> None:
        with mock.patch("stormlog.tui.run_app") as run_app:
            exit_code = entrypoint.main([])

        self.assertEqual(exit_code, 0)
        run_app.assert_called_once_with()

    def test_tui_command_launches_tui(self) -> None:
        with mock.patch("stormlog.tui.run_app") as run_app:
            exit_code = entrypoint.main(["tui"])

        self.assertEqual(exit_code, 0)
        run_app.assert_called_once_with()

    def test_tui_help_does_not_launch_tui(self) -> None:
        output = io.StringIO()
        with mock.patch("stormlog.tui.run_app") as run_app:
            with contextlib.redirect_stdout(output):
                with self.assertRaises(SystemExit) as raised:
                    entrypoint.main(["tui", "--help"])

        self.assertEqual(raised.exception.code, 0)
        run_app.assert_not_called()
        self.assertIn("Stormlog Textual TUI", output.getvalue())

    def test_tui_unknown_arg_errors_before_launching_tui(self) -> None:
        with mock.patch("stormlog.tui.run_app") as run_app:
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    entrypoint.main(["tui", "--bad"])

        self.assertEqual(raised.exception.code, 2)
        run_app.assert_not_called()

    def test_infer_command_dispatches_to_infer_cli(self) -> None:
        with mock.patch("stormlog.infer.cli.main", return_value=7) as infer_main:
            exit_code = entrypoint.main(["infer", "analyze", "artifact.jsonl"])

        self.assertEqual(exit_code, 7)
        infer_main.assert_called_once_with(["analyze", "artifact.jsonl"])

    def test_query_command_dispatches_to_query_cli(self) -> None:
        importlib.import_module("stormlog.query_cli")

        with mock.patch("stormlog.query_cli.main", return_value=8) as query_main:
            exit_code = entrypoint.main(["query", "sessions", "artifacts"])

        self.assertEqual(exit_code, 8)
        query_main.assert_called_once_with(["sessions", "artifacts"])

    def test_native_trace_command_dispatches_to_capture_cli(self) -> None:
        importlib.import_module("stormlog.native_trace_capture")

        with mock.patch(
            "stormlog.native_trace_capture.main", return_value=9
        ) as native_trace_main:
            exit_code = entrypoint.main(["native-trace", "--help"])

        self.assertEqual(exit_code, 9)
        native_trace_main.assert_called_once_with(["--help"])

    def test_help_mentions_no_arg_tui_behavior(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = entrypoint.main(["--help"])

        self.assertEqual(exit_code, 0)
        help_text = output.getvalue()
        self.assertIn("Textual TUI", help_text)
        self.assertIn("query", help_text)
        self.assertIn("infer", help_text)
        self.assertIn("native-trace", help_text)

    def test_unknown_command_exits_with_parser_error(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                entrypoint.main(["unknown"])

        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
