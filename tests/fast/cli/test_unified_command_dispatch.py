"""Fast tests for the unified CLI parser and exit-code dispatch contract."""

from __future__ import annotations

import pytest

from archimono.cli import __main__

pytestmark = pytest.mark.fast


def test_top_level_parser_requires_a_subcommand(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A bare invocation is an argparse usage error."""
    with pytest.raises(SystemExit) as exc_info:
        __main__._build_parser().parse_args([])

    assert exc_info.value.code == 2
    assert "required" in capsys.readouterr().err


def test_top_level_help_lists_every_registered_command(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        __main__._build_parser().parse_args(["--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    for command_name, _, _ in __main__._COMMANDS:
        assert command_name in output


def test_registered_command_modules_expose_parser_and_run_seams() -> None:
    for _, _, command_module in __main__._COMMANDS:
        assert callable(command_module.configure_parser)
        assert callable(command_module.run)


def test_main_returns_an_integer_command_exit_code_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, solve_module = next(item for item in __main__._COMMANDS if item[0] == "solve")
    monkeypatch.setattr(solve_module, "run", lambda args: 17)

    assert __main__.main(["solve", "square"]) == 17


def test_main_converts_a_none_command_result_to_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, solve_module = next(item for item in __main__._COMMANDS if item[0] == "solve")
    monkeypatch.setattr(solve_module, "run", lambda args: None)

    assert __main__.main(["solve", "square"]) == 0
