from __future__ import annotations

import argparse

import pytest

from torchsky import __version__
from torchsky.cli import build_parser


def _command_paths(
    parser: argparse.ArgumentParser,
    argv: tuple[str, ...] = (),
) -> list[tuple[tuple[str, ...], argparse.ArgumentParser]]:
    paths = [(argv, parser)]
    for action in parser._actions:
        if not isinstance(action, argparse._SubParsersAction):
            continue
        for name, child in action.choices.items():
            if not name.startswith("_"):
                paths.extend(_command_paths(child, (*argv, name)))
    return paths


COMMAND_PATHS = _command_paths(build_parser())


@pytest.mark.parametrize(
    ("argv", "command_parser"),
    COMMAND_PATHS,
    ids=lambda value: " ".join(value) if isinstance(value, tuple) else None,
)
def test_every_command_path_has_public_help(
    argv: tuple[str, ...],
    command_parser: argparse.ArgumentParser,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args([*argv, "--help"])
    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out.strip()
    assert help_text
    assert command_parser.prog in help_text


@pytest.mark.parametrize(
    ("argv", "command_parser"),
    COMMAND_PATHS,
    ids=lambda value: " ".join(value) if isinstance(value, tuple) else None,
)
def test_every_public_option_has_help(
    argv: tuple[str, ...],
    command_parser: argparse.ArgumentParser,
) -> None:
    del argv
    option_strings: list[str] = []
    for action in command_parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            continue
        if action.option_strings == ["-h", "--help"]:
            continue
        assert isinstance(action.help, str) and action.help.strip(), (
            command_parser.prog,
            action.option_strings,
            action.dest,
        )
        option_strings.extend(action.option_strings)
    assert len(option_strings) == len(set(option_strings))


@pytest.mark.parametrize(
    ("argv", "command_parser"),
    COMMAND_PATHS,
    ids=lambda value: " ".join(value) if isinstance(value, tuple) else None,
)
def test_every_command_has_inline_examples(
    argv: tuple[str, ...],
    command_parser: argparse.ArgumentParser,
) -> None:
    del argv
    assert isinstance(command_parser.description, str) and command_parser.description.strip()
    assert isinstance(command_parser.epilog, str)
    assert command_parser.epilog.count("torchsky ") >= 2


def test_root_version_is_discoverable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["--version"])
    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"torchsky {__version__}"
