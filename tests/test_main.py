"""Tests for the agentic_hpo_in_the_loop package entry point."""

from agentic_hpo_in_the_loop import main


def test_main_prints_greeting(capsys):
    """main() writes the package greeting to stdout."""
    main()

    captured = capsys.readouterr()
    assert captured.out == "Hello from agentic-hpo-in-the-loop!\n"
