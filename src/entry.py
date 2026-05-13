"""Entry point for console_scripts.

Minimal bootstrap that re-exports the Typer app from main.
The import here is from a simple relative path to avoid editable install issues.
"""

from src.main import cli_main

cli = cli_main
