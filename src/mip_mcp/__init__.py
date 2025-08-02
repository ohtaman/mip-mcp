"""MIP MCP Server - Mathematical optimization server using Model Context Protocol."""

__version__ = "0.1.0"

from .cli import cli

__all__ = ["cli"]


def main():
    """Legacy main function - use cli() instead."""
    cli()


if __name__ == "__main__":
    cli()
