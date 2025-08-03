#!/usr/bin/env python3
"""Pure MCP server entry point without CLI output."""

import os
import signal
import sys

from .server import MIPMCPServer


def mcp_main():
    """MCP-only entry point."""
    # Set MCP mode environment variable
    os.environ["MCP_MODE"] = "1"

    # Install simple signal handler that exits immediately
    def shutdown_handler(signum, frame):
        print(f"\nReceived signal {signum}, shutting down...", file=sys.stderr)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # Create server - cleanup hooks are set up in __init__
    server = MIPMCPServer()

    try:
        # Use server.run() method for proper initialization
        server.run(show_banner=False)
    except KeyboardInterrupt:
        # This shouldn't be reached due to our signal handler
        print("\nShutdown complete.", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    mcp_main()
