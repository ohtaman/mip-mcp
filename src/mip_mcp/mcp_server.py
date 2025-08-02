#!/usr/bin/env python3
"""Pure MCP server entry point without CLI output."""

import os

from .server import MIPMCPServer


def mcp_main():
    """MCP-only entry point."""
    # Set MCP mode environment variable
    os.environ["MCP_MODE"] = "1"

    # Create server - signal handlers are set up in __init__
    server = MIPMCPServer()
    
    # FastMCP's run method is synchronous and handles asyncio internally
    # Our signal handlers will handle Ctrl+C properly
    server.app.run(show_banner=False)


if __name__ == "__main__":
    mcp_main()
