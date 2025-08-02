#!/usr/bin/env python3
"""Pure MCP server entry point without CLI output."""

import asyncio
import sys
import os

from .server import MIPMCPServer


def mcp_main():
    """MCP-only entry point."""
    try:
        # Set MCP mode environment variable
        os.environ['MCP_MODE'] = '1'
        
        server = MIPMCPServer()
        # FastMCP's run method is synchronous and handles asyncio internally
        server.app.run(show_banner=False)
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    mcp_main()