"""MIP MCP Server - PuLP-based optimization server using Model Context Protocol."""

__version__ = "0.1.0"


def main():
    from .server import MIPMCPServer
    
    server = MIPMCPServer()
    server.app.run()


if __name__ == "__main__":
    cli()
