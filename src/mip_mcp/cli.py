#!/usr/bin/env python3
"""CLI entry point for MIP-MCP server."""

import sys
import os
import subprocess
from pathlib import Path

from .server import MIPMCPServer
from .utils.logger import get_logger

logger = get_logger(__name__)


def check_dependencies():
    """Check if required dependencies are available."""
    issues = []
    
    # Check Node.js
    try:
        result = subprocess.run(['node', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            issues.append("Node.js not found or not working")
        else:
            logger.info(f"Node.js version: {result.stdout.strip()}")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        issues.append("Node.js not found in PATH")
    
    # Check pyodide installation
    try:
        result = subprocess.run(['node', '-e', 'console.log(require.resolve("pyodide"))'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            issues.append("Pyodide not found - please run: npm install pyodide")
        else:
            pyodide_path = result.stdout.strip()
            logger.info(f"Pyodide found at: {pyodide_path}")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        issues.append("Failed to check Pyodide installation")
    
    return issues


def install_pyodide():
    """Install pyodide if not present."""
    print("🔧 Installing Pyodide...")
    
    try:
        # Try to install in current directory
        result = subprocess.run(['npm', 'install', 'pyodide'], 
                              capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ Pyodide installed successfully")
            return True
        else:
            print(f"❌ Failed to install Pyodide: {result.stderr}")
            return False
            
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ npm not found or failed to install Pyodide")
        return False


def main():
    """Main CLI entry point."""
    # Check if running in MCP mode (no stdout output for MCP)
    is_mcp_mode = os.environ.get('MCP_MODE') == '1' or '--mcp' in sys.argv
    
    if not is_mcp_mode:
        print("🚀 Starting MIP-MCP Server...")
        print("=" * 50)
    
    # Check dependencies
    if not is_mcp_mode:
        print("🔍 Checking dependencies...")
    
    issues = check_dependencies()
    
    if issues:
        if not is_mcp_mode:
            print("⚠️  Dependency issues found:")
            for issue in issues:
                print(f"   - {issue}")
            
            # Try to auto-install pyodide
            if any("pyodide" in issue.lower() for issue in issues):
                if input("Install Pyodide automatically? (y/n): ").lower() == 'y':
                    if not install_pyodide():
                        print("❌ Failed to install Pyodide. Please install manually:")
                        print("   npm install pyodide")
                        sys.exit(1)
                    
                    # Re-check after installation
                    issues = check_dependencies()
                    if issues:
                        print("❌ Still have dependency issues after installation:")
                        for issue in issues:
                            print(f"   - {issue}")
                        sys.exit(1)
                else:
                    print("❌ Cannot start server without required dependencies")
                    sys.exit(1)
            else:
                print("❌ Cannot start server without required dependencies")
                sys.exit(1)
        else:
            # In MCP mode, just exit silently if dependencies are missing
            logger.error(f"Dependencies missing: {issues}")
            sys.exit(1)
    
    if not is_mcp_mode:
        print("✅ All dependencies OK")
    
    # Start the server
    try:
        if not is_mcp_mode:
            print("\n🌟 Initializing MIP-MCP Server...")
        
        server = MIPMCPServer()
        
        if not is_mcp_mode:
            print("🎯 Server ready! Available tools:")
            print("   - execute_mip_code: Execute MIP code (PuLP, Python-MIP)")
            print("   - validate_mip_code: Validate MIP code syntax & security")
            print("   - get_mip_examples: Get example code snippets")
            print("   - get_solver_info: Get solver information")
            print("   - health_check: Server health status")
            print("\n🔐 Security: Using Pyodide WebAssembly sandbox")
            print("\n⚠️  NOTE: For MCP clients, use 'mip-mcp-server' instead")
            print("🚀 Starting server...")
        
        # Use FastMCP's synchronous run method
        server.run(show_banner=not is_mcp_mode)
        
    except KeyboardInterrupt:
        if not is_mcp_mode:
            print("\n👋 Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        if not is_mcp_mode:
            print(f"❌ Server error: {e}")
        sys.exit(1)


def cli():
    """CLI entry point for setuptools."""
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli()