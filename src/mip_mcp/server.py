"""MIP MCP Server implementation."""

from typing import Optional, Dict, Any
import asyncio

from fastmcp import FastMCP, Context

from .handlers.execute_code import (
    execute_mip_code_handler,
    get_solver_info_handler,
    validate_mip_code_handler,
    get_mip_examples_handler
)
from .utils.config_manager import ConfigManager
from .utils.logger import setup_logging, get_logger

logger = get_logger(__name__)


class MIPMCPServer:
    """MIP MCP Server for Mathematical Optimization (PuLP, Python-MIP)."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the MCP server.
        
        Args:
            config_path: Path to configuration directory or file
        """
        # Load configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        
        # Setup logging
        setup_logging(self.config_manager)
        
        # Initialize FastMCP app
        self.app = FastMCP("mip-mcp")
        
        # Register MCP tools
        self._register_tools()
        
        logger.info(f"MIP MCP Server initialized (version: {self.config.server.version})")
    
    def _register_tools(self):
        """Register MCP tools with the FastMCP app."""
        
        @self.app.tool()
        async def execute_mip_code(
            ctx: Context,
            code: str,
            data: Optional[Dict[str, Any]] = None,
            output_format: str = "mps",
            solver_params: Optional[Dict[str, Any]] = None,
            validate_solution: bool = True,
            validation_tolerance: float = 1e-6,
            library: str = "auto",
            use_pyodide: bool = True
        ) -> Dict[str, Any]:
            """Execute MIP code and solve optimization problem.
            
            Executes MIP Python code (PuLP or Python-MIP) in a secure Pyodide environment. The code can:
            - Create optimization problems with variables, constraints, and objectives
            - Use automatic problem detection (recommended)
            - Manually set content via __mps_content__ or __lp_content__ variables
            
            Security: Executes in WebAssembly sandbox for complete isolation.
            The system automatically generates MPS/LP files from problem objects.
            
            Args:
                code: MIP Python code to execute (PuLP or Python-MIP)
                data: Optional data dictionary to pass to the code
                output_format: Output format ('mps' or 'lp') 
                solver_params: Optional solver parameters for SCIP
                validate_solution: Whether to validate solution against constraints (default: True)
                validation_tolerance: Numerical tolerance for constraint validation (default: 1e-6)
                library: MIP library to use ('auto', 'pulp', 'python-mip')
                use_pyodide: Use secure Pyodide executor (default: True)
                
            Returns:
                Execution results and optimization solution from SCIP solver with validation
            """
            return await execute_mip_code_handler(
                code=code,
                data=data,
                output_format=output_format,
                solver_params=solver_params,
                validate_solution=validate_solution,
                validation_tolerance=validation_tolerance,
                library=library,
                use_pyodide=use_pyodide,
                config=self.config_manager.config.model_dump()
            )
        
        @self.app.tool()
        async def get_solver_info(ctx: Context) -> Dict[str, Any]:
            """Get information about available solvers.
            
            Returns:
                Solver information and capabilities
            """
            return await get_solver_info_handler(
                config=self.config_manager.config.model_dump()
            )
        
        @self.app.tool()
        async def validate_mip_code(
            ctx: Context,
            code: str,
            library: str = "auto",
            use_pyodide: bool = True
        ) -> Dict[str, Any]:
            """Validate MIP code for security and syntax.
            
            Performs security validation including:
            - Library detection (PuLP, Python-MIP)
            - Syntax error checking
            - Pyodide compatibility validation
            - Security analysis
            
            Args:
                code: MIP Python code to validate (PuLP or Python-MIP)
                library: MIP library to validate for ('auto', 'pulp', 'python-mip')
                use_pyodide: Use Pyodide validator (default: True)
                
            Returns:
                Validation results with status and any security issues found
            """
            return await validate_mip_code_handler(
                code=code,
                library=library,
                use_pyodide=use_pyodide,
                config=self.config_manager.config.model_dump()
            )
        
        @self.app.tool()
        async def get_mip_examples(ctx: Context) -> Dict[str, Any]:
            """Get example MIP code snippets.
            
            Provides various MIP code examples demonstrating:
            - Linear programming problems (PuLP, Python-MIP)
            - Integer programming problems  
            - Knapsack problems
            - Automatic problem detection (no file writing required)
            - Manual content setting via variables (advanced usage)
            
            Returns:
                Dictionary with categorized example code snippets and descriptions
            """
            return await get_mip_examples_handler()
        
        @self.app.tool()
        async def health_check(ctx: Context) -> Dict[str, Any]:
            """Health check endpoint.
            
            Returns:
                Server health status
            """
            try:
                # Test solver availability
                solver_info = await get_solver_info_handler(
                    config=self.config_manager.config.model_dump()
                )
                
                return {
                    "status": "healthy",
                    "version": self.config.server.version,
                    "server_name": self.config.server.name,
                    "solver_available": solver_info.get("status") == "success",
                    "timestamp": None  # FastMCP might add this
                }
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                return {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        logger.info("MCP tools registered successfully")
    
    def run(self, show_banner: bool = True):
        """Run the MCP server."""
        try:
            logger.info("Starting MIP MCP Server...")
            self.app.run(show_banner=show_banner)
        except KeyboardInterrupt:
            logger.info("Server stopped by user")
        except Exception as e:
            logger.error(f"Server error: {e}")
            raise
        finally:
            logger.info("MIP MCP Server shutdown complete")