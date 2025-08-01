"""MIP MCP Server implementation."""

from typing import Optional, Dict, Any
import asyncio

from fastmcp import FastMCP, Context

from .handlers.execute_code import (
    execute_pulp_code_handler,
    get_solver_info_handler,
    validate_pulp_code_handler,
    get_pulp_examples_handler
)
from .utils.config_manager import ConfigManager
from .utils.logger import setup_logging, get_logger

logger = get_logger(__name__)


class MIPMCPServer:
    """MIP MCP Server for PuLP optimization."""
    
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
        async def execute_pulp_code(
            ctx: Context,
            code: str,
            data: Optional[Dict[str, Any]] = None,
            output_format: str = "mps",
            solver_params: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            """Execute PuLP code and solve optimization problem.
            
            Executes PuLP Python code in a secure sandbox environment. The code can:
            - Create PuLP problems with variables, constraints, and objectives
            - Use automatic problem detection (recommended)
            - Manually set content via __mps_content__ or __lp_content__ variables
            
            Security: File writing operations (writeLP, writeMPS) are prohibited.
            The system automatically generates MPS/LP files from PuLP problem objects.
            
            Args:
                code: PuLP Python code to execute (no file writing allowed)
                data: Optional data dictionary to pass to the code
                output_format: Output format ('mps' or 'lp') 
                solver_params: Optional solver parameters for SCIP
                
            Returns:
                Execution results and optimization solution from SCIP solver
            """
            return await execute_pulp_code_handler(
                code=code,
                data=data,
                output_format=output_format,
                solver_params=solver_params,
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
        async def validate_pulp_code(
            ctx: Context,
            code: str
        ) -> Dict[str, Any]:
            """Validate PuLP code for security and syntax.
            
            Performs security validation including:
            - AST analysis for dangerous operations
            - File writing operation detection (writeLP, writeMPS, etc.)
            - Import restriction validation
            - Syntax error checking
            
            Args:
                code: PuLP Python code to validate (checked for security violations)
                
            Returns:
                Validation results with status and any security issues found
            """
            return await validate_pulp_code_handler(
                code=code,
                config=self.config_manager.config.model_dump()
            )
        
        @self.app.tool()
        async def get_pulp_examples(ctx: Context) -> Dict[str, Any]:
            """Get example PuLP code snippets.
            
            Provides various PuLP code examples demonstrating:
            - Linear programming problems
            - Integer programming problems  
            - Knapsack problems
            - Automatic problem detection (no file writing required)
            - Manual content setting via variables (advanced usage)
            
            Returns:
                Dictionary with categorized example code snippets and descriptions
            """
            return await get_pulp_examples_handler()
        
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
    
    async def run(self):
        """Run the MCP server."""
        try:
            logger.info("Starting MIP MCP Server...")
            await self.app.run()
        except KeyboardInterrupt:
            logger.info("Server stopped by user")
        except Exception as e:
            logger.error(f"Server error: {e}")
            raise
        finally:
            logger.info("MIP MCP Server shutdown complete")