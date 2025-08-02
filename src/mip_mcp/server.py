"""MIP MCP Server implementation."""

from typing import Optional, Dict, Any
import asyncio

from fastmcp import FastMCP, Context

from .models.responses import ExecutionResponse, SolverInfoResponse, ValidationResponse, ExamplesResponse

from .handlers.execute_code import (
    execute_mip_code_handler,
    execute_mip_code_with_mcp_progress,
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
            solver: Optional[str] = None,
            solver_params: Optional[Dict[str, Any]] = None,
            validate_solution: bool = True,
            validation_tolerance: float = 1e-6,
            include_solver_output: bool = False
        ) -> ExecutionResponse:
            """Execute PuLP optimization code and solve the problem.
            
            Executes PuLP Python code to create and solve optimization problems:
            - Define variables, constraints, and objectives
            - Solve linear and integer programming problems
            - Get optimal solutions with variable values
            
            Args:
                code: PuLP Python code defining the optimization problem
                data: Optional data dictionary to pass to your code
                solver: Solver to use (default: from config, fallback: "scip")
                solver_params: Optional solver configuration parameters
                validate_solution: Whether to validate the solution (default: True)
                validation_tolerance: Tolerance for constraint validation (default: 1e-6)
                include_solver_output: Include detailed solver output in response (default: False)
                
            Returns:
                Optimization results including solution status, objective value, and variable values.
            """
            return await execute_mip_code_with_mcp_progress(
                code=code,
                mcp_context=ctx,
                data=data,
                solver=solver,
                solver_params=solver_params,
                validate_solution=validate_solution,
                validation_tolerance=validation_tolerance,
                include_solver_output=include_solver_output,
                config=self.config_manager.config.model_dump()
            )
        
        @self.app.tool()
        async def get_solver_info(ctx: Context) -> SolverInfoResponse:
            """Get information about the optimization solver.
            
            Returns:
                Solver name, version, and supported problem types.
            """
            return await get_solver_info_handler(
                config=self.config_manager.config.model_dump()
            )
        
        @self.app.tool()
        async def validate_mip_code(
            ctx: Context,
            code: str
        ) -> ValidationResponse:
            """Validate PuLP code before execution.
            
            Checks code for syntax errors and potential issues before running.
            
            Args:
                code: PuLP Python code to validate
                
            Returns:
                Validation status and any issues found.
            """
            return await validate_mip_code_handler(
                code=code,
                config=self.config_manager.config.model_dump()
            )
        
        @self.app.tool()
        async def get_mip_examples(ctx: Context) -> ExamplesResponse:
            """Get example optimization code snippets.
            
            Provides ready-to-use PuLP examples:
            - Linear programming problems
            - Integer programming problems  
            - Common optimization scenarios
            
            Returns:
                Example code snippets with descriptions.
            """
            return await get_mip_examples_handler()
        
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