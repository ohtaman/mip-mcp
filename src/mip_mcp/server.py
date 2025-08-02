"""MIP MCP Server implementation."""

import asyncio
import atexit
import contextlib
from typing import Any

from fastmcp import Context, FastMCP

from .handlers.execute_code import (
    execute_mip_code_with_mcp_progress,
    get_mip_examples_handler,
    get_solver_info_handler,
    validate_mip_code_handler,
)
from .models.responses import (
    ExamplesResponse,
    ExecutionResponse,
    SolverInfoResponse,
    ValidationResponse,
)
from .utils.config_manager import ConfigManager
from .utils.executor_registry import ExecutorRegistry
from .utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


class MIPMCPServer:
    """MIP MCP Server for Mathematical Optimization (PuLP)."""

    def __init__(self, config_path: str | None = None):
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

        # Setup cleanup hooks
        self._setup_cleanup_hooks()

        # Register MCP tools
        self._register_tools()

        logger.info(
            f"MIP MCP Server initialized (version: {self.config.server.version})"
        )


    def _setup_cleanup_hooks(self):
        """Setup cleanup hooks that work with FastMCP's natural shutdown."""
        # Register atexit handler as the main cleanup mechanism
        def atexit_cleanup():
            """Clean up executors when the process exits."""
            with contextlib.suppress(Exception):
                # Only run cleanup if there are active executors
                if hasattr(ExecutorRegistry, '_executors') and ExecutorRegistry._executors:
                    logger.info("Cleaning up active executors during shutdown...")
                    try:
                        asyncio.run(ExecutorRegistry.cleanup_all(silent=True))
                        logger.info("Cleanup completed successfully")
                    except RuntimeError:
                        # Event loop might still be running, skip cleanup
                        logger.debug("Skipping cleanup - event loop still running")

        atexit.register(atexit_cleanup)

    def _register_tools(self):
        """Register MCP tools with the FastMCP app."""

        @self.app.tool()
        async def execute_mip_code(
            ctx: Context,
            code: str,
            data: dict[str, Any] | None = None,
            solver: str | None = None,
            solver_params: dict[str, Any] | None = None,
            validate_solution: bool = True,
            validation_tolerance: float = 1e-6,
            include_solver_output: bool = False,
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
                config=self.config_manager.config.model_dump(),
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
        async def validate_mip_code(ctx: Context, code: str) -> ValidationResponse:
            """Validate PuLP code before execution.

            Checks code for syntax errors and potential issues before running.

            Args:
                code: PuLP Python code to validate

            Returns:
                Validation status and any issues found.
            """
            return await validate_mip_code_handler(
                code=code, config=self.config_manager.config.model_dump()
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
        logger.info("Starting MIP MCP Server...")
        # Let FastMCP handle signals and shutdown naturally
        self.app.run(show_banner=show_banner)
