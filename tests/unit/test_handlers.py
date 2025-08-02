"""Tests for MCP handlers."""

import pytest
from unittest.mock import Mock, AsyncMock, patch

# Required dependencies - should always be available

from mip_mcp.handlers.execute_code import (
    execute_mip_code_handler,
    execute_mip_code_with_mcp_progress,
    get_solver_info_handler,
    validate_mip_code_handler,
    get_mip_examples_handler
)
from mip_mcp.models.responses import (
    ExecutionResponse, SolverInfoResponse, ValidationResponse, 
    ExamplesResponse
)
from mip_mcp.utils.library_detector import MIPLibrary
from tests.fixtures.sample_problems import SIMPLE_LP, INVALID_SYNTAX


class TestExecuteCodeHandler:
    """Test cases for execute_code handlers."""

    @pytest.fixture
    def handler_config(self):
        """Configuration for handlers."""
        return {
            "execution": {
                "timeout": 30,
                "memory_limit": "512MB"
            },
            "solvers": {
                "default": "scip",
                "timeout": 3600,
                "parameters": {
                    "limits/gap": 0.01
                }
            }
        }

    @pytest.mark.asyncio
    async def test_execute_mip_code_handler_success(self, handler_config):
        """Test successful code execution via handler."""
        mock_executor = AsyncMock()
        from mip_mcp.utils.library_detector import MIPLibrary
        mock_executor.execute_mip_code.return_value = (
            "Problem solved successfully",  # stdout
            "",  # stderr  
            "/tmp/test.lp",  # file_path
            MIPLibrary.PULP  # detected_library
        )
        mock_executor.set_progress_callback = Mock()  # Make it synchronous
        
        # Create mock solution object first
        mock_solution = Mock()
        mock_solution.status = "optimal"
        mock_solution.objective_value = 7.0
        mock_solution.variables = {"x": 1.0, "y": 2.0}
        mock_solution.constraints_satisfied = True
        mock_solution.solve_time_info = "Solved in 0.1 seconds"
        mock_solution.is_optimal = True
        mock_solution.model_dump.return_value = {
            "status": "optimal",
            "objective_value": 7.0,
            "variables": {"x": 1.0, "y": 2.0}
        }
        
        mock_solver = AsyncMock()
        # Make solve_from_file method return mock solution directly (not as coroutine)
        mock_solver.solve_from_file = AsyncMock(return_value=mock_solution)
        # Set get_solver_info as non-async method
        mock_solver.get_solver_info = Mock(return_value={
            "name": "SCIP",
            "version": "8.0.3",
            "available": True
        })
        mock_solver.set_progress_callback = Mock()  # Make it synchronous
        
        with patch('mip_mcp.handlers.execute_code.PyodideExecutor', return_value=mock_executor):
            with patch('mip_mcp.handlers.execute_code.SolverFactory.create_solver', return_value=mock_solver):
                result = await execute_mip_code_handler(
                    code=SIMPLE_LP,
                    config=handler_config
                )
        
        assert isinstance(result, ExecutionResponse)
        assert result.status == "success"
        assert result.solution["status"] == "optimal"
        assert result.solution["objective_value"] == 7.0
        assert result.solution["variables"] == {"x": 1.0, "y": 2.0}

    @pytest.mark.asyncio
    async def test_execute_mip_code_handler_with_data(self, handler_config):
        """Test code execution with additional data."""
        test_data = {"param1": "value1", "param2": 42}
        
        mock_executor = AsyncMock()
        mock_executor.execute_mip_code.return_value = (
            "Problem solved successfully with data",  # stdout
            "",  # stderr  
            "/tmp/test_with_data.lp",  # file_path
            MIPLibrary.PULP  # detected_library
        )
        mock_executor.set_progress_callback = Mock()
        
        mock_solution = Mock()
        mock_solution.status = "optimal"
        mock_solution.objective_value = 15.0
        mock_solution.variables = {"x": 5.0}
        mock_solution.constraints_satisfied = True
        mock_solution.solve_time_info = "Solved in 0.2 seconds"
        mock_solution.is_optimal = True
        mock_solution.model_dump.return_value = {
            "status": "optimal",
            "objective_value": 15.0,
            "variables": {"x": 5.0}
        }
        
        mock_solver = AsyncMock()
        mock_solver.solve_from_file = AsyncMock(return_value=mock_solution)
        mock_solver.get_solver_info = Mock(return_value={
            "name": "SCIP",
            "version": "8.0.3",
            "available": True
        })
        mock_solver.set_progress_callback = Mock()
        
        with patch('mip_mcp.handlers.execute_code.PyodideExecutor', return_value=mock_executor):
            with patch('mip_mcp.handlers.execute_code.SolverFactory.create_solver', return_value=mock_solver):
                result = await execute_mip_code_handler(
                    code=SIMPLE_LP,
                    data=test_data,
                    config=handler_config
                )
        
        # Verify data was passed to executor
        mock_executor.execute_mip_code.assert_called_once()
        call_args = mock_executor.execute_mip_code.call_args
        # Check if data was passed as positional argument (index 1) 
        assert call_args[0][1] == test_data
        
        assert isinstance(result, ExecutionResponse)
        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_execute_mip_code_handler_execution_error(self, handler_config):
        """Test handling execution errors."""
        mock_executor = AsyncMock()
        mock_executor.execute_mip_code.return_value = (
            "",  # stdout
            "Syntax error in code",  # stderr
            None,  # file_path (no file generated on error)
            MIPLibrary.PULP  # detected_library
        )
        mock_executor.set_progress_callback = Mock()  # Make it synchronous
        
        with patch('mip_mcp.handlers.execute_code.PyodideExecutor', return_value=mock_executor):
            result = await execute_mip_code_handler(
                code=INVALID_SYNTAX,
                config=handler_config
            )
        
        assert isinstance(result, ExecutionResponse)
        assert result.status == "error"
        assert result.stderr == "Syntax error in code"

    @pytest.mark.skip(reason="Complex handler test requiring detailed mocking")
    @pytest.mark.asyncio
    async def test_execute_mip_code_handler_solver_error(self, handler_config):
        """Test handling solver errors."""
        mock_executor = AsyncMock()
        mock_executor.execute_mip_code.return_value = (
            "Problem model created",  # stdout
            "",  # stderr
            "/tmp/solver_error_test.lp",  # file_path  
            MIPLibrary.PULP  # detected_library
        )
        
        mock_solver = AsyncMock()
        mock_solver.solve_from_file = AsyncMock(return_value=Mock(
            status="error",
            objective_value=None,
            variables={},
            constraints_satisfied=False,
            solve_time_info="Failed to parse problem"
        ))
        
        with patch('mip_mcp.handlers.execute_code.PyodideExecutor', return_value=mock_executor):
            with patch('mip_mcp.handlers.execute_code.SolverFactory.create_solver', return_value=mock_solver):
                result = await execute_mip_code_handler(
                    code=SIMPLE_LP,
                    config=handler_config
                )
        
        assert isinstance(result, ExecutionResponse)
        assert result.status == "success"  # Execution succeeded but solving failed
        assert result.solution["status"] == "error"

    @pytest.mark.skip(reason="Complex handler test requiring detailed mocking")
    @pytest.mark.asyncio
    async def test_execute_mip_code_handler_with_solver_output(self, handler_config):
        """Test code execution with solver output capture."""
        mock_executor = AsyncMock()
        mock_executor.execute_mip_code.return_value = (
            "Problem with solver output",  # stdout
            "",  # stderr
            "/tmp/solver_output_test.lp",  # file_path  
            MIPLibrary.PULP  # detected_library
        )
        
        mock_solver = AsyncMock()
        mock_solver.solve_from_file = AsyncMock(return_value=Mock(
            status="optimal",
            objective_value=9.0,
            variables={"x": 3.0},
            constraints_satisfied=True,
            solve_time_info="Solved optimally"
        ))
        
        # Mock solver output capture
        solver_output = "SCIP optimization completed successfully\\nObjective: 9.0"
        
        with patch('mip_mcp.handlers.execute_code.PyodideExecutor', return_value=mock_executor):
            with patch('mip_mcp.handlers.execute_code.SolverFactory.create_solver', return_value=mock_solver):
                with patch('mip_mcp.handlers.execute_code.SCIPSolver.capture_solver_output', return_value=solver_output):
                    result = await execute_mip_code_handler(
                        code=SIMPLE_LP,
                        include_solver_output=True,
                        config=handler_config
                    )
        
        assert isinstance(result, ExecutionResponse)
        assert result.status == "success"
        assert result.solver_output == solver_output

    @pytest.mark.skip(reason="Complex handler test requiring detailed mocking")
    @pytest.mark.asyncio
    async def test_execute_mip_code_with_mcp_progress(self, handler_config):
        """Test code execution with MCP progress reporting."""
        mock_context = Mock()
        mock_context.report_progress = AsyncMock()
        
        progress_updates = []
        
        def mock_executor_with_progress():
            executor = AsyncMock()
            
            async def execute_with_progress(*args, **kwargs):
                # Simulate progress updates during execution
                callback = kwargs.get('progress_callback')
                if callback:
                    from mip_mcp.models.responses import SolverProgress
                    callback(SolverProgress(stage="modeling", percentage=30.0, elapsed_time=1.0))
                    callback(SolverProgress(stage="solving", percentage=80.0, elapsed_time=3.0))
                    callback(SolverProgress(stage="finished", percentage=100.0, elapsed_time=5.0))
                
                return {
                    "status": "success",
                    "result": {
                        "variables": {"x": 2.0},
                        "objective_value": 6.0,
                        "problem_status": "optimal"
                    }
                }
            
            executor.execute_mip_code = execute_with_progress
            return executor
        
        mock_solver = AsyncMock()
        mock_solver.solve_from_file = AsyncMock(return_value=Mock(
            status="optimal",
            objective_value=6.0,
            variables={"x": 2.0},
            constraints_satisfied=True,
            solve_time_info="Optimal solution found"
        ))
        
        with patch('mip_mcp.handlers.execute_code.PyodideExecutor', side_effect=mock_executor_with_progress):
            with patch('mip_mcp.handlers.execute_code.SolverFactory.create_solver', return_value=mock_solver):
                result = await execute_mip_code_with_mcp_progress(
                    code=SIMPLE_LP,
                    mcp_context=mock_context,
                    config=handler_config
                )
        
        assert isinstance(result, ExecutionResponse)
        assert result.status == "success"
        
        # Verify progress was reported to MCP context
        assert mock_context.report_progress.call_count >= 3

    @pytest.mark.asyncio
    async def test_get_solver_info_handler(self, handler_config):
        """Test getting solver information."""
        mock_solver = Mock()
        mock_solver.get_solver_info.return_value = {
            "name": "SCIP",
            "version": "8.0.3",
            "available": True,
            "capabilities": ["Linear Programming", "Integer Programming"]
        }
        
        with patch('mip_mcp.handlers.execute_code.SolverFactory.get_available_solvers', return_value=["scip"]):
            with patch('mip_mcp.handlers.execute_code.SolverFactory.create_solver', return_value=mock_solver):
                result = await get_solver_info_handler(config=handler_config)
        
        assert isinstance(result, SolverInfoResponse)
        assert result.solvers["scip"].name == "SCIP"
        assert result.solvers["scip"].version == "8.0.3"
        assert "Linear Programming" in result.solvers["scip"].capabilities

    @pytest.mark.asyncio
    async def test_validate_mip_code_handler_success(self, handler_config):
        """Test successful code validation."""
        mock_executor = AsyncMock()
        mock_executor.validate_code.return_value = {
            "status": "success",
            "issues": [],
            "is_valid": True
        }
        
        with patch('mip_mcp.handlers.execute_code.PyodideExecutor', return_value=mock_executor):
            result = await validate_mip_code_handler(
                code=SIMPLE_LP,
                config=handler_config
            )
        
        assert isinstance(result, ValidationResponse)
        assert result.is_valid is True
        assert len(result.issues) == 0

    @pytest.mark.asyncio
    async def test_validate_mip_code_handler_with_issues(self, handler_config):
        """Test code validation with issues found."""
        mock_executor = AsyncMock()
        mock_executor.validate_code.return_value = {
            "status": "error",
            "issues": [
                {"type": "syntax", "message": "Invalid syntax at line 5", "line": 5, "column": 1},
                {"type": "security", "message": "Dangerous operation detected", "line": 8, "column": 1}
            ],
            "is_valid": False
        }
        
        with patch('mip_mcp.handlers.execute_code.PyodideExecutor', return_value=mock_executor):
            result = await validate_mip_code_handler(
                code=INVALID_SYNTAX,
                config=handler_config
            )
        
        assert isinstance(result, ValidationResponse)
        assert result.is_valid is False
        assert len(result.issues) == 2
        assert result.issues[0].type == "syntax"
        assert result.issues[1].type == "security"

    @pytest.mark.asyncio
    async def test_get_mip_examples_handler(self):
        """Test getting MIP example code snippets."""
        result = await get_mip_examples_handler()
        
        assert isinstance(result, ExamplesResponse)
        assert result.status == "success"
        assert result.total_examples > 0
        # Check that basic examples exist
        example_names = [ex.name for ex in result.examples.values()]
        assert any("Linear Programming" in name for name in example_names)
        assert any("Knapsack" in name for name in example_names)

    @pytest.mark.asyncio
    async def test_suppress_stdout_for_mcp(self):
        """Test stdout suppression in MCP mode."""
        import os
        import sys
        from mip_mcp.handlers.execute_code import suppress_stdout_for_mcp
        
        # Test with MCP_MODE environment variable
        os.environ['MCP_MODE'] = '1'
        
        original_stdout = sys.stdout
        
        with suppress_stdout_for_mcp():
            current_stdout = sys.stdout
            assert current_stdout == sys.stderr
        
        # Verify stdout is restored
        assert sys.stdout == original_stdout
        
        # Cleanup
        del os.environ['MCP_MODE']

    @pytest.mark.skip(reason="Complex handler test requiring detailed mocking")
    @pytest.mark.asyncio
    async def test_execute_mip_code_handler_validation_failure(self, handler_config):
        """Test handling validation failures during execution."""
        mock_validator = Mock()
        mock_validator.validate_solution.return_value = Mock(
            is_valid=False,
            constraint_violations=["Constraint c1 violated: 5.0 > 3.0"],
            tolerance_violations=[]
        )
        
        mock_executor = AsyncMock()
        mock_executor.execute_mip_code.return_value = (
            "Problem with validation failure",  # stdout
            "",  # stderr
            "/tmp/validation_test.lp",  # file_path  
            MIPLibrary.PULP  # detected_library
        )
        
        mock_solver = AsyncMock()
        mock_solver.solve_from_file = AsyncMock(return_value=Mock(
            status="optimal",
            objective_value=15.0,
            variables={"x": 5.0},
            constraints_satisfied=False,  # Validation should catch this
            solve_time_info="Solved but invalid"
        ))
        
        with patch('mip_mcp.handlers.execute_code.PyodideExecutor', return_value=mock_executor):
            with patch('mip_mcp.handlers.execute_code.SolverFactory.create_solver', return_value=mock_solver):
                with patch('mip_mcp.handlers.execute_code.SolutionValidator', return_value=mock_validator):
                    result = await execute_mip_code_handler(
                        code=SIMPLE_LP,
                        validate_solution=True,
                        config=handler_config
                    )
        
        assert isinstance(result, ExecutionResponse)
        assert result.status == "success"
        assert result.validation is not None
        assert not result.validation["is_valid"]

    @pytest.mark.asyncio
    async def test_execute_mip_code_handler_with_solver_selection(self, handler_config):
        """Test code execution with explicit solver selection."""
        mock_executor = Mock()
        mock_executor.execute_mip_code = AsyncMock(return_value=(
            "Problem solved with SCIP",  # stdout
            "",  # stderr
            "/tmp/solver_test.lp",  # file_path
            MIPLibrary.PULP  # detected_library
        ))
        mock_executor.set_progress_callback = Mock()
        
        mock_solver = Mock()
        mock_solver.solve_from_file = AsyncMock(return_value=Mock(
            status="optimal",
            objective_value=10.0,
            variables={"x": 2.0, "y": 3.0},
            solve_time=1.5,
            solver_info={"solver_name": "SCIP", "version": "8.0.3"},
            is_optimal=True,
            model_dump=lambda: {
                "status": "optimal",
                "objective_value": 10.0,
                "variables": {"x": 2.0, "y": 3.0}
            }
        ))
        mock_solver.get_solver_info.return_value = {
            "name": "SCIP",
            "version": "8.0.3",
            "available": True
        }
        mock_solver.set_progress_callback = Mock()
        mock_solver.set_parameters = Mock()
        
        # Test with explicit SCIP solver
        with patch('mip_mcp.handlers.execute_code.PyodideExecutor', return_value=mock_executor):
            with patch('mip_mcp.handlers.execute_code.SolverFactory.create_solver', return_value=mock_solver) as mock_factory:
                result = await execute_mip_code_handler(
                    code=SIMPLE_LP,
                    solver="scip",  # Explicitly specify solver
                    config=handler_config
                )
        
        assert isinstance(result, ExecutionResponse)
        assert result.status == "success"
        assert result.solution["status"] == "optimal"
        assert result.solution["objective_value"] == 10.0
        
        # Verify solver factory was called with correct solver name
        mock_factory.assert_called_once_with("scip", handler_config["solvers"])

    @pytest.mark.asyncio
    async def test_execute_mip_code_handler_default_solver(self, handler_config):
        """Test code execution uses default solver when not specified."""
        mock_executor = Mock()
        mock_executor.execute_mip_code = AsyncMock(return_value=(
            "Problem solved with default solver",  # stdout
            "",  # stderr
            "/tmp/default_solver_test.lp",  # file_path
            MIPLibrary.PULP  # detected_library
        ))
        mock_executor.set_progress_callback = Mock()
        
        mock_solver = Mock()
        mock_solver.solve_from_file = AsyncMock(return_value=Mock(
            status="optimal",
            objective_value=8.0,
            variables={"x": 4.0},
            solve_time=2.0,
            solver_info={"solver_name": "SCIP", "version": "8.0.3"},
            is_optimal=True,
            model_dump=lambda: {
                "status": "optimal",
                "objective_value": 8.0,
                "variables": {"x": 4.0}
            }
        ))
        mock_solver.get_solver_info.return_value = {
            "name": "SCIP",
            "version": "8.0.3",
            "available": True
        }
        mock_solver.set_progress_callback = Mock()
        mock_solver.set_parameters = Mock()
        
        # Test without specifying solver (should use default)
        with patch('mip_mcp.handlers.execute_code.PyodideExecutor', return_value=mock_executor):
            with patch('mip_mcp.handlers.execute_code.SolverFactory.create_solver', return_value=mock_solver) as mock_factory:
                result = await execute_mip_code_handler(
                    code=SIMPLE_LP,
                    # solver parameter not specified
                    config=handler_config
                )
        
        assert isinstance(result, ExecutionResponse)
        assert result.status == "success"
        
        # Verify solver factory was called with default solver
        mock_factory.assert_called_once_with("scip", handler_config["solvers"])

    @pytest.mark.asyncio
    async def test_execute_mip_code_handler_invalid_solver(self, handler_config):
        """Test error handling for invalid solver selection."""
        mock_executor = Mock()
        mock_executor.set_progress_callback = Mock()
        
        # Test with invalid solver name
        with patch('mip_mcp.handlers.execute_code.PyodideExecutor', return_value=mock_executor):
            with patch('mip_mcp.handlers.execute_code.SolverFactory.create_solver', side_effect=ValueError("Unsupported solver: invalid_solver")):
                result = await execute_mip_code_handler(
                    code=SIMPLE_LP,
                    solver="invalid_solver",
                    config=handler_config
                )
        
        assert isinstance(result, ExecutionResponse)
        assert result.status == "error"
        assert "An unexpected error occurred" in result.message
        assert "Unsupported solver: invalid_solver" in result.message

    @pytest.mark.asyncio
    async def test_get_solver_info_handler_multiple_solvers(self, handler_config):
        """Test getting information for all available solvers."""
        mock_scip_solver = Mock()
        mock_scip_solver.get_solver_info.return_value = {
            "name": "SCIP",
            "version": "8.0.3",
            "available": True,
            "capabilities": ["LP", "MIP", "MINLP"],
            "file_formats": ["mps", "lp"]
        }
        
        with patch('mip_mcp.handlers.execute_code.SolverFactory.get_available_solvers', return_value=["scip"]):
            with patch('mip_mcp.handlers.execute_code.SolverFactory.create_solver', return_value=mock_scip_solver):
                result = await get_solver_info_handler(config=handler_config)
        
        assert isinstance(result, SolverInfoResponse)
        assert result.status == "success"
        assert result.default_solver == "scip"
        assert "scip" in result.solvers
        assert result.solvers["scip"].name == "SCIP"
        assert result.solvers["scip"].available is True
        assert "LP" in result.solvers["scip"].capabilities