"""Basic tests to verify test infrastructure."""

import sys
from pathlib import Path

import pytest

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


def test_python_version():
    """Test Python version compatibility."""
    assert sys.version_info >= (3, 12), "Python 3.12+ required"


def test_import_basic_modules():
    """Test basic module imports."""
    # Test core imports that should always work

    # Verify basic classes exist
    from mip_mcp.models.config import Config, ServerConfig
    from mip_mcp.models.responses import ExecutionResponse
    from mip_mcp.models.solution import OptimizationSolution

    assert Config is not None
    assert ServerConfig is not None
    assert OptimizationSolution is not None
    assert ExecutionResponse is not None


def test_config_creation():
    """Test basic config creation."""
    from mip_mcp.models.config import Config, ServerConfig

    # Test default config
    config = Config()
    assert config.server.name == "mip-mcp"
    assert config.server.version == "0.1.0"

    # Test custom server config
    server_config = ServerConfig(name="test-server", version="1.0.0")
    assert server_config.name == "test-server"
    assert server_config.version == "1.0.0"


def test_solution_model():
    """Test optimization solution model."""
    from mip_mcp.models.solution import OptimizationSolution

    solution = OptimizationSolution(
        status="optimal",
        objective_value=42.0,
        variables={"x": 1.0, "y": 2.0},
        solve_time=0.1,
        message="Solved in 0.1 seconds",
    )

    assert solution.status == "optimal"
    assert solution.objective_value == 42.0
    assert solution.variables["x"] == 1.0
    assert solution.is_optimal is True
    assert solution.is_feasible is True


def test_response_models():
    """Test response models."""
    from mip_mcp.models.responses import ExecutionResponse, SolverInfo

    # Test execution response with required fields
    response = ExecutionResponse(
        status="success",
        message="Optimization completed",
        stdout="Optimization output",
        stderr="",
        solution={
            "status": "optimal",
            "objective_value": 10.0,
            "variables": {"x": 5.0},
        },
    )

    assert response.status == "success"
    assert response.message == "Optimization completed"
    assert response.solution["status"] == "optimal"

    # Test solver info
    solver_info = SolverInfo(
        name="SCIP",
        available=True,
        version="8.0.3",
        description="SCIP Optimization Suite",
        capabilities=["Linear Programming", "Integer Programming"],
    )

    assert solver_info.name == "SCIP"
    assert solver_info.available is True
    assert "Linear Programming" in solver_info.supported_problem_types


def test_exceptions_import():
    """Test exception classes import."""
    from mip_mcp.exceptions import SecurityError

    # Test that we can create security error
    error = SecurityError("Test security violation")
    assert str(error) == "Test security violation"
    assert isinstance(error, Exception)


@pytest.mark.skipif(
    sys.platform == "win32", reason="Path handling may differ on Windows"
)
def test_path_handling():
    """Test path handling utilities."""
    from pathlib import Path

    # Test basic path operations
    test_path = Path(__file__).parent
    assert test_path.exists()
    assert test_path.is_dir()

    # Test that src directory exists
    src_path = Path(__file__).parent.parent.parent / "src"
    assert src_path.exists()
    assert (src_path / "mip_mcp").exists()


def test_env_variable_handling():
    """Test environment variable handling."""
    import os

    # Test that we can set and read env vars
    test_key = "MIP_MCP_TEST_VAR"
    test_value = "test_value_123"

    os.environ[test_key] = test_value
    assert os.environ.get(test_key) == test_value

    # Cleanup
    del os.environ[test_key]
    assert os.environ.get(test_key) is None
