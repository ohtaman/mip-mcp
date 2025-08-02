"""Tests for model classes."""

from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


class TestConfigModels:
    """Test cases for configuration models."""

    def test_server_config_defaults(self):
        """Test server config with default values."""
        from mip_mcp.models.config import ServerConfig
        
        config = ServerConfig()
        assert config.name == "mip-mcp"
        assert config.version == "0.1.0"

    def test_server_config_custom(self):
        """Test server config with custom values."""
        from mip_mcp.models.config import ServerConfig
        
        config = ServerConfig(name="custom-server", version="2.0.0")
        assert config.name == "custom-server"
        assert config.version == "2.0.0"

    def test_logging_config_defaults(self):
        """Test logging config with default values."""
        from mip_mcp.models.config import LoggingConfig
        
        config = LoggingConfig()
        assert config.level == "INFO"
        assert "%(asctime)s" in config.format

    def test_executor_config_defaults(self):
        """Test executor config with default values."""
        from mip_mcp.models.config import ExecutorConfig
        
        config = ExecutorConfig()
        assert config.enabled is True
        assert config.timeout == 300
        assert config.memory_limit == "1GB"

    def test_solver_config_defaults(self):
        """Test solver config with default values."""
        from mip_mcp.models.config import SolverConfig
        
        config = SolverConfig()
        assert config.default == "scip"
        assert config.timeout == 3600

    def test_validation_config_defaults(self):
        """Test validation config with default values."""
        from mip_mcp.models.config import ValidationConfig
        
        config = ValidationConfig()
        assert config.max_variables == 100000
        assert config.max_constraints == 100000
        assert config.max_code_length == 10000

    def test_main_config_composition(self):
        """Test main config composition."""
        from mip_mcp.models.config import Config
        
        config = Config()
        assert config.server.name == "mip-mcp"
        assert config.logging.level == "INFO"
        assert config.executor.enabled is True
        assert config.solvers.default == "scip"
        assert config.validation.max_variables == 100000

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        from mip_mcp.models.config import Config
        
        data = {
            "server": {"name": "test-server", "version": "1.5.0"},
            "executor": {"timeout": 600, "memory_limit": "2GB"},
            "logging": {"level": "DEBUG"}
        }
        
        config = Config.from_dict(data)
        assert config.server.name == "test-server"
        assert config.server.version == "1.5.0"
        assert config.executor.timeout == 600
        assert config.executor.memory_limit == "2GB"
        assert config.logging.level == "DEBUG"


class TestSolutionModels:
    """Test cases for solution models."""

    def test_solution_variable(self):
        """Test solution variable model."""
        from mip_mcp.models.solution import SolutionVariable
        
        var = SolutionVariable(name="x", value=2.5)
        assert var.name == "x"
        assert var.value == 2.5
        assert var.reduced_cost is None
        
        var_with_cost = SolutionVariable(name="y", value=1.0, reduced_cost=0.5)
        assert var_with_cost.reduced_cost == 0.5

    def test_solution_constraint(self):
        """Test solution constraint model."""
        from mip_mcp.models.solution import SolutionConstraint
        
        constraint = SolutionConstraint(name="c1")
        assert constraint.name == "c1"
        assert constraint.slack is None
        assert constraint.dual_value is None
        
        constraint_full = SolutionConstraint(name="c2", slack=1.5, dual_value=0.8)
        assert constraint_full.slack == 1.5
        assert constraint_full.dual_value == 0.8

    def test_validation_violation(self):
        """Test validation violation model."""
        from mip_mcp.models.solution import ValidationViolation
        
        violation = ValidationViolation(
            type="constraint",
            description="Constraint c1 violated",
            severity="error"
        )
        assert violation.type == "constraint"
        assert violation.description == "Constraint c1 violated"
        assert violation.severity == "error"
        assert violation.details == {}

    def test_solution_validation(self):
        """Test solution validation model."""
        from mip_mcp.models.solution import SolutionValidation
        
        validation = SolutionValidation(
            is_valid=True,
            tolerance_used=1e-6
        )
        assert validation.is_valid is True
        assert validation.tolerance_used == 1e-6
        assert validation.constraint_violations == []
        assert validation.error is None

    def test_optimization_solution_basic(self):
        """Test basic optimization solution."""
        from mip_mcp.models.solution import OptimizationSolution
        
        solution = OptimizationSolution(
            status="optimal",
            objective_value=100.0,
            variables={"x": 5.0, "y": 10.0}
        )
        
        assert solution.status == "optimal"
        assert solution.objective_value == 100.0
        assert solution.variables["x"] == 5.0
        assert solution.is_optimal is True
        assert solution.is_feasible is True

    def test_optimization_solution_properties(self):
        """Test optimization solution properties."""
        from mip_mcp.models.solution import OptimizationSolution
        
        # Test optimal solution
        optimal_solution = OptimizationSolution(status="optimal")
        assert optimal_solution.is_optimal is True
        assert optimal_solution.is_feasible is True
        
        # Test infeasible solution
        infeasible_solution = OptimizationSolution(status="infeasible")
        assert infeasible_solution.is_optimal is False
        assert infeasible_solution.is_feasible is False
        
        # Test unbounded solution
        unbounded_solution = OptimizationSolution(status="unbounded")
        assert unbounded_solution.is_optimal is False
        assert unbounded_solution.is_feasible is False


class TestResponseModels:
    """Test cases for response models."""

    def test_execution_response_minimal(self):
        """Test minimal execution response."""
        from mip_mcp.models.responses import ExecutionResponse
        
        response = ExecutionResponse(
            status="success",
            message="Code executed successfully",
            stdout="Output",
            stderr=""
        )
        
        assert response.status == "success"
        assert response.message == "Code executed successfully"
        assert response.stdout == "Output"
        assert response.stderr == ""
        assert response.solution is None

    def test_execution_response_with_solution(self):
        """Test execution response with solution."""
        from mip_mcp.models.responses import ExecutionResponse
        
        solution_data = {
            "status": "optimal",
            "objective_value": 42.0,
            "variables": {"x": 1.0, "y": 2.0}
        }
        
        response = ExecutionResponse(
            status="success",
            message="Optimization completed",
            stdout="Solver output",
            stderr="",
            solution=solution_data,
            solver_output="Detailed solver information"
        )
        
        assert response.status == "success"
        assert response.solution["status"] == "optimal"
        assert response.solver_output == "Detailed solver information"

    def test_solver_info_basic(self):
        """Test basic solver info."""
        from mip_mcp.models.responses import SolverInfo
        
        info = SolverInfo(
            name="TestSolver",
            available=True,
            version="1.0.0"
        )
        
        assert info.name == "TestSolver"
        assert info.available is True
        assert info.version == "1.0.0"
        assert info.supported_problem_types == []

    def test_solver_info_with_capabilities(self):
        """Test solver info with capabilities."""
        from mip_mcp.models.responses import SolverInfo
        
        capabilities = ["Linear Programming", "Integer Programming"]
        
        info = SolverInfo(
            name="SCIP",
            available=True,
            version="8.0.3",
            description="SCIP Optimization Suite",
            capabilities=capabilities
        )
        
        assert info.capabilities == capabilities
        assert info.supported_problem_types == capabilities
        assert "Linear Programming" in info.supported_problem_types

    def test_solver_info_version_conversion(self):
        """Test solver info version conversion."""
        from mip_mcp.models.responses import SolverInfo
        
        # Test numeric version conversion
        info_numeric = SolverInfo(
            name="TestSolver",
            available=True,
            version=8.03  # Float version
        )
        
        assert info_numeric.version == "8.03"
        
        # Test tuple version (should work with validator)
        info_tuple = SolverInfo(
            name="TestSolver",
            available=True,
            version=(8, 0, 3)  # Tuple version
        )
        
        assert info_tuple.version == "(8, 0, 3)"

    def test_solver_info_unavailable(self):
        """Test unavailable solver info."""
        from mip_mcp.models.responses import SolverInfo
        
        info = SolverInfo(
            name="UnavailableSolver",
            available=False,
            error="Solver not found"
        )
        
        assert info.available is False
        assert info.error == "Solver not found"
        assert info.version is None