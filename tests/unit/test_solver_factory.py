"""Tests for SolverFactory class."""

import pytest
from unittest.mock import Mock, patch

from src.mip_mcp.solvers.factory import SolverFactory
from src.mip_mcp.solvers.base import BaseSolver
from src.mip_mcp.solvers.scip_solver import SCIPSolver


class TestSolverFactory:
    """Test cases for SolverFactory."""

    def test_get_available_solvers(self):
        """Test getting available solvers."""
        solvers = SolverFactory.get_available_solvers()
        assert isinstance(solvers, list)
        assert "scip" in solvers

    def test_is_solver_available(self):
        """Test checking if solver is available."""
        assert SolverFactory.is_solver_available("scip") is True
        assert SolverFactory.is_solver_available("SCIP") is True  # Case insensitive
        assert SolverFactory.is_solver_available("invalid") is False

    @patch('src.mip_mcp.solvers.scip_solver.pyscipopt')
    def test_create_scip_solver(self, mock_pyscipopt):
        """Test creating SCIP solver."""
        mock_pyscipopt.Model.return_value = Mock()
        
        config = {"timeout": 3600, "parameters": {}}
        solver = SolverFactory.create_solver("scip", config)
        
        assert isinstance(solver, SCIPSolver)
        assert solver.timeout == 3600

    def test_create_solver_case_insensitive(self):
        """Test that solver creation is case insensitive."""
        config = {"timeout": 3600}
        
        with patch('src.mip_mcp.solvers.scip_solver.pyscipopt') as mock_pyscipopt:
            mock_pyscipopt.Model.return_value = Mock()
            
            # Test different cases
            solver1 = SolverFactory.create_solver("scip", config)
            solver2 = SolverFactory.create_solver("SCIP", config)
            solver3 = SolverFactory.create_solver("Scip", config)
            
            assert all(isinstance(s, SCIPSolver) for s in [solver1, solver2, solver3])

    def test_create_invalid_solver(self):
        """Test creating invalid solver raises ValueError."""
        config = {}
        
        with pytest.raises(ValueError) as exc_info:
            SolverFactory.create_solver("invalid_solver", config)
        
        assert "Unsupported solver: invalid_solver" in str(exc_info.value)
        assert "Available solvers: scip" in str(exc_info.value)

    def test_create_solver_with_config(self):
        """Test creating solver with specific configuration."""
        config = {
            "timeout": 1800,
            "parameters": {"limits/gap": 0.01}
        }
        
        with patch('src.mip_mcp.solvers.scip_solver.pyscipopt') as mock_pyscipopt:
            mock_pyscipopt.Model.return_value = Mock()
            
            solver = SolverFactory.create_solver("scip", config)
            
            assert solver.timeout == 1800
            assert solver.parameters == {"limits/gap": 0.01}