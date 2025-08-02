"""Solver abstraction layer."""

from .base import BaseSolver, SolverError
from .scip_solver import SCIPSolver
from .factory import SolverFactory

__all__ = ["BaseSolver", "SolverError", "SCIPSolver", "SolverFactory"]