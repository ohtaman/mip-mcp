"""Solver factory for creating solver instances."""

from typing import Any, ClassVar

from .base import BaseSolver
from .scip_solver import SCIPSolver


class SolverFactory:
    """Factory class for creating solver instances."""

    # Registry of available solvers
    _SOLVER_REGISTRY: ClassVar[dict[str, type[BaseSolver]]] = {
        "scip": SCIPSolver,
    }

    @classmethod
    def get_available_solvers(cls) -> list[str]:
        """Get list of available solver names.

        Returns:
            List of available solver names
        """
        return list(cls._SOLVER_REGISTRY.keys())

    @classmethod
    def create_solver(cls, solver_name: str, config: dict[str, Any]) -> BaseSolver:
        """Create a solver instance.

        Args:
            solver_name: Name of the solver to create
            config: Configuration dictionary for the solver

        Returns:
            Solver instance

        Raises:
            ValueError: If solver is not supported
        """
        solver_name = solver_name.lower()

        if solver_name not in cls._SOLVER_REGISTRY:
            available = ", ".join(cls.get_available_solvers())
            raise ValueError(
                f"Unsupported solver: {solver_name}. Available solvers: {available}"
            )

        solver_class = cls._SOLVER_REGISTRY[solver_name]
        return solver_class(config)

    @classmethod
    def is_solver_available(cls, solver_name: str) -> bool:
        """Check if a solver is available.

        Args:
            solver_name: Name of the solver to check

        Returns:
            True if solver is available
        """
        return solver_name.lower() in cls._SOLVER_REGISTRY
