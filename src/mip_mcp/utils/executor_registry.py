"""ExecutorRegistry for tracking and cleaning up active executors."""

import asyncio
import weakref
from typing import TYPE_CHECKING, ClassVar

from ..utils.logger import get_logger

if TYPE_CHECKING:
    from ..executor.pyodide_executor import PyodideExecutor

logger = get_logger(__name__)


class ExecutorRegistry:
    """Registry to track active PyodideExecutor instances for cleanup."""

    _executors: ClassVar[set[weakref.ReferenceType["PyodideExecutor"]]] = set()
    _lock: ClassVar[asyncio.Lock] = asyncio.Lock()

    @classmethod
    async def register(cls, executor: "PyodideExecutor") -> None:
        """Register an executor for tracking.

        Args:
            executor: PyodideExecutor instance to track
        """
        async with cls._lock:
            # Use weak reference to avoid circular references
            weak_ref = weakref.ref(executor, cls._cleanup_dead_ref)
            cls._executors.add(weak_ref)
            logger.debug(f"Registered executor (total: {len(cls._executors)})")

    @classmethod
    async def unregister(cls, executor: "PyodideExecutor") -> None:
        """Unregister an executor from tracking.

        Args:
            executor: PyodideExecutor instance to stop tracking
        """
        async with cls._lock:
            # Find and remove the weak reference for this executor
            to_remove = None
            for weak_ref in cls._executors:
                if weak_ref() is executor:
                    to_remove = weak_ref
                    break

            if to_remove:
                cls._executors.discard(to_remove)
                logger.debug(f"Unregistered executor (total: {len(cls._executors)})")

    @classmethod
    def _cleanup_dead_ref(cls, weak_ref: weakref.ReferenceType) -> None:
        """Clean up dead weak references (called automatically by weakref)."""
        cls._executors.discard(weak_ref)

    @classmethod
    async def cleanup_all(cls, silent: bool = False) -> None:
        """Clean up all registered executors.
        
        Args:
            silent: If True, suppress all logging (useful during shutdown)
        """
        async with cls._lock:
            # Get live executor instances
            executors = []
            dead_refs = set()

            for weak_ref in cls._executors.copy():
                executor = weak_ref()
                if executor is not None:
                    executors.append(executor)
                else:
                    dead_refs.add(weak_ref)

            # Clean up dead references
            cls._executors -= dead_refs

            if not executors:
                if not silent:
                    logger.debug("No active executors to clean up")
                return

            if not silent:
                logger.info(f"Cleaning up {len(executors)} active executors...")

            # Create cleanup tasks with timeout
            cleanup_tasks = [executor.cleanup() for executor in executors]

            try:
                # Wait for all cleanups with timeout
                await asyncio.wait_for(
                    asyncio.gather(*cleanup_tasks, return_exceptions=True),
                    timeout=15.0  # 15 second total timeout for all cleanups
                )
                if not silent:
                    logger.info("All executors cleaned up successfully")
            except asyncio.TimeoutError:
                if not silent:
                    logger.warning("Executor cleanup timed out after 15 seconds")
            except Exception as e:
                if not silent:
                    logger.error(f"Error during executor cleanup: {e}")
            finally:
                # Clear the registry
                cls._executors.clear()

    @classmethod
    def get_active_count(cls) -> int:
        """Get count of active executors."""
        # Clean up dead references first
        alive_refs = set()
        for weak_ref in cls._executors:
            if weak_ref() is not None:
                alive_refs.add(weak_ref)
        cls._executors = alive_refs
        return len(cls._executors)
