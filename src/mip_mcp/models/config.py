"""Configuration data models."""

from typing import Dict, Any
from pydantic import BaseModel, Field


class ServerConfig(BaseModel):
    """Server configuration."""
    name: str = "mip-mcp"
    version: str = "0.1.0"


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class ExecutorConfig(BaseModel):
    """Python code executor configuration."""
    enabled: bool = True
    timeout: int = Field(300, description="Timeout for code execution in seconds")
    memory_limit: str = "1GB"


class SolverConfig(BaseModel):
    """Solver configuration."""
    default: str = "scip"
    timeout: int = Field(3600, description="Timeout for solver in seconds")


class ValidationConfig(BaseModel):
    """Input validation configuration."""
    max_variables: int = 100000
    max_constraints: int = 100000
    max_code_length: int = 10000


class Config(BaseModel):
    """Main configuration container."""
    server: ServerConfig = Field(default_factory=ServerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    executor: ExecutorConfig = Field(default_factory=ExecutorConfig)
    solvers: SolverConfig = Field(default_factory=SolverConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create config from dictionary."""
        return cls.model_validate(data)