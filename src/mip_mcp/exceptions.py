"""Exception classes for MIP-MCP."""


class CodeExecutionError(Exception):
    """Error during code execution."""
    pass


class SecurityError(Exception):
    """Security violation detected during code execution."""
    pass