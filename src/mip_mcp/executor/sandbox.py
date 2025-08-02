"""Security sandbox for Python code execution."""

import ast
from typing import Set, List, Optional


class SecurityError(Exception):
    """Security-related error during code validation."""
    pass


class SecurityChecker:
    """Validates Python code for security threats."""
    
    # Dangerous functions that should not be allowed
    DANGEROUS_FUNCTIONS = {
        'eval', 'exec', 'compile', '__import__', 'open', 'file',
        'input', 'raw_input', 'reload', 'vars', 'globals', 'locals',
        'dir', 'getattr', 'setattr', 'delattr', 'hasattr', 'exit', 'quit'
    }
    
    # File writing methods that should be prohibited
    PROHIBITED_FILE_METHODS = {
        # PuLP methods
        'writeLP', 'writeMPS', 
        # Python-MIP methods
        'write',
        # General file operations
        'writelines', 'flush', 'close'
    }
    
    # Dangerous modules that should not be imported
    DANGEROUS_MODULES = {
        'os', 'sys', 'subprocess', 'shutil', 'glob', 'socket',
        'urllib', 'http', 'ftplib', 'smtplib', 'multiprocessing',
        'threading', 'pickle', 'marshal', 'shelve', 'importlib',
        'inspect', 'types', 'code', 'codeop'
    }
    
    # Allowed modules for optimization
    ALLOWED_MODULES = {
        'pulp', 'mip', 'math', 'statistics', 'numpy', 'pandas', 'scipy'
    }
    
    def validate_code(self, code: str) -> bool:
        """Validate Python code for security threats.
        
        Args:
            code: Python code to validate
            
        Returns:
            True if code is safe
            
        Raises:
            SecurityError: If dangerous operations are detected
        """
        if not code.strip():
            raise SecurityError("Empty code is not allowed")
        
        if len(code) > 10000:  # Configurable limit
            raise SecurityError("Code is too long (max 10000 characters)")
        
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise SecurityError(f"Syntax error in code: {e}")
        
        # Check for dangerous patterns
        visitor = DangerousNodeVisitor()
        visitor.visit(tree)
        
        if visitor.dangerous_nodes:
            raise SecurityError(
                f"Dangerous operations detected: {', '.join(visitor.dangerous_nodes)}"
            )
        
        return True


class DangerousNodeVisitor(ast.NodeVisitor):
    """AST visitor to detect dangerous operations."""
    
    def __init__(self):
        self.dangerous_nodes: List[str] = []
    
    def visit_Call(self, node):
        """Check function calls for dangerous operations."""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in SecurityChecker.DANGEROUS_FUNCTIONS:
                self.dangerous_nodes.append(f"Function call: {func_name}")
        
        elif isinstance(node.func, ast.Attribute):
            # Check for method calls on dangerous modules
            if isinstance(node.func.value, ast.Name):
                obj_name = node.func.value.id
                method_name = node.func.attr
                
                # Check for dangerous method patterns
                if method_name in ['system', 'popen', 'spawn', 'fork']:
                    self.dangerous_nodes.append(f"Method call: {obj_name}.{method_name}")
                
                # Check for prohibited file writing methods
                if method_name in SecurityChecker.PROHIBITED_FILE_METHODS:
                    self.dangerous_nodes.append(f"Prohibited file operation: {obj_name}.{method_name}")
        
        self.generic_visit(node)
    
    def visit_Import(self, node):
        """Check imports for dangerous modules."""
        for alias in node.names:
            module_name = alias.name.split('.')[0]  # Get top-level module
            
            if module_name in SecurityChecker.DANGEROUS_MODULES:
                self.dangerous_nodes.append(f"Import: {module_name}")
            elif module_name not in SecurityChecker.ALLOWED_MODULES:
                # Check if it's a standard library module (basic whitelist)
                if not self._is_safe_stdlib_module(module_name):
                    self.dangerous_nodes.append(f"Unauthorized import: {module_name}")
        
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        """Check 'from module import ...' statements."""
        if node.module:
            module_name = node.module.split('.')[0]
            
            if module_name in SecurityChecker.DANGEROUS_MODULES:
                self.dangerous_nodes.append(f"Import from: {module_name}")
            elif module_name not in SecurityChecker.ALLOWED_MODULES:
                if not self._is_safe_stdlib_module(module_name):
                    self.dangerous_nodes.append(f"Unauthorized import from: {module_name}")
        
        self.generic_visit(node)
    
    def visit_Attribute(self, node):
        """Check attribute access for dangerous patterns."""
        if isinstance(node.value, ast.Name):
            obj_name = node.value.id
            attr_name = node.attr
            
            # Check for dangerous attribute access patterns
            if attr_name in ['__dict__', '__class__', '__bases__', '__subclasses__']:
                self.dangerous_nodes.append(f"Dangerous attribute access: {obj_name}.{attr_name}")
        
        self.generic_visit(node)
    
    def visit_Subscript(self, node):
        """Check subscript operations for dangerous patterns."""
        # Check for attempts to access __builtins__ or similar
        if isinstance(node.value, ast.Name) and node.value.id == '__builtins__':
            self.dangerous_nodes.append("Access to __builtins__")
        
        self.generic_visit(node)
    
    def _is_safe_stdlib_module(self, module_name: str) -> bool:
        """Check if a module is a safe standard library module."""
        # Allow basic math/data modules
        safe_stdlib = {
            'math', 'statistics', 'random', 'decimal', 'fractions',
            'collections', 'itertools', 'functools', 'operator',
            'json', 're', 'string', 'datetime', 'time', 'calendar',
            'copy', 'pprint', 'enum', 'typing'
        }
        return module_name in safe_stdlib