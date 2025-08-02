"""MIP library detection utilities."""

import ast
import re
from typing import Set, Optional, List
from enum import Enum

from ..utils.logger import get_logger

logger = get_logger(__name__)


class MIPLibrary(Enum):
    """Supported MIP libraries."""
    PULP = "pulp"
    PYTHON_MIP = "python-mip"
    UNKNOWN = "unknown"


class MIPLibraryDetector:
    """Detects which MIP library is being used in Python code."""
    
    # Import patterns for each library
    LIBRARY_IMPORTS = {
        MIPLibrary.PULP: {
            'import_patterns': [
                r'import\s+pulp',
                r'from\s+pulp\s+import',
            ],
            'usage_patterns': [
                r'pulp\.',
                r'LpProblem\s*\(',
                r'LpVariable\s*\(',
                r'LpMaximize',
                r'LpMinimize',
                r'writeLP\s*\(',
                r'writeMPS\s*\(',
            ]
        },
        MIPLibrary.PYTHON_MIP: {
            'import_patterns': [
                r'import\s+mip',
                r'from\s+mip\s+import',
            ],
            'usage_patterns': [
                r'mip\.',
                r'Model\s*\(',
                r'add_var\s*\(',
                r'add_constr\s*\(',
                r'MAXIMIZE',
                r'MINIMIZE',
                r'\.write\s*\(',
                r'\.optimize\s*\(',
            ]
        }
    }
    
    def detect_library(self, code: str) -> MIPLibrary:
        """Detect MIP library from code.
        
        Args:
            code: Python code to analyze
            
        Returns:
            Detected MIP library
        """
        try:
            # Try AST analysis first
            library = self._detect_via_ast(code)
            if library != MIPLibrary.UNKNOWN:
                logger.info(f"Library detected via AST: {library.value}")
                return library
            
            # Fallback to regex pattern matching
            library = self._detect_via_patterns(code)
            logger.info(f"Library detected via patterns: {library.value}")
            return library
            
        except Exception as e:
            logger.error(f"Library detection failed: {e}")
            return MIPLibrary.UNKNOWN
    
    def _detect_via_ast(self, code: str) -> MIPLibrary:
        """Detect library using AST analysis."""
        try:
            tree = ast.parse(code)
            visitor = ImportVisitor()
            visitor.visit(tree)
            
            imports = visitor.imports
            
            # Check for library-specific imports
            if any('pulp' in imp for imp in imports):
                return MIPLibrary.PULP
            elif any('mip' in imp for imp in imports):
                return MIPLibrary.PYTHON_MIP
            
            return MIPLibrary.UNKNOWN
            
        except SyntaxError:
            logger.warning("Code has syntax errors, falling back to pattern matching")
            return MIPLibrary.UNKNOWN
        except Exception as e:
            logger.error(f"AST analysis failed: {e}")
            return MIPLibrary.UNKNOWN
    
    def _detect_via_patterns(self, code: str) -> MIPLibrary:
        """Detect library using regex patterns."""
        scores = {library: 0 for library in MIPLibrary if library != MIPLibrary.UNKNOWN}
        
        for library, patterns in self.LIBRARY_IMPORTS.items():
            if library == MIPLibrary.UNKNOWN:
                continue
                
            # Check import patterns
            for pattern in patterns['import_patterns']:
                if re.search(pattern, code, re.IGNORECASE):
                    scores[library] += 10  # High weight for imports
            
            # Check usage patterns
            for pattern in patterns['usage_patterns']:
                matches = len(re.findall(pattern, code, re.IGNORECASE))
                scores[library] += matches * 2  # Medium weight for usage
        
        # Return library with highest score
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        
        return MIPLibrary.UNKNOWN
    
    def get_supported_libraries(self) -> List[str]:
        """Get list of supported library names."""
        return [lib.value for lib in MIPLibrary if lib != MIPLibrary.UNKNOWN]
    
    def validate_library_choice(self, library: str) -> MIPLibrary:
        """Validate and convert library string to enum.
        
        Args:
            library: Library name string
            
        Returns:
            MIPLibrary enum value
        """
        library_lower = library.lower()
        
        if library_lower in ['pulp']:
            return MIPLibrary.PULP
        elif library_lower in ['python-mip', 'mip', 'python_mip']:
            return MIPLibrary.PYTHON_MIP
        elif library_lower in ['auto', 'detect']:
            return MIPLibrary.UNKNOWN  # Will trigger auto-detection
        else:
            logger.warning(f"Unknown library: {library}")
            return MIPLibrary.UNKNOWN


class ImportVisitor(ast.NodeVisitor):
    """AST visitor to extract import statements."""
    
    def __init__(self):
        self.imports: Set[str] = set()
    
    def visit_Import(self, node):
        """Visit import statements."""
        for alias in node.names:
            self.imports.add(alias.name)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        """Visit from-import statements."""
        if node.module:
            self.imports.add(node.module)
            for alias in node.names:
                self.imports.add(f"{node.module}.{alias.name}")
        self.generic_visit(node)


def detect_mip_library(code: str) -> str:
    """Convenience function to detect MIP library.
    
    Args:
        code: Python code to analyze
        
    Returns:
        Library name as string
    """
    detector = MIPLibraryDetector()
    library = detector.detect_library(code)
    return library.value