"""Solution validation utilities for PuLP optimization problems."""

from typing import Dict, Any, List, Optional
from ..utils.logger import get_logger

logger = get_logger(__name__)


class SolutionValidator:
    """Validates optimization solutions against problem constraints."""
    
    def __init__(self, tolerance: float = 1e-6):
        """Initialize the validator.
        
        Args:
            tolerance: Numerical tolerance for constraint checking
        """
        self.tolerance = tolerance
    
    def validate_solution(
        self, 
        pulp_problem: Any, 
        solution: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate solution against problem constraints.
        
        Args:
            pulp_problem: PuLP problem object
            solution: Solution dictionary with variable values
            
        Returns:
            Validation result dictionary
        """
        try:
            # Extract solution values
            variable_values = solution.get('variables', {})
            
            validation_result = {
                'is_valid': True,
                'constraint_violations': [],
                'bound_violations': [],
                'integer_violations': [],
                'tolerance_used': self.tolerance,
                'summary': {
                    'total_constraints_checked': 0,
                    'total_variables_checked': len(variable_values),
                    'violations_found': 0
                }
            }
            
            # Check linear constraints
            constraint_violations = self._check_linear_constraints(
                pulp_problem, variable_values
            )
            validation_result['constraint_violations'] = constraint_violations
            validation_result['summary']['total_constraints_checked'] = len(
                getattr(pulp_problem, 'constraints', {})
            )
            
            # Check variable bounds
            bound_violations = self._check_variable_bounds(
                pulp_problem, variable_values
            )
            validation_result['bound_violations'] = bound_violations
            
            # Check integer constraints
            integer_violations = self._check_integer_constraints(
                pulp_problem, variable_values
            )
            validation_result['integer_violations'] = integer_violations
            
            # Determine overall validity
            total_violations = (
                len(constraint_violations) + 
                len(bound_violations) + 
                len(integer_violations)
            )
            validation_result['summary']['violations_found'] = total_violations
            validation_result['is_valid'] = total_violations == 0
            
            logger.info(f"Solution validation completed: {total_violations} violations found")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Solution validation failed: {e}")
            return {
                'is_valid': False,
                'error': f"Validation failed: {e}",
                'constraint_violations': [],
                'bound_violations': [],
                'integer_violations': [],
                'tolerance_used': self.tolerance
            }
    
    def _check_linear_constraints(
        self, 
        problem: Any, 
        variable_values: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Check linear constraints satisfaction."""
        violations = []
        
        try:
            import pulp
            
            constraints = getattr(problem, 'constraints', {})
            
            for constraint_name, constraint in constraints.items():
                try:
                    # Calculate left-hand side value
                    lhs_value = 0.0
                    
                    # Extract coefficients and variables from constraint
                    for var, coeff in constraint.items():
                        if hasattr(var, 'name') and var.name in variable_values:
                            lhs_value += coeff * variable_values[var.name]
                    
                    # Get constraint constant (right-hand side)
                    rhs_value = constraint.constant if hasattr(constraint, 'constant') else 0.0
                    
                    # Get constraint sense
                    sense = constraint.sense if hasattr(constraint, 'sense') else None
                    
                    # Check constraint satisfaction
                    violation = self._check_constraint_satisfaction(
                        lhs_value, sense, rhs_value, constraint_name
                    )
                    
                    if violation:
                        violations.append(violation)
                        
                except Exception as e:
                    logger.warning(f"Failed to check constraint {constraint_name}: {e}")
                    
        except ImportError:
            logger.error("PuLP not available for constraint checking")
        except Exception as e:
            logger.error(f"Error checking linear constraints: {e}")
        
        return violations
    
    def _check_constraint_satisfaction(
        self, 
        lhs_value: float, 
        sense: Any, 
        rhs_value: float, 
        constraint_name: str
    ) -> Optional[Dict[str, Any]]:
        """Check if a single constraint is satisfied."""
        try:
            import pulp
            
            if sense == pulp.LpConstraintLE:  # <=
                if lhs_value > rhs_value + self.tolerance:
                    return {
                        'constraint_name': constraint_name,
                        'lhs_value': lhs_value,
                        'sense': '<=',
                        'rhs_value': rhs_value,
                        'violation': lhs_value - rhs_value,
                        'type': 'upper_bound'
                    }
            elif sense == pulp.LpConstraintGE:  # >=
                if lhs_value < rhs_value - self.tolerance:
                    return {
                        'constraint_name': constraint_name,
                        'lhs_value': lhs_value,
                        'sense': '>=',
                        'rhs_value': rhs_value,
                        'violation': rhs_value - lhs_value,
                        'type': 'lower_bound'
                    }
            elif sense == pulp.LpConstraintEQ:  # ==
                if abs(lhs_value - rhs_value) > self.tolerance:
                    return {
                        'constraint_name': constraint_name,
                        'lhs_value': lhs_value,
                        'sense': '==',
                        'rhs_value': rhs_value,
                        'violation': abs(lhs_value - rhs_value),
                        'type': 'equality'
                    }
                    
        except ImportError:
            logger.error("PuLP not available for constraint sense checking")
        
        return None
    
    def _check_variable_bounds(
        self, 
        problem: Any, 
        variable_values: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Check variable bound constraints."""
        violations = []
        
        try:
            variables = getattr(problem, 'variables', lambda: [])()
            
            for var in variables:
                if hasattr(var, 'name') and var.name in variable_values:
                    value = variable_values[var.name]
                    
                    # Check lower bound
                    if hasattr(var, 'lowBound') and var.lowBound is not None:
                        if value < var.lowBound - self.tolerance:
                            violations.append({
                                'variable': var.name,
                                'value': value,
                                'bound_type': 'lower',
                                'bound_value': var.lowBound,
                                'violation': var.lowBound - value
                            })
                    
                    # Check upper bound
                    if hasattr(var, 'upBound') and var.upBound is not None:
                        if value > var.upBound + self.tolerance:
                            violations.append({
                                'variable': var.name,
                                'value': value,
                                'bound_type': 'upper',
                                'bound_value': var.upBound,
                                'violation': value - var.upBound
                            })
                            
        except Exception as e:
            logger.error(f"Error checking variable bounds: {e}")
        
        return violations
    
    def _check_integer_constraints(
        self, 
        problem: Any, 
        variable_values: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Check integer and binary constraints."""
        violations = []
        
        try:
            import pulp
            
            variables = getattr(problem, 'variables', lambda: [])()
            
            for var in variables:
                if hasattr(var, 'name') and var.name in variable_values:
                    value = variable_values[var.name]
                    
                    # Check integer constraint
                    if hasattr(var, 'cat') and var.cat == pulp.LpInteger:
                        if abs(value - round(value)) > self.tolerance:
                            violations.append({
                                'variable': var.name,
                                'value': value,
                                'expected_type': 'integer',
                                'deviation': abs(value - round(value)),
                                'rounded_value': round(value)
                            })
                    
                    # Check binary constraint
                    elif hasattr(var, 'cat') and var.cat == pulp.LpBinary:
                        # Check if it's integer first
                        if abs(value - round(value)) > self.tolerance:
                            violations.append({
                                'variable': var.name,
                                'value': value,
                                'expected_type': 'binary_integer',
                                'deviation': abs(value - round(value)),
                                'valid_values': [0, 1]
                            })
                        # Check if it's in [0, 1]
                        elif not (0 <= value <= 1):
                            violations.append({
                                'variable': var.name,
                                'value': value,
                                'expected_type': 'binary_range',
                                'valid_range': '[0, 1]',
                                'valid_values': [0, 1]
                            })
                            
        except ImportError:
            logger.error("PuLP not available for integer constraint checking")
        except Exception as e:
            logger.error(f"Error checking integer constraints: {e}")
        
        return violations