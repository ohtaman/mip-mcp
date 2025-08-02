"""Sample optimization problems for testing."""

SIMPLE_LP = """
import pulp

# Simple Linear Programming Problem
prob = pulp.LpProblem("Simple_LP", pulp.LpMaximize)

# Variables
x = pulp.LpVariable("x", lowBound=0)
y = pulp.LpVariable("y", lowBound=0)

# Objective
prob += 3*x + 2*y

# Constraints
prob += 2*x + y <= 20
prob += 4*x + 5*y <= 10
prob += -x + 2*y <= 2

# Solve
prob.solve()
"""

KNAPSACK_PROBLEM = """
import pulp

# 0-1 Knapsack Problem
weights = [10, 20, 30]
values = [60, 100, 120]
capacity = 50

prob = pulp.LpProblem("Knapsack", pulp.LpMaximize)

# Binary variables for each item
x = [pulp.LpVariable(f"x_{i}", cat='Binary') for i in range(len(weights))]

# Objective: maximize value
prob += pulp.lpSum([values[i] * x[i] for i in range(len(values))])

# Constraint: weight limit
prob += pulp.lpSum([weights[i] * x[i] for i in range(len(weights))]) <= capacity

# Solve
prob.solve()
"""

TRANSPORTATION_PROBLEM = """
import pulp

# Transportation Problem
supply = [20, 30, 25]
demand = [15, 25, 35]
costs = [
    [4, 6, 8],
    [5, 3, 7],
    [6, 4, 5]
]

prob = pulp.LpProblem("Transportation", pulp.LpMinimize)

# Variables: amount shipped from supply i to demand j
x = [[pulp.LpVariable(f"x_{i}_{j}", lowBound=0)
      for j in range(len(demand))]
     for i in range(len(supply))]

# Objective: minimize total cost
prob += pulp.lpSum([costs[i][j] * x[i][j]
                   for i in range(len(supply))
                   for j in range(len(demand))])

# Supply constraints
for i in range(len(supply)):
    prob += pulp.lpSum([x[i][j] for j in range(len(demand))]) <= supply[i]

# Demand constraints
for j in range(len(demand)):
    prob += pulp.lpSum([x[i][j] for i in range(len(supply))]) >= demand[j]

# Solve
prob.solve()
"""

INVALID_SYNTAX = """
import pulp

# Invalid syntax
prob = LpProblem("Test")  # Missing pulp prefix
x = LpVariable("x", lowBound=0)  # Missing pulp prefix

# Invalid constraint
prob += x >= "not_a_number"

# Solve
prob.solve()
"""

SECURITY_VIOLATION = """
import os
import subprocess
import pulp

# Security violations
os.system("echo 'potential security issue'")
subprocess.run(["ls", "-la"])

# File operations
with open("/etc/passwd", "r") as f:
    content = f.read()

# Valid PuLP code mixed with security issues
prob = pulp.LpProblem("Test", pulp.LpMaximize)
x = pulp.LpVariable("x", lowBound=0)
prob += x
prob += x <= 10
prob.solve()
"""

NO_PROBLEM_DEFINED = """
import pulp
import numpy as np

# Code that doesn't define an optimization problem
data = [1, 2, 3, 4, 5]
result = sum(data)
print(f"Sum is: {result}")

# No LpProblem created
x = 42
y = x * 2
"""

COMPLEX_INTEGER_PROGRAM = """
import pulp

# Complex Integer Programming Problem
n_items = 5
n_bins = 3

# Item weights and bin capacities
weights = [12, 7, 18, 6, 15]
bin_capacity = 25

prob = pulp.LpProblem("Bin_Packing", pulp.LpMinimize)

# Binary variables: x[i][j] = 1 if item i is in bin j
x = [[pulp.LpVariable(f"x_{i}_{j}", cat='Binary')
      for j in range(n_bins)]
     for i in range(n_items)]

# Binary variables: y[j] = 1 if bin j is used
y = [pulp.LpVariable(f"y_{j}", cat='Binary') for j in range(n_bins)]

# Objective: minimize number of bins used
prob += pulp.lpSum(y)

# Each item must be assigned to exactly one bin
for i in range(n_items):
    prob += pulp.lpSum([x[i][j] for j in range(n_bins)]) == 1

# Bin capacity constraints
for j in range(n_bins):
    prob += pulp.lpSum([weights[i] * x[i][j] for i in range(n_items)]) <= bin_capacity * y[j]

# Solve
prob.solve()
"""
