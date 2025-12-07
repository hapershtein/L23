import numpy as np
from scipy.optimize import linprog

# Coefficients of the objective function (to be minimized).
# We want to maximize 3*x1 + 5*x2, so we minimize -3*x1 - 5*x2.
c = np.array([-3, -5])

# Coefficients of the inequality constraints (A_ub @ x <= b_ub)
# Resource 1: x1 + x2 <= 4
# Resource 2: 2*x1 + x2 <= 6
# Resource 3: x1 + 3*x2 <= 9
A_ub = np.array([
    [1, 1],
    [2, 1],
    [1, 3]
])

# Right-hand side of the inequality constraints
b_ub = np.array([4, 6, 9])

# Bounds for the variables (x1 >= 0, x2 >= 0)
x1_bounds = (0, None)
x2_bounds = (0, None)
bounds = [x1_bounds, x2_bounds]

# Solve the linear programming problem
result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

# Print the results
if result.success:
    print("Optimization successful!")
    print(f"Optimal production quantity for product A (x1): {result.x[0]:.2f} units")
    print(f"Optimal production quantity for product B (x2): {result.x[1]:.2f} units")
    print(f"Maximum daily profit: {-result.fun:.2f} NIS") # -result.fun because we minimized the negative objective
else:
    print("Optimization failed.")
    print(f"Status: {result.message}")

