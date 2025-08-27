from gurobipy import Model, GRB, quicksum


def gurobi_solver(u, D, n_select, lam=1.0, time_limit=5.0):
    """Solve quadratic integer programming problem for subset selection with unary and pairwise terms.

    Args:
        u: Unary scores for each item
        D: Pairwise similarity matrix (upper triangular)
        n_select: Number of items to select
        lam: Weight for pairwise term (default: 1.0)
        time_limit: Solver time limit in seconds (default: 5.0)
    """
    n = len(u)
    model = Model()
    model.Params.LogToConsole = 0
    model.Params.TimeLimit = time_limit
    model.Params.OutputFlag = 0

    # Variables: x[i] in {0,1}
    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    # Constraint: exactly k items selected
    model.addConstr(quicksum(x[i] for i in range(n)) == n_select, name="select_k")

    # Objective: sum of unary + lambda * pairwise
    linear_part = quicksum(u[i] * x[i] for i in range(n))
    quadratic_part = quicksum(lam * D[i, j] * x[i] * x[j] for i in range(n) for j in range(i + 1, n))

    model.setObjective(linear_part + quadratic_part, GRB.MAXIMIZE)

    model.optimize()
    selected_indices = [i for i in range(n) if x[i].X > 0.5]
    return selected_indices
