import cvxpy as cp
import numpy as np


def graph_problem_relaxation(gcs, problem, callback=None):

    # compute conic programs on edges and vertices
    gcs.to_conic()

    # binary variables
    yv = gcs.vertex_relaxed_binaries()
    ye = gcs.edge_relaxed_binaries()

    # continuous variables
    # conic.num_variables 为 锥规划问题中变量的数量
    xv = np.array([cp.Variable(v.conic.num_variables) for v in gcs.vertices])
    zv = np.array([cp.Variable(v.conic.num_variables) for v in gcs.vertices])
    ze = np.array([cp.Variable(e.conic.num_variables) for e in gcs.edges])
    ze_out = np.array([cp.Variable(e.tail.conic.num_variables) for e in gcs.edges])
    ze_inc = np.array([cp.Variable(e.head.conic.num_variables) for e in gcs.edges])

    cost = 0
    constraints = []

    for i, v in enumerate(gcs.vertices):
        
        # cost on the vertices including domain constraint
        # v.conic.c @ zv[i] + v.conic.d * yv[i]
        cost += v.conic.eval_cost(zv[i], yv[i])
        # Ai @ zv[i] + bi * yv[i], 再转换成锥约束
        constraints += v.conic.eval_constraints(zv[i], yv[i])
        
    for k, e in enumerate(gcs.edges):
        
        # cost on the edges including domain constraint
        cost += e.conic.eval_cost(ze[k], ye[k])
        constraints += e.conic.eval_constraints(ze[k], ye[k])
        constraints += e.tail.conic.eval_constraints(ze_out[k], ye[k])
        constraints += e.head.conic.eval_constraints(ze_inc[k], ye[k])
        
        # euqate auxiliary variables on the egdes
        for variable in e.tail.variables:
            ze_var = e.conic.select_variable(variable, ze[k], reshape=False)
            ze_out_var = e.tail.conic.select_variable(variable, ze_out[k], reshape=False)
            if ze_var is not None and ze_out_var is not None:
                constraints.append(ze_var == ze_out_var)
        for variable in e.head.variables:
            ze_var = e.conic.select_variable(variable, ze[k], reshape=False)
            ze_inc_var = e.head.conic.select_variable(variable, ze_inc[k], reshape=False)
            if ze_var is not None and ze_inc_var is not None:
                constraints.append(ze_var == ze_inc_var)

    probelm_specific_constraints = problem(gcs, xv, zv, ze_out, ze_inc)
    constraints += probelm_specific_constraints

    # solve problem
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(verbose=False)
    # prob.solve(verbose=False, solver=cp.SCS, eps=1e-3, max_iters=10000)
    # prob.solve(solver=cp.SCS, verbose=True, max_iters=1000)
    if callback is not None:
        print('callback is not None')
        while True:
            new_constraints = callback(yv, ye)
            if len(new_constraints) == 0:
                break
            constraints += new_constraints
            prob = cp.Problem(cp.Minimize(cost), constraints)
            prob.solve()

    if prob.status == 'optimal':
        tol = 1e-4

        # set values for vertices
        for i, vertex in enumerate(gcs.vertices):
            if prob.status == "optimal" and vertex.y_relaxed.value > tol:
                for variable in vertex.variables:
                    variable.value = vertex.conic.select_variable(variable, xv[i].value)
                    # print(variable.value)
            else:
                # vertex.y_relaxed.value = None
                for variable in vertex.variables:
                    variable.value = None

        # set values for edges
        for k, edge in enumerate(gcs.edges):
            if prob.status == "optimal" and edge.y_relaxed.value > tol:
                # print(edge.y_relaxed.value)
                for variable in edge.variables:
                    ze_var = edge.conic.select_variable(variable, ze[k].value)
                    variable.value = ze_var / edge.y_relaxed.value
                    # print(variable.value)
            else:
                # edge.y_relaxed.value = None
                for variable in edge.variables:
                    variable.value = None

    return prob
