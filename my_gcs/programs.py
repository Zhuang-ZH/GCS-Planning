import numpy as np
import cvxpy as cp
from numbers import Number


class ConvexProgram:

    var_attributes = ["nonneg", "nonpos", "symmetric", "PSD", "NSD"]

    def __init__(self):
        self.variables = []
        self.constraints = []
        self.cost = 0
        self.conic = None
        self.y = cp.Variable(boolean=True)
        self.y_relaxed = cp.Variable(nonneg=True)
        self.constraints.append(self.y_relaxed >= 0)  # 添加约束 y_relaxed >= 0
        self.constraints.append(self.y_relaxed <= 1)  # 添加约束 y_relaxed <= 1

    def add_variable(self, shape, **kwargs):
        for attribute in kwargs:
            if not attribute in self.var_attributes:
                raise ValueError(f"Variable attribute {attribute} is not supported.")
        variable = cp.Variable(shape, **kwargs)
        self.variables.append(variable)
        return variable

    def add_constraint(self, constraint):
        self._verify_variables(constraint.variables())
        self.constraints.append(constraint)

    def add_constraints(self, constraints):
        for constraint in constraints:
            self.add_constraint(constraint)

    def add_cost(self, cost):
        self._verify_variables(cost.variables())
        self.cost += cost

    def to_conic(self):
        self.conic = ConicProgram(self.constraints, self.cost)

    def _verify_variables(self, variables):
        raise NotImplementedError


class ConicProgram:

    def __init__(self, constraints, cost):

        # corner case with constant cost and no constraints
        if isinstance(cost, Number) and len(constraints) == 0:
            self.aux_variables = []
            self.c = dict()
            self.d = float(cost)
            self.A = []
            self.b = []
            self.K = []
            return

        # 将给定的凸规划问题转换为锥规划问题
        prob = cp.Problem(cp.Minimize(cost), constraints)
        # 检查问题是否是DCP（凸规划）问题
        if not prob.is_dcp():
            raise ValueError(f"Problem is not DCP.")
        solver_opts = {"use_quad_obj": False}
        # 构建求解链
        chain = prob._construct_chain(solver_opts=solver_opts)
        # 移除求解链中的最后一个转换步骤，简化求解过程
        chain.reductions = chain.reductions[:-1]
        # 应用求解链，返回转换后的锥规划问题 
        conic_prob = chain.apply(prob)[0]

        # 每个变量在锥规划问题中的列索引范围创建一个映射，并将其存储在 self.columns 字典中
        self.columns = {}
        for v in conic_prob.variables:
            start = conic_prob.var_id_to_col[v.id]
            stop = start + v.size
            self.columns[v.id] = range(start, stop)

        # cost function
        # 将 conic_prob.c 转换为一个一维的 NumPy 数组，并将其分割为 c 和 d
        cd = conic_prob.c.toarray().flatten()
        # TODO: convert to sparse format
        self.c = cd[:-1]
        self.d = cd[-1]
        self.num_variables = len(self.c)

        # constraints
        cols = conic_prob.c.shape[0]
        Ab = conic_prob.A.toarray().reshape((-1, cols), order='F')
        # TODO: convert to sparse format
        self.A = [] # 约束矩阵
        self.b = [] # 约束向量
        self.K = [] # 约束类型
        first_row = 0
        for cone in conic_prob.constraints:
            last_row = first_row + cone.size
            self.A.append(Ab[first_row:last_row, :-1])
            self.b.append(Ab[first_row:last_row, -1])
            self.K.append(type(cone))
            first_row = last_row

    def eval_cost(self, x, t=1):
        return self.c @ x + self.d * t

    def eval_constraints(self, x, t=1):
        constraints = []
        for Ai, bi, Ki in zip(self.A, self.b, self.K):
            Axbt = Ai @ x + bi * t
            constraints.append(self.constrain_in_cone(Axbt, Ki))
        return constraints

    def select_variable(self, variable, x, reshape=True):
        if not variable.id in self.columns:
            return None
        value = x[self.columns[variable.id]]
        if reshape:
            if variable.is_matrix():
                if variable.is_symmetric():
                    n = variable.shape[0]
                    full = np.zeros((n, n))
                    full[np.triu_indices(n)] = value
                    value = full + full.T
                    value[np.diag_indices(n)] /= 2
                else:
                    value = value.reshape(variable.shape)
        return value

    @staticmethod
    def constrain_in_cone(x, K):
        if K == cp.constraints.Zero:
             return cp.constraints.Zero(x)
        elif K == cp.constraints.NonNeg:
             return cp.constraints.NonNeg(x)
        elif K == cp.constraints.NonPos:
             return cp.constraints.NonPos(x)
        elif K == cp.constraints.SOC:
            return cp.constraints.SOC(x[0], x[1:])
        elif K == cp.constraints.PSD:
            n = round(np.sqrt(x.size))
            x_mat = cp.reshape(x, (n, n))
            return cp.constraints.PSD(x_mat)
        else:
            raise NotImplementedError
