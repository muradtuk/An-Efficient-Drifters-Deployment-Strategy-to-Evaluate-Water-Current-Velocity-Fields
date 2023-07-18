"""*****************************************************************************************
MIT License
Copyright (c) 2022 Murad Tukan, Eli Biton, Roee Diamant
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


This file is an implementation of the volume approximation methodology used in
"Tukan, M., Maalouf, A., Feldman, D., & Poranne, R. (2022, October). Obstacle aware
sampling for path planning. In 2022 IEEE/RSJ International Conference on Intelligent
Robots and Systems (IROS) (pp. 13676-13683). IEEE."
*****************************************************************************************"""
import cvxpy as cp
import numpy as np
from scipy.linalg import null_space
import copy
from timeit import default_timer as timer
from skimage.morphology import convex_hull_image



lower_bound = None
upper_bound = None

temp_list = [2 ** i if i > 0 else -2 ** (-i) for i in range(-3,4)]
temp_list.append(0)
range_of_valid_steps = np.array(temp_list)


class OptimalSolver(object):
    def __init__(self, is_pixels=True, oracle=None, P=None, p=None, verbose=False):
        self.main_variable_type = is_pixels
        self.M = 2  # this is only to ensure that the problem is solvable (MIP)
        self.oracle = oracle
        self.P = P.flatten()
        self.d = None
        self.entry_convex_body = p
        self.verbose = verbose

    def __getstate__(self):
        # this method is called when you are
        # going to pickle the class, to know what to pickle
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def convertIndexToMatrixMultiplication(self, x, z, sizes):
        v = np.zeros(self.P.shape)
        print('here')
        idx = 0
        h = np.flip(np.array(sizes))
        h[-1] = 1
        idx = cp.matmul(x, h)
        v[int(idx.value)] = 1
        z.value = v
        return z

    def solveOptimizationProblem(self, direction_type_cost, needed_input, constraints):
        d = needed_input[1]
        constraints = []
        if self.oracle.plain_oracle:
            x = cp.Variable(self.oracle.flattened_data.shape[0], integer=self.main_variable_type)
            bounds = np.array([x[1] - x[0] for x in self.oracle.bounding_box if x[1] != x[0]])
            x.value = self.oracle.convertPhysicalIndex(self.entry_convex_body.astype("int"), is_bounding_box=True)
            constraints.extend([cp.matmul(self.oracle.coordinates[:-1,:], x) >= self.oracle.bounding_box[:,0],
                                cp.matmul(self.oracle.coordinates[:-1,:], x) <= self.oracle.bounding_box[:,1],
                                self.oracle.checkIfInsidePixelStyle(x) <= self.oracle.threshold,
                                x >= 0,
                                cp.sum(x) == 1])

            # constraints.extend([cp.matmul(self.oracle.coordinates, x) >= self.oracle.bounding_box[:, 0],
            #                     cp.matmul(self.oracle.coordinates, x) <= self.oracle.bounding_box[:, -1],
            #                     self.oracle.checkIfInsidePixelStyle(x) <= self.oracle.threshold,
            #                     x >= 0,
            #                     cp.sum(x) == 1])

        else:
            x = cp.Variable(needed_input[1], integer=self.main_variable_type)
            x.value = (np.hstack((self.entry_convex_body,1)) if d > self.entry_convex_body.shape[0]
                       else self.entry_convex_body).astype("int")
            constraints.extend([x >= self.oracle.bounding_box[:, 0], x <= self.oracle.bounding_box[:, -1],
                            self.oracle.checkIfInsideCPVersion(x) <= 0])

        if direction_type_cost:
            # simple linear programming, and in general convex programming
            if self.oracle.plain_oracle:
                cost = cp.matmul(needed_input[2], cp.matmul(self.oracle.coordinates[:-1,:], x))
            else:
                cost = cp.matmul(needed_input[2], x)
            problem_min = cp.Problem(cp.Minimize(cost), constraints)
            problem_min.solve()
            if problem_min.status != 'optimal':
                raise ValueError('We are doomed!')
            x_min = copy.deepcopy(x.value)
            x.value = self.oracle.convertPhysicalIndex(self.entry_convex_body.astype("int"), is_bounding_box=True)
            problem_max = cp.Problem(cp.Maximize(cost), constraints)
            problem_max.solve()#solver=cp.MOSEK, mosek_params={'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 0.7}, verbose=True)
            if problem_max.status != 'optimal':
                raise ValueError('We are doomed!')
            x_max = copy.deepcopy(x.value)
            if not self.oracle.plain_oracle:
                return x_min.astype("float" if not self.main_variable_type else "int"), \
                       x_max.astype("float" if not self.main_variable_type else "int")
            else:
                return np.dot(self.oracle.coordinates[:-1,:], x_min).astype("float" if not
                self.main_variable_type else "int"),\
                       np.dot(self.oracle.coordinates[:-1,:], x_max).astype("float" if not
                       self.main_variable_type else "int")

        B = cp.Variable(d, boolean=True)
        y = cp.Variable(d)
        helperF = cp.sum(y)

        temp_constraints = []

        phi = lambda z: cp.matmul(self.oracle.coordinates, z) if self.oracle.plain_oracle else z

        temp_constraints.extend([y >= cp.matmul(needed_input[2], phi(x) - needed_input[3])])
        temp_constraints.extend([y >= -cp.matmul(needed_input[2], phi(x) - needed_input[3])])

        diam = np.linalg.norm(needed_input[2].dot(np.hstack((self.oracle.bounding_box,
                                                             np.ones((self.oracle.bounding_box.shape[0],1)))).T))
        diam = np.linalg.norm(needed_input[2], 'fro') * np.linalg.norm(np.hstack((self.oracle.bounding_box,
                                                             np.ones((self.oracle.bounding_box.shape[0],1)))), 'fro')
        temp_constraints.extend([cp.matmul(needed_input[2], phi(x) - needed_input[3]) +
                                 2* diam * B >= y])
        temp_constraints.extend([-cp.matmul(needed_input[2], phi(x) - needed_input[3]) +
                                 2 * diam * (1-B) >= y])
        temp_constraints.extend([B <= 1])

        problem = cp.Problem(cp.Maximize(helperF), constraints + temp_constraints)
        start = timer()
        problem.solve()
        if problem.status != 'optimal':
            raise ValueError('What happened? Change the M value!')
        if self.verbose:
            print('Solving the relaxed \'NP\' hard problem took {} seconds'.format(timer() - start))
        return phi(x).value[:-1].astype("float" if not self.main_variable_type else "int")


class LinearProgrammingInTheDark(object):
    def __init__(self, P, cost_func, d, epsilon, point, is_pixels=True, hull_hyper=None, tol=0.01,
                 matrix_of_vecs=False, add_rows_one=True, verbose=False):
        self.P = P
        self.cost_func = cost_func
        upper_bound = np.array(P.shape)
        if matrix_of_vecs:
            upper_bound = upper_bound[:-1] - 1
        self.oracle = MemOracle(self.cost_func, cost_func_cp=hull_hyper, flattened_data=P.flatten(), dimensions=P.shape,
                                matrix_of_vecs=matrix_of_vecs, upper_bound=upper_bound, verbose=False)
        self.d = d
        self.lower_d = None
        self.eps = epsilon
        self.starting_point = point
        self.oracle.obtainBoundBox(point)
        self.lower_dimensional_body = False
        self.dims_to_keep = None
        self.irrelevant_dims = None
        self.verbose = verbose
        self.get_all_points = False
        # self.focusOracleOnPoint(point)

        sizes = tuple([x[1] + 1 - x[0] for x in self.oracle.bounding_box])
        coor = [np.arange(x[0], x[1] + 1) for x in self.oracle.bounding_box]
        base = np.array([x[0] for x in self.oracle.bounding_box]).astype("int")
        X = np.meshgrid(*coor, sparse=False)
        X = np.hstack([x.flatten()[:, np.newaxis] for x in X])
        XX = self.sortByColumns(X)
        Z = np.empty(sizes)
        for x in X:
            Z[tuple(x-base)] = int(self.oracle.cost_func(x))

        # check if lower dimenionsional is needed.
        idxs = np.where(np.array(sizes) == 1)[0]
        Z[Z <= 0] = 2
        Z = convex_hull_image(Z == 1).astype("int")
        Z[Z <= 0] = 2

        per_row = np.max(np.sum(Z == 1, axis=1))
        per_col = np.max(np.sum(Z == 1, axis=0))
        number_of_interest_points = np.count_nonzero(Z == 1)
        # if len(idxs) == self.d:
        #     raise ValueError('Apparently the smoothing technique is not working well!')
        # elif len(idxs) >= 1:
        if len(idxs) >= 1:
            list1 = [':' if i not in idxs else '0' for i in range(d)]
            remaining_dims = list(set(list(range(self.d))).difference(set(idxs.tolist())))
            exec('Z = Z[{}]'.format(','.join(list1)))
            point2 = 'point[[{}]]'.format(','.join([str(x) for x in remaining_dims]))
            self.lower_dimensional_body = True
            self.lower_d = self.d - len(idxs)
            self.dims_to_keep = remaining_dims
            self.irrelevant_dims = idxs
        else:
            point2 = 'point'
            self.lower_d = self.d
            if number_of_interest_points < 2 * self.d or (per_row == 1 and per_col == 1):
                self.get_all_points = True

        # chull, hull_equations = convex_hull_image(Z, return_hull_eq=True)
        if len(Z.flatten()) > 1:
            self.oracle.focusOracleOnPoint(Z.flatten(), add_rows_one=add_rows_one)
        self.optimal_solver = OptimalSolver(is_pixels, self.oracle, self.P, p=eval(point2, locals()))
        self.tol = tol


        # self.P_constant = cp.Constant()
    def __getstate__(self):
        # this method is called when you are
        # going to pickle the class, to know what to pickle
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def sortByColumns(self, Z):
        for i in range(len(Z.shape)-1, -1, -1):
            if i == len(Z.shape)-1:
                Z = Z[Z[:,i].argsort()]
            else:
                Z = Z[Z[:, i].argsort(kind='mergesort')]

        return Z

    def setInitialPoint(self, point):
        self.starting_point = point
        self.oracle.obtainBoundBox(self.starting_point)

    def volumeApproximation(self):
        """
        make sure to embed p in to the equation, i.e., first we find the farthest points in both {+,-} of the direction
        :return:
        """
        # direction = np.ones((self.lower_d, 1)).flatten()
        direction = np.eye(self.lower_d, 1).flatten()
        needed_input = [self.P.shape, self.lower_d]
        S = np.empty((self.lower_d, self.lower_d))
        Q = np.empty((2 * self.lower_d, self.lower_d))
        counter = 0
        x = cp.Variable(self.lower_d, integer=True)
        x.value = np.zeros((self.lower_d, )).astype("int")

        constraints = None

        while counter < self.lower_d:
            if len(needed_input) == 2:
                needed_input.append(direction)
            else:
                needed_input[2] = direction
            needed_input.append(x)
            alpha, beta = self.optimal_solver.solveOptimizationProblem(direction_type_cost=True,
                                                                       needed_input=needed_input,
                                                                       constraints=constraints)
            v = beta - alpha
            v = v / np.linalg.norm(v)
            S[counter, :] = v
            orthogonal_null_space = null_space(S[:counter+1, :]).T
            direction = orthogonal_null_space[0] if orthogonal_null_space.shape[0] > 1 else orthogonal_null_space.flatten()
            Q[2 * counter, :] = alpha
            Q[2 * counter + 1, :] = beta
            counter += 1

        return Q

    def computeLambda(self, Q, weights):
        QQ = np.hstack((Q, np.ones((Q.shape[0], 1))))
        l = [QQ[i][:, np.newaxis].dot(QQ[i][:, np.newaxis].T) * weights[i] for i in range(QQ.shape[0])]
        return np.linalg.inv(sum(l))

    def stoppingCriterion(self,Q, weights):
        QQ = np.hstack((Q, np.ones((Q.shape[0], 1))))
        c = np.dot(Q.T, weights)
        L = self.computeLambda(Q, weights)
        val = np.array([L.dot(QQ[i]).dot(QQ[i]) for i in range(QQ.shape[0]) if weights[i] > 0])
        return (np.all(np.logical_and(val >= ((1 - self.eps) * (1 + self.lower_d)),
                                     val <= ((1 + self.eps) * (1 + self.lower_d)))) and
                (self.oracle.checkIfInsidePixelStyleNumpyVer(np.round(c))<=1))

    def computeAMVEE(self):
        """
        Yildirim way

        :return:
        """
        start = timer()
        Q = self.volumeApproximation()
        end = timer()
        if self.verbose:
            print('The volume approximation technique finished after {} seconds'.format(end - start))
        weights = np.ones((Q.shape[0], )) / Q.shape[0]
        _, C, L = self.computeEllipsoid(Q, weights=weights, chol=True, expand=False)
        iter = 0
        # needed_input = [self.P.shape, self.lower_d, L, C.flatten()]
        needed_input = [self.P.shape, self.lower_d+1, np.linalg.cholesky(self.computeLambda(Q, weights)),
                        np.zeros((self.lower_d + 1,))]
        constraints = [lambda x: self.oracle.checkIfInside(x) <= 1]
        i = weights.shape[0]
        old_weights = copy.deepcopy(weights)
        start1 = timer()
        while iter <= np.ceil(self.lower_d / self.eps):
            # print(weights)
            start = timer()
            farthest_point_from_ellipsoid = self.optimal_solver.solveOptimizationProblem(direction_type_cost=False,
                                                                          needed_input=needed_input,
                                                                          constraints=constraints)
            end = timer()
            if self.verbose:
                print('Finding the farthest point from the ellipsoid took {}'.format(end - start))
            start = timer()
            j_minus = np.argmin(np.linalg.norm(L.dot(Q.T - C), axis=0) ** 2)
            end = timer()
            if self.verbose:
                print('Find the minimal index took {} seconds'.format(end - start))
            entered = False
            try:
                start = timer()
                idx_inside = np.where(np.equal(Q, farthest_point_from_ellipsoid).all(1))[0][0]
                j_plus = min(i, idx_inside)
                entered = True
                end = timer()
                if self.verbose:
                    print('Checking if the point is inside already took {} seconds'.format(end - start))
            except IndexError:
                j_plus = i
                i += 1


            start = timer()
            # changed here due to a bug!
            farthest_point_from_ellipsoid_prime = np.hstack((farthest_point_from_ellipsoid, 1))
            Lambda = self.computeLambda(Q, weights)
            dist_func = lambda x: x.dot(Lambda).dot(x)

            # redefine distance
            if False:
                dist_plus = self.d * np.linalg.norm(L.dot(farthest_point_from_ellipsoid - C.flatten())) ** 2 + 1
                dist_minus = self.d * np.linalg.norm(L.dot(Q[j_minus] - C.flatten())) ** 2
            else:
                dist_plus = dist_func(farthest_point_from_ellipsoid_prime)
                # j_minus = \
                #     np.argmin(np.linalg.norm(np.linalg.cholesky(Lambda).dot(np.hstack((Q, np.ones((Q.shape[0],1)))).T),
                #                              axis=0) ** 2)
                dists = np.array([dist_func(np.hstack((x, 1))) for (i,x) in enumerate(Q) if weights[i] > 0])
                valid_idxs = np.where(weights > 0)[0]
                j_minus, dist_minus = valid_idxs[np.argmin(dists)], np.min(dists)
                if dist_plus < np.max(dists) and j_plus < weights.shape[0]:
                    dist_plus = np.max(dists)
                    j_plus = valid_idxs[np.argmax(dists)]
                    entered = True


            eps_plus, eps_minus = dist_plus / (self.lower_d + 1) -1, 1 - dist_minus / (self.lower_d + 1)
            beta_plus, beta_minus = (dist_plus - self.lower_d - 1) / (self.lower_d + 1) / (dist_plus - 1), \
                                        min((self.lower_d + 1 - dist_minus) / (dist_minus-1) / (self.lower_d + 1),
                                            weights[j_minus] / (1 - weights[j_minus]))
            end = timer()
            if self.verbose:
                print('Updating the parameters take {} seconds'.format(end - start))
            if eps_plus >= eps_minus:
                weights *= (1 - beta_plus)

                if entered and j_plus < weights.shape[0]:
                    weights[j_plus] += beta_plus
                else:
                    start = timer()
                    # weights *= 1-beta_plus
                    weights = np.hstack((weights, beta_plus))
                    Q = np.vstack((Q, farthest_point_from_ellipsoid[np.newaxis, :]))
                    end = timer()
                    if self.verbose:
                        print('Adding new points and weights take {} seconds'.format(end - start))
            else:
                weights *= (1 + beta_minus)
                weights[j_minus] -= beta_minus

            if np.any(weights < 0) and self.verbose:
                print("HALT!!!!")

            # if old_weights.shape[0] == weights.shape[0]:
            #     if np.linalg.norm(old_weights - weights) <= self.tol:
            #         if self.verbose:
            #             print("Tolerance Reached!")
            #         break
            if self.stoppingCriterion(Q, weights):
                break
            old_weights = copy.deepcopy(weights)

            start = timer()
            _, C, L = self.computeEllipsoid(Q, weights=weights, chol=True, expand=False)
            end = timer()
            if self.verbose:
                print('Computing the ellipsoid took {} seconds'.format(end - start))
            needed_input[-2] = np.linalg.cholesky(self.computeLambda(Q, weights))  # L
            needed_input[-1] = np.zeros((C.flatten().shape[0] + 1,))  #C.flatten()
            iter += 1
        end = timer()
        if self.verbose:
            print('Updating the ellipsoid took {} seconds'.format(end-start1))
            print('The method finished after {} iterations'.format(iter))
        return self.computeEllipsoid(Q, weights=weights, chol=True, flattened=True, expand=True), self.lower_dimensional_body, \
               self.dims_to_keep, self.irrelevant_dims, ([self.oracle.bounding_box[x][0] for x in self.irrelevant_dims] if self.lower_dimensional_body else [])

    def computeEllipsoid(self, S, weights=None, chol=False, flattened=False, expand=False):
        if weights is None:
            weights = np.ones((S.shape[0], )) * 1 / S.shape[0]

        if weights.ndim == 1:  # make sure that the weights are not flattened
            weights = np.expand_dims(weights, 1)

        c = np.dot(S.T, weights)  # attain the center of the MVEE

        Q = S[np.where(weights.flatten() > 0.0)[0], :]  # get all the points with positive weights
        weights2 = weights[np.where(weights.flatten() > 0.0)[0], :]  # get all the positive weights

        # compute a p.s.d matrix which will represent the ellipsoid
        ellipsoid2 = 1.0 / ((1 + self.eps) * S.shape[1]) * np.linalg.inv(np.dot(np.multiply(Q, weights2).T, Q) -
                                                      np.multiply.outer(c.T.ravel(), c.T.ravel()))
        if expand and False:
            term = (((1 + (self.d+1)/self.d*self.eps)*self.d) ** 2)
        else:
            term = 1
        if chol:
            return ellipsoid2 * term, c if not flattened else c.flatten(), np.linalg.cholesky(ellipsoid2)\
                   / term

        return ellipsoid2, c if not flattened else c.flatten()


class MemOracle(object):
    def __init__(self, cost_func, dtype="int", cost_func_cp=None, flattened_data=None, dimensions=None,
                 matrix_of_vecs=False, upper_bound=None, bounding_box_sum_step_limit=4, verbose=False):
        self.cost_func = cost_func
        self.threshold = 1
        self.bounding_box = None
        # self.pool = Pool()
        self.dtype = dtype
        self.cost_func_cp = cost_func_cp
        self.coordinates = None
        self.flattened_data = flattened_data
        self.dimensions = dimensions
        self.plain_oracle = False if flattened_data is None else True
        self.matrix_of_vecs = matrix_of_vecs
        self.upper_bound=upper_bound
        self.bounding_box_sum_step_limit = bounding_box_sum_step_limit
        self.verbose = verbose
        if self.plain_oracle:
            self.preprocessInCaseOfGridData()

    def __getstate__(self):
        # this method is called when you are
        # going to pickle the class, to know what to pickle
        state = self.__dict__.copy()
        return state

    def __reduce__(self):
        # we return a tuple of class_name to call,
        # and optional parameters to pass when re-creating
        return (self.__class__, (self.cost_func, self.dtype, self.cost_func_cp, self.flattened_data, self.dimensions,
                                 self.matrix_of_vecs, self.upper_bound, self.bounding_box_sum_step_limit, self.verbose))

    def __setstate__(self, state):
        self.__dict__.update(state)

    def checkIfInsideCPVersion(self, p):
        return cp.matmul(self.cost_func_cp[:,:-1], p) + self.cost_func_cp[:, -1]

    def updateAdaptiveThreshold(self, threshold):
        self.threshold = threshold

    def checkIfInsidePixelStyle(self, p):
        return cp.matmul(p, self.flattened_data)

    def checkIfInsidePixelStyleNumpyVer(self, p):
        return self.flattened_data.dot(self.convertPhysicalIndex(p.astype("int"), is_bounding_box=True))

    def obtainBoundBox(self, point):
        self.bounding_box = self.attain_bounding_box(point)
        # self.createNewCostFunction()

    def checkIfInside(self, point):
        """
        here we check whether a point is a point of interesting object, and whether the same point is in some bounded
        focused world.
        :param point:
        :return:
        """
        term = True
        if self.bounding_box is not None:
            term = np.logical_and(self.bounding_box[:,0] <= point, self.bounding_box[:,1] >= point).all()

        term = np.logical_and(term, np.all(np.logical_and(np.less_equal(point, self.upper_bound),
                                                   np.greater_equal(point, np.zeros(point.shape)))))

        if self.matrix_of_vecs is False:
            return True if term and self.cost_func(point) <= self.threshold else False
        else:
            return True if (term and self.cost_func(point)) else False

    def checkIfInsideCPVersion2(self, point):
        """
        here we check whether a point is a point of interesting object, and whether the same point is in some bounded
        focused world.
        :param point:
        :return:
        """
        global upper_bound, lower_bound

        term = True
        if self.bounding_box is not None:
            term = np.logical_and(self.bounding_box[:,0] <= point, self.bounding_box[:,1] >= point).all()

        term = np.logical_and(term, np.all(np.logical_and(np.less_equal(point, upper_bound),
                                                   np.greater_equal(point, lower_bound))))

        return True if term and self.cost_func(point) <= self.threshold else False

    def attain_bounding_box(self, point):
        """
        We need to apply exponential jumping in order to attain a bounding box. This is done using elements
        from coordinate descent and exponential jumping (similar in nature to binary search)
        :param point:
        :return:
        """
        start = timer()
        steps = np.empty((point.shape[0]))
        box_coordinates = np.empty((point.shape[0], 2), dtype=self.dtype)
        boxes = []
        for i in range(point.shape[0]):
            # go ever every dimension
            for increase in [False, True]:
                updated_point, _, steps[i] = self.binary_search(point, i, increase=increase)  # from this point, we
                                                                                              # start performing
                                                                                              # like coordinate descent
                j = 0
                done_here = 0
                while done_here <= (point.shape[0] - 1):
                    if j >= point.shape[0]:
                        j = np.mod(j, point.shape[0])
                    if j == i:
                        j += 1
                        continue
                    objective_func = (lambda x: self.binary_search(updated_point, j, increase=x,
                                                                   bounded_number_of_steps=int(np.ceil(
                                                                       steps[i]))))
                    # with self.pool:
                    candidates = np.array([np.nan, np.nan])
                    updated_candidate_points = np.empty((2, point.shape[0]), dtype=self.dtype)
                    results = [objective_func(x) for x in [False, True]]
                    for check in range(2):
                        any_change,any_change2 = False, False
                        if results[check][1]:
                            updated_candidate_points[check, :], any_change2, _ = self.binary_search(results[check][0],
                                                                                                   i,
                                                                                                   increase=increase)
                        if any_change2:
                            candidates[check] = updated_candidate_points[check, i]
                        elif any_change:
                            candidates[check] = results[check][0,i]
                    idxs = np.argwhere(np.logical_not(np.isnan(candidates)))
                    if idxs.size != 0:
                        max_idx = np.argmax(updated_candidate_points[idxs, i] * (-1) ** (increase + 1))
                        updated_point = updated_candidate_points[idxs[max_idx], :].flatten()
                        done_here = 0
                    else:
                        updated_point_temp = copy.deepcopy(updated_point)
                        updated_point_temp[i] += (-1) ** (int(increase) + 1)
                        updated_point_temp, found = self.getInsideBody(updated_point_temp, coordinate=i,
                                                                       increase=increase)
                        if not found:
                            done_here += 1
                        else:
                            done_here = 0
                            updated_point = updated_point_temp
                    j += 1
                boxes.append(updated_point)
                box_coordinates[i, 2 ** (int(increase)) - 1] = updated_point[i]
        end = timer()
        if self.verbose:
            print('The box bounding technique finished after {} seconds'.format(end - start))
        max_lim, min_lim = np.max(np.array(boxes), axis=0), np.min(np.array(boxes), axis=0)
        box_coordinates = np.hstack((min_lim[:,np.newaxis], max_lim[:,np.newaxis]))
        return box_coordinates

    def preprocessInCaseOfGridData(self, ranges=None, add_rows_one=False):
        if ranges is None:
            coor = [np.arange(x) for x in self.dimensions]
        else:
            coor = [np.arange(x[0], x[1]+1) for x in ranges if x[0] != x[1]]
        X = np.meshgrid(*coor, sparse=False)
        X = [x.flatten()[:, np.newaxis] for x in X]

        self.coordinates = self.sortByColumns(np.hstack(X), add_rows_one=add_rows_one).T

    def convertPhysicalIndex(self, phy_idx, return_vec=True, is_bounding_box=False):
        if not is_bounding_box:
            idx = np.sum([x * np.prod(self.dimensions[-i:])  if i > 0 else x
                      for i,x in enumerate(reversed(phy_idx))])
        else:
            if np.all(np.logical_and(phy_idx >= self.bounding_box[:,0], phy_idx <= self.bounding_box[:,1])):
                dimensions = np.array([x[1] - x[0] + 1 for x in self.bounding_box])
                idx = np.sum([x * np.prod(dimensions[-i:]) if i > 0 else x for i, x in enumerate(reversed(phy_idx - self.bounding_box[:,0]))])
            else:
                return np.ones(self.flattened_data.shape).flatten()
        if return_vec:
            v = np.eye(self.flattened_data.shape[0], 1).flatten()
            v = np.roll(v, idx)
            return v
        return idx

    @staticmethod
    def cartesian_product(*arrays):
        la = len(arrays)
        dtype = np.result_type(*arrays)
        arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[..., i] = a
        return arr.reshape(-1, la)

    def cartesian_product_transpose(*arrays):
        broadcastable = np.ix_(*arrays)
        broadcasted = np.broadcast_arrays(*broadcastable)
        rows, cols = np.prod(broadcasted[0].shape), len(broadcasted)
        dtype = np.result_type(*arrays)

        out = np.empty(rows * cols, dtype=dtype)
        start, end = 0, rows
        for a in broadcasted:
            out[start:end] = a.reshape(-1)
            start, end = end, end + rows
        return out.reshape(cols, rows).T

    def getInsideBody(self, point, coordinate, increase=False):
        global range_of_valid_steps
        updated_point = copy.deepcopy(point)
        step = 0
        found = False
        range_of_valid_coordinates = np.array([i for i in range(point.shape[0]) if i != coordinate])

        if point.shape[0] > 2:
            all_possible_changes = MemOracle.cartesian_product_transpose(*([range_of_valid_steps] * range_of_valid_coordinates.shape[0]))
        else:
            all_possible_changes = range_of_valid_steps

        for row in all_possible_changes:
            if np.all(row == 0.0):
                continue
            else:
                updated_point[range_of_valid_coordinates] += row
                if self.checkIfInside(updated_point):
                    updated_point, any_change, _ = self.binary_search(updated_point, coordinate, increase)
                    if updated_point[coordinate] * (-1) ** (int(increase) + 1) \
                                >= point[coordinate] * (-1) ** (int(increase) + 1):
                        found = True
                        break
                else:
                    updated_point[range_of_valid_coordinates] -= row
        return None if not found else updated_point, found

    def binary_search(self, point, coordinate, increase=False, bounded_number_of_steps=np.inf):
        # is_inside_orig = self.checkIfInside(point)
        step_size = 1
        any_change = False
        steps = 0

        current_point = copy.deepcopy(point)
        keep_searching = True
        old_point = copy.deepcopy(current_point)

        while keep_searching:
            current_point[coordinate] += int(2 ** (step_size - 1) * (-1) ** (int(increase) + 1))
            if (1 - 2 ** step_size) / (1 - 2) >= self.bounding_box_sum_step_limit:
                step_size = 1
                current_point = copy.deepcopy(old_point)
            elif self.checkIfInside(current_point):
                step_size += 1
                old_point = copy.deepcopy(current_point)
                any_change = True
            else:
                current_point = copy.deepcopy(old_point)
                if step_size == 1:
                    break
                else:
                    step_size = 1
            steps += 1
            if steps >= bounded_number_of_steps:
                break
        return current_point.astype("int"), any_change, steps

    def sortByColumns(self, Z, add_rows_one=False):
        if len(Z.shape) > 1:
            for i in range(len(Z.shape)-1, -1, -1):
                if (i == (len(Z.shape)-1)) and Z.shape[i] > 1:
                    Z = Z[Z[:, i].argsort()]
                elif (i < (len(Z.shape)-1)) and (Z.shape[i] > 1):
                    Z = Z[Z[:, i].argsort(kind='mergesort')]

        return Z if not add_rows_one else np.hstack((Z, np.ones((Z.shape[0], 1))))

    def focusOracleOnPoint(self, flattened_data, add_rows_one=False):
        self.flattened_data = flattened_data
        self.preprocessInCaseOfGridData(ranges=self.bounding_box, add_rows_one=add_rows_one)




def main():
    # load test data
    start = timer()
    global lower_bound, upper_bound
    data = np.load(r'Tests/Test 2/convex_shape.npz')
    P, hull_hyper = data['P'], data['oracle_hull']

    # create the membership cost function
    cost_func = (lambda x: P[tuple(x)])


    # initial object point
    p = np.array([4, 58, 196])

    lower_bound = np.zeros((P.ndim, ))
    upper_bound = np.array(P.shape)-1

    # create an instance of the convex shape finder
    object_finder = LinearProgrammingInTheDark(P=P, cost_func=cost_func, point=p, d=P.ndim, epsilon=0.01, hull_hyper=hull_hyper)

    # find the ellipsoid
    object_finder.computeAMVEE()
    end = timer()
    print('The program finished after {} seconds'.format(end - start))





# def computeFarthestPointsAlongdirection(direction):
#     global P, oracle
#     x = cp.Variable(direction.shape[0], )
#     loss = cp.matmul(x, direction)
#     constraints = [oracle.checkIfInside(x) <= 1]
#     prob = cp.Problem(cp.Minimize(loss), constraints)
#     prob.solve()
#     return x.value()



# def applyPolynomialGrid(d):
#     global grid
#     pass


if __name__ == '__main__':
    main()