import copy
import sys
sys.path.append('SetsClustering')
from multiprocessing import Process ,Manager
import numpy as np
import LinearProgrammingInTheDarkClassVersion as LPD
from multiprocessing import Pool
from jgrapht.algorithms.shortestpaths import johnson_allpairs
import jgrapht
from SetsClustering import Utils, PointSet, KMeansAlg
from SetsClustering import KMeansForSetsSensitivityBounder as SensBounder
from SetsClustering import Coreset as CS
from scipy.spatial.distance import cdist
import seaborn as sns
from copy import deepcopy
import itertools
from scipy.ndimage import convolve
from timeit import default_timer as timer
from tqdm import tqdm
import dill
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from scipy.linalg import null_space
import scipy.ndimage as ndi
from scipy.spatial import ConvexHull
import argparse, os, pickle
from scipy.io import netcdf
POWER = 4
FORCE_NEIGHBORING = 20
import psutil
CPUS = psutil.cpu_count()
# import multiprocessing
# # from pathos.multiprocessing import ProcessingPool as Pool
# # from sklearn.externals.joblib import Parallel, delayed
# from multiprocessing import Process

parser = argparse.ArgumentParser(description='Initial Location Generator')
parser.add_argument('-d', type=str, default=None, help='Directory containing all maps')
parser.add_argument('-pp', default=False, action='store_true', help='preprocess map')
parser.add_argument('-ft', default='.nc', type=str, help='Type of map file')
parser.add_argument('-nf', default=1, type=int, help='Number of files describing a map of velocities')
parser.add_argument('-eps_g', default=None, type=float, help=r'resolution of the \varepsilon-grid')
parser.add_argument('-eps_b', default=0.08, type=float,
                    help=r'epsilon approximation for each of the patches of the currents')
parser.add_argument('-k', default=10, type=int, help='Desired number of drifters')
parser.add_argument('-bs', default=2, type=int, help='size of the blob prior to the clustering phase')
parser.add_argument('-coreset_sample_size', default=1000, type=int,
                    help='The size of the coreset for the clustering phase')
parser.add_argument('-time', default=False, action='store_true', help='Apply our system over time')
parser.add_argument('-tol', default=0.2, type=float, help='Tolerance for minimum volume ellipsoid')
parser.add_argument('-resume', default=False, action='store_true', help='In case of code being killed, you can resume from last map')
parser.add_argument('-show', default=False, action='store_true', help='Show only our segementation and clustering. Must have preporcessed these data before')


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    NORMAL = '\033[0m'

plt.rcParams.update({'font.size': 16})

manager = Manager()




def removeInclusionsJob(lst, ids, path_str):
    global resdict
    for i in range(len(lst)):
        resdict[ids[i]] = True
        if lst[i] in path_str:
            resdict[ids[i]] = False


def removeInclusions(unified_paths, file_path='', file_prefix=''):
    global manager
    global resdict
    global A

    

    unified_paths_strings = [str(x[0]).strip('[]') for x in unified_paths]
    unified_paths_strings.sort(key=(lambda x: len(x.split(','))))
    lst = [list(grp) for i, grp in itertools.groupby(unified_paths_strings, key=(lambda x: len(x.split(','))))]
    sizes = np.cumsum([len(x) for x in lst])
    unique_ids = [list(range(sizes[i-1], sizes[i]) if i > 0 else range(sizes[i])) for i in range(len(sizes))]
    
    if len(unified_paths_strings) > 10000:
        with Manager() as manager:
            proc_list = []
            resdict = manager.dict()
            for i, item in enumerate(lst):
                if i != (len(lst) - 1):
                    proc_list.append(
                        Process(target=removeInclusionsJob,
                            args=(item, unique_ids[i], '\n'.join(unified_paths_strings[sizes[i]:])))
                    )
                    proc_list[-1].start()
            for proc in proc_list:
                proc.join()        
            mask = [x[1] for x in resdict.items()]
    else:
        resdict = dict()
        for i, item in enumerate(lst):
            if i != (len(lst) - 1):
                removeInclusionsJob(item, unique_ids[i], '\n'.join(unified_paths_strings[sizes[i]:]))
        mask = [x[1] for x in resdict.items()]

    mask.extend([True for _ in range(len(lst[-1]))])
    np.save('{}mask_unified_paths_{}.npy'.format(file_path, file_prefix), mask)
    return [[int(y) for y in x.split(', ')] for x in list(itertools.compress(unified_paths_strings, mask))]


def removeDuplicates(list_1):
    list2 = list(set(list_1))
    list2.sort(key=list_1.index)
    return list2

def makedir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError as error:
        print(error)


def saveVels(data, file_path, smoothed=True):
    if smoothed:
        file_path += 'Smoothed_Vel/'
    else:
        file_path += 'Original_Vel/'
    makedir(file_path)
    temp = np.tile(data[:, :, 0][:, :, np.newaxis], 10)
    temp.dump(file_path + 'matrix_vel_x.dat')
    temp = np.tile(data[:, :, 1][:, :, np.newaxis], 10)
    temp.dump(file_path + 'matrix_vel_y.dat')


def readNetCDFFile(file_path, over_time):
    file2read = netcdf.NetCDFFile(file_path, 'r')
    U = file2read.variables['u'].data  # velocity in x-axis
    V = file2read.variables['v'].data  # velocity in y-axis
    mask = np.logical_and(np.abs(U) <= 1e3, np.abs(V) <= 1e3)
    V = np.multiply(V, mask)
    U = np.multiply(U, mask)
    if not over_time:
        U = U[0, :, :, :]
        V = V[0, :, :, :]
    return U,V


def innerFunction(current_possible_combs, unique_keys):
    global resdict
    for i, element in enumerate(current_possible_combs):
        resdict[unique_keys[i]] = (removeDuplicates(element[0][0] + element[1][0]), element[0][1] + element[1][1])


def getAllPossiblePaths(list1, list2):
    global CPUS
    global manager
    global resdict

    if len(list1) * len(list2) > 10000:
        manager = Manager()
        resdict = manager.dict()
        all_possible_combs = np.array_split(list(itertools.product(list1, list2)), CPUS)
        unique_ids = np.array_split(np.arange(sum([x.size for x in all_possible_combs])), CPUS)
        proc_list = []
        for i, item in enumerate(all_possible_combs):
            proc_list.append(
                Process(target=innerFunction, args=(item, unique_ids[i]))
            )
            proc_list[-1].start()
        for proc in proc_list:
            proc.join()

        temp = list(resdict.values())
    else:
        temp = []
        for element in itertools.product(list1, list2):
            temp.append((removeDuplicates(element[0][0] + element[1][0]), element[0][1] + element[1][1]))

    return temp

class CurrentEstimation(object):
    def __init__(self, grid, k=10, epsilon_grid=0.06, tolerance=0.001, epsilon_body=2, is_grid=True, is_data_vectorized=True,
                 blob_size=3, sens_file_name='sens.npz', coreset_sample_size = int(1e3), save_mode=True,
                 matrix_of_velocities=True, save_path='', file_prefix='', show=False, verbose=False):
        self.grid = grid
        self.is_grid=is_grid
        self.d = (self.grid.ndim - 1) if matrix_of_velocities else self.grid.ndim
        self.epsilon_grid = epsilon_grid
        self.epsilon_body = epsilon_body
        self.tolerance = tolerance
        self.g = jgrapht.create_graph(directed=True)
        self.cost_func = (lambda x: self.grid[tuple(x.astype("int") if is_grid else x)])  # create a simple membership cost function
        self.iocsAlg = None
        self.segments = []
        self.eps_star = None
        self.bodies = []
        self.full_bodies = []
        self.is_data_vectorized = is_data_vectorized
        self.k = k
        self.blob_size = blob_size
        self.coreset_sample_size = coreset_sample_size
        self.save_mode = save_mode
        self.binary_grid = None
        self.matrix_of_velocities = matrix_of_velocities
        self.sens_file_name = sens_file_name
        self.ellipsoids = []
        self.convex_hulls = []
        self.verbose = verbose
        self.save_path = save_path
        self.file_prefix = file_prefix
        self.show = show

    def polynomialGridSearchParallelizedVersion(self):
        with Pool() as pool:
            pass

    def checkIfContained(self, point):
        for i,body in enumerate((self.full_bodies if self.epsilon_body == 0 else self.bodies)):
            if body.ndim > 1:
                temp_in_body = np.equal(body, point).all(1).any()
                temp_in_CH = False
                if self.convex_hulls[i] is not None:
                    temp_in_CH = np.all(self.convex_hulls[i][:,:-1].dot(point) <= -self.convex_hulls[i][:,-1])
                if temp_in_body or temp_in_CH:
                    return True
            else:
                if np.linalg.norm(body - point) == 0:
                    return True
        return False

    def IOCS(self, p):
        cost_func = lambda x: 0.85 <= np.dot(np.nan_to_num(self.grid[tuple(p)]/np.linalg.norm(self.grid[tuple(p)])),
                                            np.nan_to_num(self.grid[tuple(x)]/np.linalg.norm(self.grid[tuple(x)]))) \
                              <= 1 and  0.5 <= np.linalg.norm(self.grid[tuple(p)])/np.linalg.norm(self.grid[tuple(x)]) <= 2
        self.iocsAlg = LPD.LinearProgrammingInTheDark(P=self.grid,cost_func=cost_func, point=p,
                                                      d=self.d, epsilon=self.tolerance, hull_hyper=None,
                                                      matrix_of_vecs=True)
        if self.iocsAlg.lower_d <= 1:
            if self.iocsAlg.lower_d == 0:
                self.bodies.append(p)
                self.full_bodies.append(p)
                self.ellipsoids.append(None)
                self.convex_hulls.append(None)
            else:
                idxs = np.where(self.iocsAlg.oracle.flattened_data == 1)[0]
                Z = np.empty((idxs.shape[0], p.shape[0]))
                Z[:, self.iocsAlg.irrelevant_dims] = p[self.iocsAlg.irrelevant_dims]
                Z[:, self.iocsAlg.dims_to_keep[0]] = \
                    np.arange(*(self.iocsAlg.oracle.bounding_box[self.iocsAlg.dims_to_keep].flatten() +
                                np.array([0, 1])).tolist())[idxs]
                self.bodies.append(Z)
                self.full_bodies.append(Z)
                self.ellipsoids.append(None)
                self.convex_hulls.append(None)
        elif self.iocsAlg.get_all_points:
            idxs = np.where(self.iocsAlg.oracle.flattened_data == 1)[0]
            Z = self.iocsAlg.oracle.coordinates[:-1, idxs].T
            self.bodies.append(Z)
            self.full_bodies.append(Z)
            self.ellipsoids.append(None)
            self.convex_hulls.append(None)
        else:
            self.ellipsoids.append(self.iocsAlg.computeAMVEE() + (p, ))
            if self.epsilon_body > 0:
                s = timer()
                self.approximateBody(self.ellipsoids[-1][0][-1], self.ellipsoids[-1][0][-2],
                                     idx_dims_retrieve=self.ellipsoids[-1][-3], dims_value=self.ellipsoids[-1][-1],
                                     rest_dims=self.ellipsoids[-1][-2])
            else:
                self.attainWholeBody(self.ellipsoids[-1][0][-1], self.ellipsoids[-1][0][-2],
                                     idx_dims_retrieve=self.ellipsoids[-1][-3], dims_value=self.ellipsoids[-1][-1],
                                     rest_dims=self.ellipsoids[-1][-2])

    def polynomialGridSearch(self):
        dims = list(self.grid.shape[:-1] if self.matrix_of_velocities else self.grid.shape)

        for i in range(len(dims)):
            dims[i] = np.arange(0, dims[i], int(np.round(dims[i] * self.epsilon_grid)))

        try:
            X = np.array(np.meshgrid(*dims)).T.reshape(-1, len(dims))
            return X
        except MemoryError:
            raise MemoryError("Cant handle this much data! Lower your epsilon or simply run the parallelized version")

    @staticmethod
    def semiBinarizeGrid(grid, kernel_size=None):
        # Apply Mean-Filter
        kernel = np.ones(tuple([grid.ndim if kernel_size is None else kernel_size for i in range(grid.ndim)]),
                         np.float32) / (kernel_size ** grid.ndim if kernel_size is not None else grid.ndim ** grid.ndim)
        return convolve(grid, kernel, mode='constant', cval=0)

    def generateEpsilonStar(self, degree=None):
        if degree is None:
            degree = self.epsilon_body
        Z = np.arange(0, 2*np.pi, degree * np.pi)
        V = np.array(np.meshgrid(*[Z for i in range(self.d)])).T.reshape(-1, self.d)
        V = np.divide(V, np.linalg.norm(V, axis=1)[:, np.newaxis], out=np.zeros_like(V), where=(V != 0))
        V = np.unique(np.around(np.unique(V[1:], axis=0), self.d+1), axis=0)
        return V


    @staticmethod
    def run_dill_encoded(payload):
        fun, args = dill.loads(payload)
        return fun(*args)

    @staticmethod
    def apply_async(pool, fun, args):
        payload = dill.dumps((fun, args))
        return pool.apply_async(CurrentEstimation.run_dill_encoded, (payload,))

    def attainWholeBody(self, E, c, idx_dims_retrieve=None, dims_value=None, rest_dims=None):
        if self.iocsAlg.oracle.checkIfInsidePixelStyleNumpyVer(np.round(c)) > 1.0:
            raise ValueError('Something is wrong with the ellipsoid!')
        bounding_box = self.iocsAlg.oracle.bounding_box
        indices = np.vstack(map(np.ravel, np.meshgrid(*[np.arange(bounding_box[x, 0], bounding_box[x, 1]+1)
                                for x in range(bounding_box.shape[0])]))).T
        body = []
        temp = 0
        for idx in indices:
            if self.iocsAlg.oracle.checkIfInsidePixelStyleNumpyVer(idx) == 1 and np.linalg.norm(E.dot(idx - c)) <= 1 \
                    and not self.checkIfContained(idx):
                temp += 1
                if np.linalg.norm(self.grid[tuple(idx)]) > 1e-10:
                    body.append(idx)
        if len(body) > 0:
            self.full_bodies.append(np.vstack(body))

    def approximateBody(self, E, c, idx_dims_retrieve=None, dims_value=None, rest_dims=None):
        bounding_box = self.iocsAlg.oracle.bounding_box
        indices_of_lengths = np.argsort([x[0] - x[1] for x in bounding_box])
        coeffs = np.zeros((indices_of_lengths.shape[0],))
        for i in range(coeffs.shape[0]):
            if i == (coeffs.shape[0] - 1):
                coeffs[indices_of_lengths[i]] = 1
            else:
                coeffs[indices_of_lengths[i]] = max(((bounding_box[indices_of_lengths[i],1] -
                                                  bounding_box[indices_of_lengths[i],0]) * self.epsilon_body),1)
        V = np.vstack(map(np.ravel, np.meshgrid(*[np.arange(start=x[0], stop=x[1],
                                                            step=coeffs[j]) for (j,x) in enumerate(bounding_box)]))).T
        V = np.unique(V.astype("int"), axis=0)
        body = []
        for v in V:
            if (self.iocsAlg.oracle.checkIfInsidePixelStyleNumpyVer(v) <= 1.0) and\
                    (np.linalg.norm(E.dot(v - c)) <= np.sqrt(1 + (1 + self.iocsAlg.eps) * E.shape[0])) and\
                    (np.linalg.norm(self.grid[tuple(v)]) > 0) and (not self.checkIfContained(v)):
                body.append(v)
        if len(body) > 0:
            self.bodies.append(np.vstack(body))
        if len(body) > (self.d + 1):
            try:
                self.convex_hulls.append(ConvexHull(self.bodies[-1]).equations)
            except:
                self.convex_hulls.append(None)
        else:
            self.convex_hulls.append(None)

    def createBlobs(self, body):
        if body.ndim == 1:
            return [PointSet.PointSet(body[np.newaxis,:])]
        elif body.shape[0] < self.blob_size:
            return [PointSet.PointSet(body)]
        else:
            blob = []
            for x_val in np.unique(body[:,0]):
                idxs = np.where(body[:, 0] == x_val)[0]
                if body[idxs].shape[0] < self.blob_size:
                    blob.extend([PointSet.PointSet(body[idxs])])
                else:
                    splitted_array = np.array_split(body[idxs], int(body[idxs].shape[0] / self.blob_size))
                    blob.extend([PointSet.PointSet(x) for x in splitted_array])
            return blob

    def clusteringAssignment(self, set_P, Q):
        assignments_per_point = []
        assignments_per_blob = []
        for P in set_P:
            dists = cdist(P.P, Q)
            cols_idxs = np.argmin(dists, axis=1)
            min_idx = np.argmin(np.min(dists, axis=1))
            assignments_per_point.extend([cols_idxs[min_idx] for p in P.P])
            assignments_per_blob.append(cols_idxs[min_idx])

        return assignments_per_point, assignments_per_blob

    def clusterWaves(self, continue_from=0,return_full_bodies=True):
        P = []
        blobs = []
        if self.epsilon_body != 0:
            for body in self.bodies:
                P = []
                # need to make a way to make sure that there is a trade-off between the first 3 entries and last two
                if body.ndim == 1:
                    body = body[np.newaxis, :]
                for point in body:
                    a = self.grid[tuple(point.astype("int"))]
                    b = np.linalg.norm(a)
                    P.append(
                        np.hstack((point*FORCE_NEIGHBORING, np.divide(a,b, out=np.zeros_like(a), where=b!=0)
                                   * np.linalg.norm(point))))
                blobs.extend(self.createBlobs(np.array(deepcopy(P))))
        else:
            for body in self.full_bodies:
                # need to make a way to make sure that there is a trade-off between the first 3 entries and last two
                P = []
                if body.ndim == 1:
                    body = body[np.newaxis, :]
                for point in body:
                    P.append(
                        np.hstack((point*FORCE_NEIGHBORING, self.grid[tuple(point.astype("int"))] /
                                   np.linalg.norm(self.grid[tuple(point.astype("int"))]) * np.linalg.norm(point))))
                blobs.extend(self.createBlobs(np.array(deepcopy(P))))
        set_P_indiced = [(P, idx) for (idx, P) in enumerate(blobs)]  # taking the full!
        if continue_from > 0 or self.show:
            sensitivity = np.load(self.save_path + self.file_prefix + self.sens_file_name)['s']
            print("Loaded sensitivity for sets clustering!")
        else:
            k_means_sens_bounder = SensBounder.KMeansForSetsSensitivityBounder(set_P_indiced, self.k, None, None)
            sensitivity = k_means_sens_bounder.boundSensitivity()
            if self.save_mode:
                np.savez(self.save_path + self.file_prefix + self.sens_file_name, s=sensitivity)
                print('Sum of sensitivity is {}'.format(np.sum(sensitivity)))
                print("Saved sensitivity for sets clustering!")
        if continue_from <= 1 and not self.show:
            k_means_alg = KMeansAlg.KMeansAlg(blobs[0].d, self.k)
            coreset = CS.Coreset()
            C = coreset.computeCoreset(set_P_indiced, sensitivity, int(self.coreset_sample_size))
            _, Q, _ = k_means_alg.computeKmeans(C[0], False)
            np.savez('{}Optimal_clustering_{}.npz'.format(self.save_path, self.file_prefix), Q=Q)
        else:
            Q = np.load('{}Optimal_clustering_{}.npz'.format(self.save_path,self.file_prefix))['Q']
            print("Loaded optimal clustering of coreset")

        assignments_per_point, assignments_per_blob = self.clusteringAssignment(blobs, Q)
        return np.array(blobs), np.array(assignments_per_blob), assignments_per_point

    def addConnections(self, pairs, g_all, i, j, list_of_vertices, shift_idx_root, shift_idx_leaf, is_leaf=None,
                       enable_weights=False, connections=[]):
        dists = np.linalg.norm(self.clustered_bodies[i][pairs[:,0]] - self.clustered_bodies[j][pairs[:,1]], axis=1)
        pairs_of_interest = pairs[np.where(dists <= 2)[0]]
        if len(pairs_of_interest) != 0:
            if enable_weights:
                for pair in pairs_of_interest:
                    root_of_path_of_interest = self.clustered_bodies[i][pair[0]]
                    leaf_of_path_of_interest = self.clustered_bodies[j][pair[1]]
                    direction = root_of_path_of_interest - leaf_of_path_of_interest
                    direction = direction / np.linalg.norm(direction)
                    target_direction = self.grid[tuple(root_of_path_of_interest.astype("int"))]
                    alpha = np.dot(direction, target_direction/np.linalg.norm(target_direction))
                    if alpha > 0.7:
                        try:
                            g_all.add_edge(int(pair[0] + shift_idx_root), int(pair[1] + shift_idx_leaf))
                            list_of_vertices = np.delete(list_of_vertices, np.where(list_of_vertices == (pair[1]+shift_idx_leaf)))
                            if is_leaf is not None:
                                is_leaf = np.delete(is_leaf, np.where(is_leaf == (pair[0] + shift_idx_root)))
                        except:
                            continue
            else:
                roots = np.unique(pairs_of_interest[:, 0])
                for root in roots:
                    try:
                        idxs_of_interest = np.where(pairs_of_interest[:, 0] == root)[0]
                        pairs_of_interest_per_root = pairs_of_interest[idxs_of_interest, :]
                        root_of_path_of_interest = self.clustered_bodies[i][root][np.newaxis, :]
                        leaf_of_path_of_interest = self.clustered_bodies[j][pairs_of_interest_per_root[:, 1]]
                        directions = leaf_of_path_of_interest - root_of_path_of_interest
                        directions = np.divide(directions,
                                               np.linalg.norm(directions, axis=1)[:, np.newaxis],
                                               out=np.zeros_like(directions),
                                               where=np.linalg.norm(directions, axis=1)[:, np.newaxis]!=0, casting="unsafe")
                        target_direction = self.grid[tuple(root_of_path_of_interest.flatten().astype("int"))]
                        alpha = np.dot(directions, target_direction / np.linalg.norm(target_direction))
                        l = np.argmax(alpha)
                        if alpha[l] >= 0.7:
                            g_all.add_edge(int(root + shift_idx_root),
                                           int(pairs_of_interest[idxs_of_interest[l]][1] + shift_idx_leaf))
                            list_of_vertices = \
                                np.delete(list_of_vertices,
                                          np.where(list_of_vertices == (pairs_of_interest[idxs_of_interest[l]][1]
                                                                        + shift_idx_leaf)))
                            if is_leaf is not None:
                                is_leaf = np.delete(is_leaf, np.where(is_leaf == (root + shift_idx_root)))
                            connections.append((i, int(root), j, int(pairs_of_interest[idxs_of_interest[l]][1])))
                    except:
                        continue
        return g_all, list_of_vertices, is_leaf, connections

    def containedInMap(self, point):
        temp = point + self.grid[tuple(point.astype("int"))]
        if np.any(temp < 0) or np.any(temp >= np.array(list(self.grid.shape[:-1]))):
            return False

        return True

    def attainDiameterOfSetOfPoints(self, P):
        return np.max(np.linalg.norm(P - P[np.argmax(np.linalg.norm(P - np.mean(P, axis=0)[np.newaxis, :],
                                                                    axis=1))][np.newaxis, :], axis=1))

    def avoidRedundantConnection(self, point, P, orig_idxs):
        norms = np.linalg.norm(P - point[np.newaxis, :], axis=1)
        idxs = np.argsort(norms)
        temp = P - point[np.newaxis, :]
        temp = np.around(np.multiply(temp[idxs], (1 / norms[idxs])[:, np.newaxis]), 2)
        _, idx2 = np.unique(temp, axis=0, return_index=True)
        return orig_idxs[idxs[idx2]]

    def generateGraph(self, is_full=True, enable_weights=False, enable_all=False):
        leaves = []
        roots = []
        all_others = []
        roots_all = np.array([])
        leaves_all = np.array([])
        idx_shift = 0
        g_all = jgrapht.create_graph(directed=True, weighted=False)
        graphs = [jgrapht.create_graph(directed=True, weighted=False) for i in range(self.k)]
        counter_bad_vertices = np.zeros((self.k, ))
        cnt = 0
        for body_idx,body in enumerate(self.clustered_bodies):
            idxs_leafs = np.arange(body.shape[0])
            idxs_roots = np.arange(body.shape[0])
            idxs_all_others = np.arange(body.shape[0])
            for i in range(idx_shift, idx_shift + body.shape[0]):
                graphs[body_idx].add_vertex(i-idx_shift)
                g_all.add_vertex(i)
            for i, point in enumerate(body):
                temp = body-point[np.newaxis, :]
                norms = np.linalg.norm(temp, axis=1)[:, np.newaxis]
                if is_full:
                    norms = norms.flatten()
                    neighbors = np.where(np.logical_and(norms.flatten() <= np.sqrt(2), norms.flatten() > 0))[0]
                    norms = norms.flatten()[neighbors][:, np.newaxis]
                    temp = temp[neighbors,:]
                else:
                    norms = norms.flatten()
                    min_dist = self.attainDiameterOfSetOfPoints(body) * self.epsilon_body
                    neighbors = np.where(np.logical_and(norms.flatten() <= min_dist, norms.flatten() > 0))[0]
                    norms = norms.flatten()[neighbors][:, np.newaxis]
                    temp = temp[neighbors, :]
                dots = np.clip(np.dot(np.multiply(temp, np.divide(1, norms, out=np.zeros_like(norms), where=norms != 0)),
                              self.grid[tuple(point)] / np.linalg.norm(self.grid[tuple(point)])), -1,1)
                vals = np.arccos(dots)
                normal = null_space((self.grid[tuple(point)] / np.linalg.norm(self.grid[tuple(point)]))[np.newaxis, :])
                vals2 = np.linalg.norm(np.dot(np.multiply(temp, np.divide(1, norms, out=np.zeros_like(norms),
                                                                          where=norms != 0)), normal), axis=1)
                try:
                    if not self.containedInMap(point):
                        counter_bad_vertices[body_idx] += 1
                        idxs_roots = np.delete(idxs_roots, np.where(idxs_roots ==i))
                        idxs_all_others = np.delete(idxs_all_others, np.where(idxs_all_others == i))
                        raise ValueError('Will not consider coordinates {} as root.'.format(point))
                    idxs = np.where(np.logical_and(dots >= 0, np.logical_and(vals <= (15 * np.pi/180), vals2 <= 0.3)))[0]
                    if idxs.size == 0:
                        raise ValueError('Continue to next point')
                    sign_temp = np.sign(temp[idxs])
                    idxs = idxs[np.where(sign_temp.dot(np.sign(self.grid[tuple(point)])) == point.size)[0]]
                    idxs = idxs[np.argsort(vals[idxs])[:min(1, idxs.shape[0])]]
                    if not is_full:
                        if not enable_all:
                            l = [np.argmin(vals[idxs.astype("int")])]  # take all the points that might be reached from
                                                                 # current vertex via the dominating direction
                                                                 # of the body
                        else:
                            l = np.arange(idxs.shape[0]).astype("int")

                        idxs = np.unique(self.avoidRedundantConnection(point, body[neighbors[idxs], :], neighbors[idxs]))
                        for j in idxs:
                            edge_endpoint = j
                            graphs[body_idx].add_edge(int(i), int(edge_endpoint))
                            idxs_leafs = np.delete(idxs_leafs, np.where(idxs_leafs == i))
                            idxs_roots = np.delete(idxs_roots, np.where(idxs_roots == edge_endpoint))
                            g_all.add_edge(int(i+idx_shift), int(edge_endpoint+idx_shift))
                            cnt+=1
                    else:
                        if enable_weights:
                            for j in idxs: # This requires a graph with weights
                                edge_endpoint = neighbors[j]
                                graphs[body_idx].add_edge(int(i), int(edge_endpoint))
                                idxs_leafs = np.delete(idxs_leafs, np.where(idxs_leafs == (i+idx_shift)))
                                idxs_roots = np.delete(idxs_roots, np.where(idxs_roots == (edge_endpoint + idx_shift)))
                                g_all.add_edge(int(i + idx_shift), int(edge_endpoint + idx_shift))
                        else:
                            if not enable_all:
                                l = np.argmin(vals[idxs])
                            else:
                                l = np.arange(idxs.shape[0]).astype("int")
                            for j in l:
                                edge_endpoint = neighbors[idxs[j]]
                                graphs[body_idx].add_edge(int(i), int(edge_endpoint))
                                idxs_leafs = np.delete(idxs_leafs, np.where(idxs_leafs == (i + idx_shift)))
                                idxs_roots = np.delete(idxs_roots, np.where(idxs_roots == (edge_endpoint + idx_shift)))
                                g_all.add_edge(int(i + idx_shift), int(edge_endpoint + idx_shift))
                except:
                    continue
            idx_shift += body.shape[0]
            idxs_leafs = np.array(list(set(idxs_leafs) - set(idxs_roots)))
            idxs_all_others = np.array(list(set(idxs_all_others) - (set(idxs_leafs).union(set(idxs_roots)))))
            leaves.append(deepcopy(idxs_leafs))
            roots.append(deepcopy(idxs_roots))
            all_others.append(deepcopy(idxs_all_others))
            roots_all = np.hstack((roots_all, idxs_roots+idx_shift))
            leaves_all = np.hstack((leaves_all, idxs_leafs+idx_shift))
            print(bcolors.BOLD + "Graph {} contains {} vertices and {} edges".format(body_idx, graphs[body_idx].number_of_vertices,
                                                                      graphs[body_idx].number_of_edges))
            print(bcolors.NORMAL)
        shifts = np.cumsum([x.shape[0] for x in self.clustered_bodies])
        connections = []
        for i in range(len(graphs)):
            for j in range(len(graphs)):
                if i == j:
                    continue
                else:
                    from_roots = np.array(np.meshgrid(roots[i],
                                                          np.unique(np.hstack((roots[j], leaves[j], all_others[j]))))).T.reshape(-1, 2)
                    from_leaves = np.array(np.meshgrid(leaves[i],
                                                          np.unique(np.hstack((roots[j], leaves[j], all_others[j]))))).T.reshape(-1, 2)
                    from_others = np.array(np.meshgrid(all_others[i],
                                                          np.unique(np.hstack((roots[j], leaves[j],all_others[j]))))).T.reshape(-1, 2)
                    g_all, roots_all, _, connections= \
                        self.addConnections(from_roots, g_all, i,j,roots_all,
                                            shift_idx_root=(0 if i == 0 else shifts[i-1]),
                                            shift_idx_leaf=(0 if j == 0 else shifts[j-1]),
                                            enable_weights=enable_weights,
                                            connections=connections)
                    g_all, roots_all, leaves_all, connections = \
                        self.addConnections(from_leaves, g_all, i,j,roots_all,
                                            shift_idx_root=(0 if i == 0 else shifts[i-1]),
                                            shift_idx_leaf=(0 if j == 0 else shifts[j-1]),
                                            is_leaf=leaves_all,
                                            enable_weights=enable_weights,
                                            connections=connections)
                    g_all, roots_all, leaves_all, connections = \
                        self.addConnections(from_others, g_all, i,j, roots_all,
                                            shift_idx_root=(0 if i == 0 else shifts[i-1]),
                                            shift_idx_leaf=(0 if j == 0 else shifts[j-1]),
                                            enable_weights=enable_weights,
                                            connections=connections)
        np.savez('{}Graphs_{}.npz'.format(self.save_path, self.file_prefix), g_all=jgrapht.io.exporters.generate_csv(g_all),
                 graphs=[jgrapht.io.exporters.generate_csv(x) for x in graphs], leaves=leaves, roots=roots,
                 roots_all=roots_all, leaves_all=leaves_all, connections=connections)
        return g_all, graphs, roots, leaves, roots_all, leaves_all, connections

    def findTheStartingVertexOfLongestPathInGraph(self, graph=None, return_all=False):
        all_paths_alg = johnson_allpairs(self.g if graph is None else graph)
        path_lengths = []
        all_paths = []
        for root in (self.g if graph is None else graph).vertices:
            longest_path_len = 0
            longest_path = None
            for leaf in (self.g if graph is None else graph).vertices:
                if root == leaf:
                    continue
                path = all_paths_alg.get_path(root, leaf)
                if path is not None:
                    all_paths.append((path.vertices, len(path.vertices)))
                if path is not None and longest_path_len <= len(path.vertices):
                    longest_path,longest_path_len = path.vertices, len(path.vertices)
            if longest_path_len > 0:
                path_lengths.append((longest_path, longest_path_len))

        if not return_all:
            i = np.argmax(np.array([x[1] for x in path_lengths]))
            return path_lengths[i][0][0], all_paths
        else:
            return all_paths,path_lengths

    def saveFile(self, file_name, data):
        with open(file_name, 'wb') as outfile:
            pickle.dump(data, outfile, protocol=pickle.HIGHEST_PROTOCOL)

    def loadFile(self, file_name):
        with open(file_name, 'rb') as outfile:
            return pickle.load(outfile)

    def plotResults(self):
        x_min, x_max, y_min, y_max = 31.0583, 33.6917, 31.5100, 35.4300
        ax = plotMap(self.grid, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
        if self.epsilon_body > 0:
            bodies = self.bodies
        else:
            bodies = self.full_bodies
        colors = pl.cm.jet(np.linspace(0, 1, len(bodies)))
        for i in range(len(bodies)):
            if bodies[i].ndim > 1:
                ax.scatter(bodies[i][:, 0] / (self.grid.shape[0] - 1) * (x_max - x_min) + x_min, bodies[i][:, 1]/ (self.grid.shape[1] - 1) * (y_max - y_min) + y_min, color=colors[i])
            else:
                ax.scatter(bodies[i][0]/ (self.grid.shape[0] - 1) * (x_max - x_min) + x_min, bodies[i][1]/ (self.grid.shape[1] - 1) * (y_max - y_min) + y_min, color=colors[i])
        plt.xticks(np.arange(31.5, 34 , 0.5))
        plt.yticks(np.arange(32, 35.5 , 0.5))
        plt.xlabel('Latitude')
        plt.ylabel('Longitude')
        plt.gcf().tight_layout()
        plt.savefig('{}Segmentation_{}.png'.format(self.save_path,self.file_prefix))

        # plot clustering
        ax = plotMap(self.grid, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
        colors = pl.cm.jet(np.linspace(0, 1, self.k))
        for i in range(self.k):
            ax.scatter(self.clustered_bodies[i][:,0]/ (self.grid.shape[0] - 1) * (x_max - x_min) + x_min, self.clustered_bodies[i][:,1]/ (self.grid.shape[1] - 1) * (y_max - y_min) + y_min, color=colors[i])
        
        plt.xticks(np.arange(31.5, 34 , 0.5))
        plt.yticks(np.arange(32, 35.5 , 0.5))
        plt.xlabel('Latitude')
        plt.ylabel('Longitude')
        plt.gcf().tight_layout()
        plt.savefig('{}Clustering_{}.png'.format(self.save_path,self.file_prefix))

        # close all figures
        plt.close('all')

    def findSubOptimalPlacing(self, continue_from=-1):
        if continue_from == -1 and not self.show:
            start_ellip = timer()
            points = self.polynomialGridSearch()
            for point in tqdm(points,ncols=100):
                if np.linalg.norm(self.grid[tuple(point)]) > 0 and not self.checkIfContained(point):
                    self.IOCS(point)
            end_ellip = timer()
            print(bcolors.BOLD + bcolors.OKGREEN + 'IOCS ended in {} seconds'.format(end_ellip - start_ellip))
            print(bcolors.NORMAL)
            self.saveFile(file_name=('{}Ellipsoids_{}.dat'.format(self.save_path, self.file_prefix)),
                          data=dict(zip(['ellipsoids','bodies','full_bodies'],
                                        [self.ellipsoids, self.bodies, self.full_bodies])))
        else:
            temp = self.loadFile(file_name=('{}Ellipsoids_{}.dat'.format(self.save_path, self.file_prefix)))
            self.ellipsoids = temp['ellipsoids']
            self.bodies = temp['bodies']
            self.full_bodies = temp['full_bodies']

        start_clustering = timer()
        blobs, assignments_per_blob, assignments_per_point = self.clusterWaves(continue_from)
        self.clustered_bodies = []
        for idx in range(self.k):
            cluster_idx = np.where(assignments_per_blob == idx)[0].astype("int")
            self.clustered_bodies.append(np.unique(np.vstack([(x.P[:, [0, 1]] / FORCE_NEIGHBORING).astype("int")
                                                              for x in blobs[cluster_idx]]), axis=0))
        print(bcolors.BOLD + bcolors.OKGREEN + 'Total time for clustering WC is {} seconds'.format(timer() - start_clustering))
        print(bcolors.NORMAL)
        self.plotResults()
        if self.show:
            exit(-9)
        start_graph_based = timer()
        if continue_from < 3:
            g_all, graphs, roots, leaves, roots_all, leaves_all, connections = self.generateGraph(enable_all=True,
                                                                                                  is_full=(self.epsilon_body == 0.0))
        else:
            G = np.load('{}Graphs_{}.npz'.format(self.save_path,self.file_prefix), allow_pickle=True)
            g_all = jgrapht.create_graph(directed=True, weighted=False)
            graphs_strings = G['graphs']
            graphs = [jgrapht.create_graph(directed=True,weighted=False) for i in range(self.k)]
            jgrapht.io.importers.parse_csv(g_all, str(G['g_all']))
            for i in range(self.k):
                jgrapht.io.importers.parse_csv(graphs[i], str(graphs_strings[i]))
            roots = G['roots'].tolist()
            leaves = G['leaves'].tolist()
            roots_all = G['roots_all'].tolist()
            leaves_all = G['leaves_all'].tolist()
            connections = G['connections'].tolist()

        # retrieve only $k$ largest paths where for any two paths, no path is a subpath of the other
        positions = np.empty((3,self.k, self.d))

        if continue_from < 4:
            # Heuristic choice
            for i,body in enumerate(self.clustered_bodies):
                A = np.vstack([self.grid[tuple(x)] for x in body])
                u_vecs, counts = np.unique(np.sign(A), return_counts=True, axis=0)
                dominating_vec = u_vecs[np.argmax(counts)] / np.linalg.norm(u_vecs[np.argmax(counts)])
                idxs = np.where(np.sign(A).dot(np.sign(dominating_vec)) == dominating_vec.shape[0])[0]
                vecs = body[idxs] - np.mean(body[idxs], axis=0)
                vals = np.dot(vecs, dominating_vec)
                positions[0, i, :] = body[idxs[int(np.argmin(vals))]]
            print(bcolors.OKGREEN + 'Finished computing initial positions for drifters via heuristical methods' + bcolors.ENDC)
            # Find longest path in each graph seperately
            paths_in_graph = [[] for i in range(len(graphs))]
            for i,graph in enumerate(graphs):
                idx, paths_in_graph[i] = self.findTheStartingVertexOfLongestPathInGraph(graph=graph, return_all=False)
                positions[1, i, :] = self.clustered_bodies[i][idx]
            print(bcolors.OKGREEN + 'Finished computing initial positions for drifters via graph based methods' + bcolors.ENDC)
            np.savez('{}paths_in_graphs_{}.npz'.format(self.save_path, self.file_prefix), positions=positions, paths_in_graph=paths_in_graph)
        # Find k longest paths in the combined graph
        # old technique
        else:
            temp = np.load('{}paths_in_graphs_{}.npz'.format(self.save_path, self.file_prefix), allow_pickle=True)
            positions = temp['positions']
            paths_in_graph = temp['paths_in_graph'].tolist()
        print(bcolors.BOLD + 'Starting to compute initial positions for drifters via inter-connected graphs' + bcolors.ENDC)
        if continue_from < 5:
            parsed_paths = [item for sublist in paths_in_graph for item in sublist]
            johnson_graphs = [johnson_allpairs(x) for x in graphs]
            shift_idxs = np.hstack((0,np.cumsum([x.number_of_vertices for x in graphs])))
            unified_paths = []
            for connection in connections:
                i, vertex_i, j, vertex_j = connection
                temp_paths_from_j = [x for x in paths_in_graph[j] if x[0][0] == vertex_j]
                temp_paths_to_i = [x for x in paths_in_graph[i] if x[0][-1] == vertex_i]
                unified_temp_paths_to_i = []
                # shift indices
                for list_i in range(len(temp_paths_to_i)):
                    if len(temp_paths_to_i) > 0:
                        temp_paths_to_i[list_i] = ([x + shift_idxs[i] for x in temp_paths_to_i[list_i][0]],
                                                        temp_paths_to_i[list_i][1])

                for list_j in range(len(temp_paths_from_j)):
                    if len(temp_paths_from_j) > 0:
                        temp_paths_from_j[list_j] = ([x + shift_idxs[j] for x in temp_paths_from_j[list_j][0]],
                                                          temp_paths_from_j[list_j][1])

                # check if there are inter_graph paths including vertex_i
                temp_paths_to_i = [x for x in paths_in_graph[i] if x[0][-1] == vertex_i]
                if len(unified_paths) > 0:
                    unified_temp_paths_to_i = [x for x in unified_paths if x[0][-1] == (vertex_i + shift_idxs[i])]

                if len(temp_paths_to_i) > 0 and len(temp_paths_from_j) > 0:
                    temp = getAllPossiblePaths(temp_paths_to_i, temp_paths_from_j)
                    unified_paths.extend(copy.deepcopy(temp))

                if len(unified_temp_paths_to_i) > 0:
                    temp2 = getAllPossiblePaths(unified_temp_paths_to_i, temp)
                    print('Length of temp_2 is {}'.format(len(temp2)))
                    unified_paths.extend(copy.deepcopy(temp2))
            unified_paths.sort(key = lambda x: x[1])
            i = 0
            if False:
                while True:
                    advance_i = True
                    if i < (len(unified_paths) - 1):
                        for j in range(i+1, len(unified_paths)):
                            if set(unified_paths[i][0]).issubset(unified_paths[j][0]):
                                advance_i = False
                                del(unified_paths[i])
                                break
                        if advance_i:
                            i += 1
                    else:
                        break
            else:
                print('Removing Inclusions has been initiated')
                unified_paths = removeInclusions(unified_paths, self.save_path, self.file_prefix)
            print('Number of possibe paths is {}'.format(len(unified_paths)))
            np.save('{}unified_paths_{}.npy'.format(self.save_path, self.file_prefix), unified_paths)
            print('Saved unified paths')
        else:
            temp = np.load('{}unified_paths_{}.npy'.format(self.save_path, self.file_prefix), allow_pickle=True)
            unified_paths = temp.tolist()
        print('length of Connections is {}'.format(len(connections)))
        print('length of Unified Paths are {}'.format(len(unified_paths)))
        if False:
            unified_graph = johnson_allpairs(g_all)
            paths = []
            for root in roots_all:
                for leaf in leaves_all:
                    try:
                        path = unified_graph.get_path(int(root), int(leaf))
                    except:
                        continue
                    dont = False
                    replace_idx = None
                    if path is not None:
                        if len(paths) > 0:
                            if np.any([set(path.vertices).issubset(x[0]) for x in paths]):
                                dont = True
                            if np.any([set(x[0]).issubset(path.vertices) for x in paths]):
                                replace_idx = np.where([set(x[0]).issubset(path.vertices) for x in paths])[0][0]
                        if not dont:
                            if replace_idx is None:
                                paths.append((path.vertices, len(path.vertices)))
                            else:
                                paths[replace_idx] = (path.vertices, len(path.vertices))

        # make sure that paths chosen that start from different nodes
        temp = copy.deepcopy(unified_paths)
        while True:
            len_paths = np.array([len(x) for x in temp])
            sorted_idxs = np.argsort((-1) * len_paths)
            idxs = sorted_idxs[:self.k].astype("int")
            initials = [temp[x][0] for x in idxs]
            to_delete = []
            for i in range(self.k):
                for j in range(i+1, self.k):
                    if initials[i] == initials[j]:
                        to_delete.append(temp[idxs[j]])

            if len(to_delete) == 0 or len(len_paths) == self.k:
                break
            else:
                for element in to_delete:
                    try:
                        temp.remove(element)
                    except:
                        continue

        unified_paths = copy.deepcopy(temp)
        len_paths = np.array([len(x) for x in unified_paths])
        sorted_idxs = np.argsort((-1) * len_paths)
        sizes = np.cumsum([x.shape[0] for x in self.clustered_bodies])
        idxs = sorted_idxs[:self.k].astype("int")
        raw_paths = [unified_paths[x] for x in idxs]
        raw_paths_initial_pos = [x[0] for x in raw_paths]
        print('raw paths initial are {}'.format(raw_paths_initial_pos))
        print('Sizes are {}'.format(sizes))
        for i in range(len(raw_paths_initial_pos)):
            idx_shift = np.where(raw_paths_initial_pos[i] < sizes)[0][0]
            if self.clustered_bodies[idx_shift].shape[0] == (raw_paths_initial_pos[i] -sizes[idx_shift-1]):
                idx_shift += 1
                positions[2, i, :] = self.clustered_bodies[idx_shift][0]
            else:
                positions[2, i, :] = self.clustered_bodies[idx_shift][raw_paths_initial_pos[i] -
                                                                  (0 if idx_shift == 0 else sizes[idx_shift-1])]
        print(bcolors.BOLD + 'Finished computing initial positions for drifters via inter-connected graphs' +
              bcolors.ENDC)

        np.save('{}initial_locations_{}.npy'.format(self.save_path,self.file_prefix), positions)
        print(bcolors.BOLD + bcolors.OKGREEN + 'Time for finding suboptimal dropping positions is {} seconds'.format(timer() - start_graph_based))
        print(bcolors.NORMAL)
        return positions

    def plotEllipsoid(self, ellipsoid, center):
        """
        This function serves only for plotting a 2D ellipsoid.

        :param ellipsoid: An orthogonal matrix representing the ellipsoid's axes lenghts and rotation
        :param center: The center of ellipsoid represented by a numpy array.
        :return: None.
        """
        N = 10000  # numer of points on the boundary of the ellipsoid.
        _, D, V = np.linalg.svd(ellipsoid, full_matrices=True)  # attain the axes lengthes and rotation of the ellipsoid
        a = 1.0 / D[0]
        b = 1.0 / D[1]
        theta = np.expand_dims(np.arange(start=0, step=1.0 / N, stop=2.0*np.pi + 1.0/N), 1).T

        state = np.vstack((a * np.cos(theta), b * np.sin(theta)))
        X = np.dot(V, state) + center[:,np.newaxis]
        plt.plot(X[0, :], X[1, :], color='blue')


def plotMap(grid, indices=None, x_min=None, x_max=None, y_min=None, y_max=None):
    positions = np.indices((grid.shape[0], grid.shape[1])).T.reshape(-1, 2)
    fig, ax = plt.subplots()
    idxs = [i for i in range(positions.shape[0]) if np.linalg.norm(grid[tuple(positions[i])]) > 0]
    if indices is None:
        if x_min is None:
            q = ax.quiver(positions[idxs,0], positions[idxs,1], grid[positions[idxs,0], positions[idxs,1],0], grid[positions[idxs,0], positions[idxs,1],1], angles='xy')
        else:
            q = ax.quiver(positions[idxs,0] / (grid.shape[0] - 1) * (x_max - x_min) + x_min, positions[idxs,1] / (grid.shape[1] - 1) * (y_max - y_min) + y_min, grid[positions[idxs,0], positions[idxs,1],0], grid[positions[idxs,0], positions[idxs,1],1], angles='xy')
    else:
        q = ax.quiver(indices[:, 0], indices[:, 1], grid[indices[:, 0], indices[:, 1],0],
                      grid[indices[:,0], indices[:, 1], 1], angles='xy')
    
    return ax


def main(data_folder, preprocess=True, file_type='.dat', number_of_files=1, eps_g=None, eps_b=0, k=10,
         coreset_sample_size=1000, over_time=False, tol=0.02, resume=False, show=False):
    paths = [x for x in os.walk(data_folder)]
    done = []
    if resume:
    	with open("resume_from_maps_init.pkl", "rb") as open_file:
    	    paths = pickle.load(open_file)

    for i, file_path_tuple in enumerate(paths):
        for file_name in file_path_tuple[-1]:
            if file_name.endswith(file_type):
                if file_type =='.nc':
                    with open('resume_from_maps_init.pkl', "wb") as open_file:
                        pickle.dump(paths[i:], open_file)
                    start_main = timer()
                    print(bcolors.WARNING + '****************************************************************************')
                    print(bcolors.BOLD + bcolors.WARNING + "Proccessing File: {}".format(file_name))
                    print(bcolors.NORMAL)
                    U, V = readNetCDFFile(file_path_tuple[0]+'/'+file_name, over_time=over_time)
                    if not over_time:
                        preprocessed_files = [ndi.correlate(np.mean(x,0), np.full((3, 3), 1 / 9)).T[None] for x in [U,V]]
                        preprocessed_files_2 = [np.mean(x,0).T[None] for x in [U,V]]
                    grid = np.append(*preprocessed_files, axis=0).T
                    grid2 = np.append(*preprocessed_files_2, axis=0).T
                    
                    saveVels(grid, file_path=file_path_tuple[0]+'/', smoothed=True)
                    saveVels(grid2, file_path=file_path_tuple[0]+'/', smoothed=False)
                    if eps_g is None:
                        eps_g = np.around(10 / grid.shape[0], 2)
                    drifter_placer = CurrentEstimation(grid, epsilon_grid=eps_g, k=k, epsilon_body=eps_b,
                                                       coreset_sample_size=coreset_sample_size, tolerance=tol,
                                                       save_path=file_path_tuple[0]+'/', show=show)
                    drifter_placer.findSubOptimalPlacing(continue_from=-1)
                    end_main = timer()
                    print(bcolors.HEADER + bcolors.OKGREEN + 'Whole program took {} seconds'.format(end_main - start_main))
                    print(bcolors.NORMAL)
                    np.save(file_path_tuple[0] + '/' + 'Time.npy', end_main - start_main)
    

if __name__ == '__main__':
    ns = parser.parse_args()  # parser
    main(data_folder=ns.d, preprocess=ns.pp, file_type=ns.ft, number_of_files=ns.nf, eps_g=ns.eps_g,
         eps_b=ns.eps_b, k=ns.k, coreset_sample_size = ns.coreset_sample_size, over_time=ns.time,
         tol=ns.tol, resume=ns.resume, show=ns.show)


