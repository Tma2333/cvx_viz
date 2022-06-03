import sys
import cvxpy as cp
import numpy as np

from manim import *
config.media_width = "75%"
config.verbosity = "WARNING"

sys.path.append('../')
from cvx_viz import ProjectedSubGradientDescent


A = np.array([[1, -1], [-0.9, 0.2], [1, 0.3], [-1, -0.9]])
b = np.array([1, 1, 0.5, 1])


def fx(self, x1, x2):
    fx = 0.5 * x1**2 +  0.05 * x2 ** 2
    return fx


def proj2feasible(self, x1, x2):
    x = cp.Variable(2)
    z = np.array([x1, x2])
    obj = cp.Minimize(0.5*cp.norm2(x-z)**2)
    con = [A@x<=b]
    cp.Problem(obj, con).solve()
    return x.value[0], x.value[1]
    
def dfx(self, x1, x2):
    dx1 = x1
    dx2 = 0.1 * x2
    return np.array([dx1, dx2])


def constraints(self, x1, x2):
    if isinstance(x1, np.ndarray):
        array_shape = x1.shape
        x1 = x1.reshape(-1)
        x2 = x2.reshape(-1)
        x = np.vstack([x1, x2])
        out = np.all(A@x < b[:,None], axis=0)
        out = out.reshape(array_shape)
    else:
        x = np.array([x1, x2])
        out = np.all(A@x < b)
    
    return out

ProjectedSubGradientDescent.max_iter=5
ProjectedSubGradientDescent.x10 = -1.5
ProjectedSubGradientDescent.x20 = 1.5
ProjectedSubGradientDescent.surface_resolution=20
ProjectedSubGradientDescent.constraints = constraints
ProjectedSubGradientDescent.proj2feasible = proj2feasible
ProjectedSubGradientDescent.step_size = 1.5
ProjectedSubGradientDescent.mode = '3d'
ProjectedSubGradientDescent.fx = fx
ProjectedSubGradientDescent.dfx = dfx
ProjectedSubGradientDescent.verbose = False
ProjectedSubGradientDescent.display_context = True
ProjectedSubGradientDescent.anime = True

class QuadraticPGD(ProjectedSubGradientDescent):
    pass