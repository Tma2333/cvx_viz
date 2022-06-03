import sys
import numpy as np

from manim import *
config.media_width = "75%"
config.verbosity = "WARNING"

sys.path.append('../')
from cvx_viz import SubGradientDescent

def fx(self, x1, x2):
    fx = 0.5 * x1**2 +  0.05 * x2 ** 2
    return fx
    
    
def dfx(self, x1, x2):
    dx1 = x1
    dx2 = 0.1 * x2
    return np.array([dx1, dx2])

SubGradientDescent.max_iter=10
SubGradientDescent.x10 = 1.5
SubGradientDescent.x20 = 1.5
SubGradientDescent.step_size = 1.5
SubGradientDescent.mode = '3d'
SubGradientDescent.fx = fx
SubGradientDescent.dfx = dfx
SubGradientDescent.verbose = False
SubGradientDescent.display_context = True

class Quadratic(SubGradientDescent):
    pass