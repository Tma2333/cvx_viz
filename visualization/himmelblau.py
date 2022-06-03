import sys
import numpy as np

from manim import *
config.media_width = "75%"
config.verbosity = "WARNING"

sys.path.append('../')
from cvx_viz import SubGradientDescent

def fx(self, x1, x2):
    fx = 0.1*((x1**2 + x2 - 1.1)**2 +  (x1 + x2**2 - 0.7) **2)
    return fx
    
    
def dfx(self, x1, x2):
    dx1 = 0.1 * (4 * x1 * (x1**2 + x2 - 1.1) + 2 * (x1 + x2**2 - 0.7))
    dx2 = 0.1 * (2 * (x1**2 + x2 - 1.1) + 4 * x2 * (x1 + x2**2 - 0.7))
    return np.array([dx1, dx2])

SubGradientDescent.max_iter=10
SubGradientDescent.x10 = -1.91
SubGradientDescent.x20 = 1
SubGradientDescent.step_size = 1
SubGradientDescent.mode = '3d'
SubGradientDescent.fx = fx
SubGradientDescent.dfx = dfx
SubGradientDescent.verbose = False
SubGradientDescent.display_context = True

class Himmelblau(SubGradientDescent):
    pass