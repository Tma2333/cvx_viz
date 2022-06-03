import sys
import numpy as np

from manim import *
config.media_width = "75%"
config.verbosity = "WARNING"

sys.path.append('../')
from cvx_viz import SubGradientDescent

def fx(self, x1, x2):
    fx = np.abs(x1) + np.abs(x2)
    return fx
    
    
def dfx(self, x1, x2):
    if x1 < 0:
        dx1 = -1
    if x1 > 0:
        dx1 = 1
    if x2 < 0:
        dx2 = -1
    if x2 > 0:
        dx2 = 1
    if x1 == 0:
        dx1 = 0
    if x2 == 0:
        dx2 = 0
    return np.array([dx1, dx2])

SubGradientDescent.max_iter=10
SubGradientDescent.x10 = 0.2
SubGradientDescent.x20 = 1.5
SubGradientDescent.step_size = 0.25
SubGradientDescent.mode = '3d'
SubGradientDescent.fx = fx
SubGradientDescent.dfx = dfx
SubGradientDescent.verbose = False
SubGradientDescent.display_context = True
SubGradientDescent.cam_mode = 'zaxis_rotation'

class L1norm(SubGradientDescent):
    pass