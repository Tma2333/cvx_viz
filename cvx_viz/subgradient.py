from manim import *
import numpy as np
import matplotlib.pyplot as plt

from .viz_base import CvxOptViz


class SubGradientDescent (CvxOptViz):
    def dfx(self, x1, x2):
        return NotImplementedError('You need to overide dfx method')

    
    def update_step (self, x1, x2):
        g = self.dfx(x1, x2)
        x1 = x1 - self.step_size * g[0]
        x2 = x2 - self.step_size * g[1]
        return x1, x2 
    
    
    def _test_derive_attribute(self):
        try:
            self.step_size
        except AttributeError:
            self.step_size = 1


class SubGradientDescentWithMomentum (CvxOptViz):
    def dfx(self, x1, x2):
        return NotImplementedError('You need to overide dfx method')

    
    def update_step (self, x1, x2):
        g = self.dfx(x1, x2)
        self.v = self.beta * self.v - self.step_size * g
        x1 = self.v[0] + x1
        x2 = self.v[1] + x2
        return x1, x2 
    
    
    def _test_derive_attribute(self):
        try:
            self.step_size
        except AttributeError:
            self.step_size = 1

        try:
            self.beta
        except AttributeError:
            self.beta = 0.01
        
        self.v = np.zeros(2)