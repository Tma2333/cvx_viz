from manim import *
import numpy as np

from .viz_base import CvxOptViz


class SubGradientDescent (CvxOptViz):
    def dfx(self, x1, x2):
        return NotImplementedError('You need to overide dfx method')

    
    def update_step (self, x1, x2, it=None):
        g = self.dfx(x1, x2)
        x1_next = x1 - self.step_size * g[0]
        x2_next = x2 - self.step_size * g[1]
        return x1_next, x2_next 

    
    def check_derive_attribute(self):
        default_val = {'step_size': 1}

        self._check_attribute(default_val)

        self.apply_constraints_3D = False
        self.apply_constraints_2D = False

        if self.verbose:
            self._print_attribute(default_val, 'Derive')
    
    
    def initial_context(self):
        g = self.dfx(self.x10, self.x20)
        x1_next, x2_next = self.update_step(self.x10, self.x20, 0)
        formula = MathTex(f'x^{{(k+1)}} = x^k - \\alpha g ', font_size=30)
        context = MathTex(f'\\begin{{bmatrix}} {x1_next:.2f} \\\\ {x2_next:.2f} \\end{{bmatrix}} = \\begin{{bmatrix}} {self.x10:.2f} \\\\ {self.x20:.2f} \\end{{bmatrix}} - {self.step_size} \\begin{{bmatrix}} {g[0]:.2f} \\\\ {g[1]:.2f} \\end{{bmatrix}} ', font_size=25)
        return formula, context
    

    def update_context(self, x1, x2, it=None):
        g = self.dfx(x1, x2)
        x1_next, x2_next = self.update_step(x1, x2, it)
        formula = MathTex(f'x^{{(k+1)}} = x^k - \\alpha g ', font_size=30)
        context = MathTex(f'\\begin{{bmatrix}} {x1_next:.2f} \\\\ {x2_next:.2f} \\end{{bmatrix}} = \\begin{{bmatrix}} {x1:.2f} \\\\ {x2:.2f} \\end{{bmatrix}} - {self.step_size} \\begin{{bmatrix}} {g[0]:.2f} \\\\ {g[1]:.2f} \\end{{bmatrix}} ', font_size=25)
        return formula, context


class ProjectedSubGradientDescent (CvxOptViz):
    def dfx(self, x1, x2):
        return NotImplementedError('You need to overide dfx method')

    
    def proj2feasible(self, x1, x2):
        return NotImplementedError('You need to overide proj2feasible method')

    
    def update_step (self, x1, x2, it=None):
        # project 
        if it == 0: 
            x1_next, x2_next = self.proj2feasible(x1, x2)
        elif it % 2:
            g = self.dfx(x1, x2)
            x1_next = x1 - self.step_size * g[0]
            x2_next = x2 - self.step_size * g[1]
        else:
            x1_next, x2_next = self.proj2feasible(x1, x2)
        return x1_next, x2_next 

    
    def check_derive_attribute(self):
        default_val = {'step_size': 1}

        self._check_attribute(default_val)

        self.apply_constraints_3D = True
        self.apply_constraints_2D = True

        self.max_iter = self.max_iter * 2 + 1

        if self.verbose:
            self._print_attribute(default_val, 'Derive')
    
    
    def initial_context(self):
        x1_next, x2_next = self.update_step(self.x10, self.x20, 0)
        formula = MathTex(f'x^{{(k+1)}} = \Pi_C(z^{{(k+1)}}) ', font_size=30)
        context = MathTex(f'\\begin{{bmatrix}} {x1_next:.2f} \\\\ {x2_next:.2f} \\end{{bmatrix}} = \Pi_C(\\begin{{bmatrix}} {self.x10:.2f} \\\\ {self.x20:.2f} \\end{{bmatrix}})', font_size=25)
        return formula, context
    

    def update_context(self, x1, x2, it=None):
        g = self.dfx(x1, x2)
        x1_next, x2_next = self.update_step(x1, x2, it)
        if it % 2:
            formula = MathTex(f'z^{{(k+1)}} = x^k - \\alpha g ', font_size=30)
            context = MathTex(f'\\begin{{bmatrix}} {x1_next:.2f} \\\\ {x2_next:.2f} \\end{{bmatrix}} = \\begin{{bmatrix}} {x1:.2f} \\\\ {x2:.2f} \\end{{bmatrix}} - {self.step_size} \\begin{{bmatrix}} {g[0]:.2f} \\\\ {g[1]:.2f} \\end{{bmatrix}} ', font_size=25)
        else:
            formula = MathTex(f'x^{{(k+1)}} = \Pi_C(z^{{(k+1)}}) ', font_size=30)
            context = MathTex(f'\\begin{{bmatrix}} {x1_next:.2f} \\\\ {x2_next:.2f} \\end{{bmatrix}} = \Pi_C(\\begin{{bmatrix}} {x1:.2f} \\\\ {x2:.2f} \\end{{bmatrix}})', font_size=25)
        return formula, context


class SubGradientDescentWithMomentum (CvxOptViz):
    def dfx(self, x1, x2):
        return NotImplementedError('You need to overide dfx method')

    
    def update_step (self, x1, x2, it=None):
        g = self.dfx(x1, x2)
        self.v = self.beta * self.v - self.step_size * g
        x1 = self.v[0] + x1
        x2 = self.v[1] + x2
        return x1, x2 
    
    
    def check_derive_attribute(self):
        default_val = {'step_size': 1, 'beta': 0.01}

        self._check_attribute(default_val)
    
        self.v = np.zeros(2)

        self.apply_constraints_3D = False
        self.apply_constraints_2D = False

        self.display_context = False

        if self.verbose:
            self._print_attribute(default_val, 'Derive')
