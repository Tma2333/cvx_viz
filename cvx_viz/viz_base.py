from distutils import core
from inspect import Attribute
from cloup import constraint
from manim import *
import numpy as np
import matplotlib.pyplot as plt

class CvxOptViz(ThreeDScene):
    def construct(self):
        # initialize necessary class variable
        self.check_core_attribute()
        self.check_derive_attribute()

        # create 3d axes
        self._create_3D_axes()

        self._set_level_set_info()
        self._set_initial_camera_location()
        
        if self.mode == '3d':
            self._create_opt_surface(apply_constraints=self.apply_constraints_3D)
        
        # create the contour plot
        self._create_contour()

        # animating the optimization trace
        if self.anime:
            self._trace_optimization()
    

    def _create_3D_axes(self):
        x0_offset = (self.x0_range[1] - self.x0_range[0]) * 0.1
        x0_tick = (self.x0_range[1] - self.x0_range[0])  * 0.1
        x1_offset = (self.x1_range[1] - self.x1_range[0]) * 0.1
        x1_tick = (self.x0_range[1] - self.x0_range[0])  * 0.1
        fx_offset = (self.fx_range[1] - self.fx_range[0]) * 0.1
        fx_tick = (self.fx_range[1] - self.fx_range[0])  * 0.1
        axe_x_range = [self.x0_range[0] - x0_offset, self.x0_range[1] + x0_offset, x0_tick]
        axe_y_range = [self.x1_range[0] - x1_offset, self.x1_range[1] + x1_offset, x1_tick]
        axe_z_range = [self.fx_range[0] - fx_offset, self.fx_range[1] + fx_offset, fx_tick]

        self.axes = ThreeDAxes(x_range=axe_x_range,
                               y_range=axe_y_range,
                               z_range=axe_z_range)

        if self.display_axes:
            self.add(self.axes)
    

    def _set_level_set_info (self):
        if self.num_level_set % 2:
            self.num_level_set += 1
        
        # Generate sample
        x0_delta = (self.x0_range[1] - self.x0_range[0]) *0.0025
        x0_offset = (self.x0_range[1] - self.x0_range[0]) * 0.025
        x1_delta = (self.x1_range[1] - self.x1_range[0]) *0.0025
        x1_offset = (self.x1_range[1] - self.x1_range[0]) * 0.025
        x = np.arange(self.x0_range[0]-x0_offset, self.x0_range[1]+x0_offset, x0_delta)
        y = np.arange(self.x1_range[0]-x1_offset, self.x1_range[1]+x1_offset, x1_delta)
        X, Y = np.meshgrid(x, y)
        Z = self.fx(X, Y)

        # Set min and max
        if self.fstar is None:
            self.fstar = Z.min()
        
        if self.fmax is None:
            self.fmax = Z.max()
        
        if self.apply_constraints_2D:
            idx = np.logical_not(self.constraints(X, Y))
            Z[idx] = self.fmax

        # Create level set
        scale = np.logspace(self.min_log_level, 0, self.num_level_set)    
        self.level_val = self.fstar + (self.fmax - self.fstar) * scale

        # Generate level set color
        r_channel = np.linspace(0, 255, self.num_level_set//2, dtype='int')
        r_channel = np.concatenate([r_channel, np.ones(self.num_level_set//2, dtype='int')*255])
        g_channel = np.linspace(255, 0, self.num_level_set//2, dtype='int')
        g_channel = np.concatenate([np.ones(self.num_level_set//2, dtype='int')*255, g_channel])

        self.level_color = []
        for i in range(self.num_level_set):
            self.level_color.append((f'#{r_channel[i]:02X}{g_channel[i]:02X}{0:02X}', self.level_val[i]))

        # Creat contour plot
        fig = plt.figure()
        contour = plt.contour(X, Y, Z, levels=self.level_val)
        plt.close(fig)

        self.level_contour_line = contour.allsegs
    
    
    def _set_initial_camera_location(self):
        if self.mode == '2d':
            self.set_camera_orientation(phi = 0 * DEGREES, theta = 0 * DEGREES)
            return
        
        if self.fixed_cam_preset == 'Q1':
            theta_angle = 45
            phi_angle = 45
        elif self.fixed_cam_preset == 'Q2':
            theta_angle = 135
            phi_angle = 45
        elif self.fixed_cam_preset == 'Q3':
            theta_angle = -135
            phi_angle = 45
        elif self.fixed_cam_preset == 'Q4':
            theta_angle = -45
            phi_angle = 45
        elif self.fixed_cam_preset == 'x0':
            theta_angle = -90
            phi_angle = 90
        elif self.fixed_cam_preset == 'x1':
            theta_angle = 0
            phi_angle = 90
        if self.cam_theta is not None:
            theta_angle = self.cam_theta
        if self.cam_phi is not None:
            phi_angle = self.cam_phi
            
        self.set_camera_orientation(phi = phi_angle * DEGREES, theta = theta_angle * DEGREES)


    def _create_opt_surface(self, apply_constraints=False):
        resolution_fa = self.surface_resolution

        opt_landscape = Surface(
            lambda u, v: self.axes.c2p(u, v, self._opt_surface(u, v, apply_constraints)),
            resolution=(resolution_fa, resolution_fa),
            v_range=self.x0_range,
            u_range=self.x1_range,
            should_make_jagged=True
        )
        
        opt_landscape.set_style(fill_opacity=self.surface_opacity, stroke_color=GREEN)
        opt_landscape.set_fill_by_value(axes=self.axes, colors=self.level_color, axis=2)
        
        self.add(opt_landscape)

    
    def _create_contour (self):
        if self.mode == '3d':
            contour_z = self.fstar
        elif self.mode == '2d':
            contour_z = 0.0
        
        for i, contour_lines in enumerate(self.level_contour_line):
            if len(contour_lines) == 0:
                continue
            for contour_line in contour_lines:
                x = contour_line[:, 0]
                y = contour_line[:, 1]
                z = np.ones_like(x) * contour_z

                line_graph = self.axes.plot_line_graph(x_values=x, y_values=y, z_values=z, 
                    add_vertex_dots=False, 
                    line_color=self.level_color[i][0])
                self.add(line_graph)
                

    def _trace_optimization (self):
        x1 = self.x10
        x2 = self.x20

        x1t = ValueTracker(x1)
        x2t = ValueTracker(x2)

        fx = self.fx
        
        def _x_updater (x):
            x1 = x1t.get_value()
            x2 = x2t.get_value()
            x.move_to(self.axes.c2p(x1, x2, fx(x1, x2)))

        def _x2D_updater (x):
            x1 = x1t.get_value()
            x2 = x2t.get_value()
            x.move_to(self.axes.c2p(x1, x2, self.fstar))

        x = Dot3D(self.axes.c2p(x1, x2, fx(x1, x2)), radius=0.05, color=BLUE)
        x_2D = Dot3D(self.axes.c2p(x1, x2, self.fstar), radius=0.05, color=BLUE)
        
        line =DashedLine(x.get_center(), x_2D.get_center())
        line.add_updater(lambda z: z.become(DashedLine(x.get_center(), x_2D.get_center())))

        x.add_updater(_x_updater)
        x_2D.add_updater(_x2D_updater)
        
        if self.mode == '3d':
            tracer_point = x
        elif self.mode == '2d':
            tracer_point = x_2D
        
        def _x_trace_updater (path):
            previous_path = path.copy()
            previous_path.add_points_as_corners([tracer_point.get_center()])
            path.become(previous_path)

        path = VMobject()
        path.set_color(BLUE)
        path.set_points_as_corners([tracer_point.get_center(), tracer_point.get_center()])
        path.add_updater(_x_trace_updater)

        self.add(x_2D)
        if self.mode == '3d':
            self.add(x)
            self.add(line)
        self.add(path)

        # Set up motion camera if cam mode set to
        if 'cam_mode' == 'zaxis_rotation':
            if self.zaxis_rotation_rate is None:
                self.zaxis_rotation_rate = PI/15
            self.begin_ambient_camera_rotation(rate= self.zaxis_rotation_rate, about="theta") 

        if 'cam_mode' == '3d_illusion':
            if self.illusion_rate is None:
                self.illusion_rate = 1
            self.begin_3dillusion_camera_rotation(rate=self.illusion_rate)

        if self.display_context:
            self._write_initial_context()
        
        for i in range(self.max_iter):
            if self.display_context:
                self._transform_context(x1, x2)
            x1, x2 = self.update_step(x1, x2)
            self.play(x1t.animate.set_value(x1), x2t.animate.set_value(x2))
        self.wait(1)

        # Stop camera movement
        if 'cam_mode' == 'zaxis_rotation':
            self.stop_ambient_camera_rotation()
        if 'cam_mode' == '3d_illusion':
            self.stop_3dillusion_camera_rotation()


    def _write_initial_context(self):
        self.ctxs = VGroup(*self.initial_context())
        self.ctxs.arrange(DOWN, aligned_edge=LEFT)
        self.ctxs.to_corner(UL)
        self.add_fixed_in_frame_mobjects(self.ctxs)
    

    def _transform_context(self, x1, x2):
        transform_target = VGroup(*self.update_context(x1, x2))
        transform_target.arrange(DOWN, aligned_edge=LEFT)
        transform_target.to_corner(UL)
        self.play(Transform(self.ctxs, transform_target))


    def _opt_surface(self, x1, x2, apply_constraints=False):
        if isinstance(x1, np.ndarray) or isinstance(x2, np.ndarray):
            out = self.fx(x1, x2)
            if apply_constraints:
                idx = np.logical_not(self.constraints(x1, x2))
                out[idx] = self.fmax
                return out
        else:
            out = self.fx(x1, x2)
            if apply_constraints:
                if self.constraints(x1, x2):
                    return out
                else:
                    return self.fmax
            else:
                return out

    
    def _print_attribute(self, attr_dict, attr_level):
        print('='*40)
        print(f'{attr_level} attribute')
        print('='*40)
        for key in attr_dict:
            value = CvxOptViz.__getattribute__(self, key)
            print(f'{key:<20}|{value}')
            print('-'*40)


    def _check_attribute(self, attr_dict):
        for attr in attr_dict:
            try:
                CvxOptViz.__getattribute__(self, attr)
            except AttributeError:
                CvxOptViz.__setattr__(self, attr, attr_dict[attr])


    def check_core_attribute(self):
        default_val = {'max_iter': 5, 'mode': '3d', 'num_level_set': 20,
                       'x0_range': [-2, 2], 'x1_range': [-2, 2], 'fx_range': [-3, 3],
                       'fstar': None, 'fmax': None, 
                       'min_log_level': -3, 
                       'surface_opacity': 0.5, 'surface_resolution': 10, 
                       'x10': None, 'x20': None,
                       'verbose': True, 'display_axes': True, 'anime': True,
                       'cam_mode': 'fixed',
                       'fixed_cam_preset': 'Q4', 'cam_theta': None, 'cam_phi': None, 
                       'illusion_rate': None, 'zaxis_rotation_rate': None,
                       'apply_constraints_2D': False, 'apply_constraints_3D': False,
                       'display_context': False}
        
        mode_choice = ['3d', '2d']
        cam_mode_choice = ['fixed', '3d_illusion', 'zaxis_rotation']
        fixed_cam_preset_choice = ['Q1', 'Q2', 'Q3', 'Q4', 'x0', 'x1']

        # Primary initialization and checking
        self._check_attribute(default_val)

        if self.mode not in mode_choice:
            raise ValueError(f'mode {self.mode} is not supported, supported mode: {mode_choice}')

        if self.cam_mode not in cam_mode_choice:
            raise ValueError(f'cam mode {self.cam_mode} is not supported, supported mode: {cam_mode_choice}')

        if self.fixed_cam_preset not in fixed_cam_preset_choice:
            raise ValueError(
                f'fixed camera preset {self.fixed_cam_preset_choice} is not supported, supported mode: {fixed_cam_preset_choice}')

        if self.x10 is None:
            if np.random.rand() < 0.5:
                sign = -1
            else:
                sign = 1
            self.x10 = sign * (np.random.rand()*0.5 + 1.5)

        if self.x20 is None:
            if np.random.rand() < 0.5:
                sign = -1
            else:
                sign = 1
            self.x20 = sign * (np.random.rand()*0.5 + 1.5)
        
        if self.verbose:
            self._print_attribute(default_val, 'Core')
        
        # Safe guard for not implemented mnethod
        try:
            self.constraints(0, 0)
        except NotImplementedError:
            self.apply_constraints_3D = False
            self.apply_constraints_2D = False


    def check_derive_attribute(self):
        raise NotImplementedError('You need to overide check_derive_attribute method')


    def fx(self, x1, x2):
        raise NotImplementedError('You need to overide fx method')
    

    def constraints(self, x1, x2):
        raise NotImplementedError('You need to overide constraints method')

    
    def update_step (self, x1, x2):
        raise NotImplementedError('You need to overide update_step method')


    def initial_context (self):
        raise NotImplementedError('You need to overide initial_context method')
    

    def update_context (self, x1, x2):
        raise NotImplementedError('You need to overide update_context method')