from manim import *
import numpy as np
import matplotlib.pyplot as plt

class CvxOptViz(ThreeDScene):
    def construct(self):
        self._test_core_attribute()
        self._test_derive_attribute()
        
        self.axes = ThreeDAxes()
        self.add(self.axes)

        self._set_level_set_info()

        if self.mode == '3d':
            self.set_camera_orientation(phi = 45 * DEGREES, theta = -45 * DEGREES)
            self._create_opt_surface()
        elif self.mode == '2d':
            self.set_camera_orientation(phi = 0 * DEGREES, theta = 0 * DEGREES)
        
        self._create_contour()
        self._trace_optimization()

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
    

    def _set_level_set_info (self):
        if self.num_level % 2:
            self.num_level += 1

        delta = 0.01
        x = np.arange(-2.1, 2.1, delta)
        y = np.arange(-2.1, 2.1, delta)
        X, Y = np.meshgrid(x, y)
        Z = self.fx(X, Y)

        if self.fstar is None:
            self.fstar = Z.min()
        
        if self.fmax is None:
            self.fmax = Z.max()

        scale = np.logspace(-3, 0, self.num_level)    
        self.level_val = self.fstar + (self.fmax - self.fstar) * scale

        r_channel = np.linspace(0, 255, self.num_level//2, dtype='int')
        r_channel = np.concatenate([r_channel, np.ones(self.num_level//2, dtype='int')*255])
        g_channel = np.linspace(255, 0, self.num_level//2, dtype='int')
        g_channel = np.concatenate([np.ones(self.num_level//2, dtype='int')*255, g_channel])

        self.level_color = []
        for i in range(self.num_level):
            self.level_color.append((f'#{r_channel[i]:02X}{g_channel[i]:02X}{0:02X}', self.level_val[i]))

        fig = plt.figure()
        contour = plt.contour(X, Y, Z, levels=self.level_val)
        plt.close(fig)

        self.level_contour_line = contour.allsegs

    def _create_opt_surface(self):
        resolution_fa = 10

        opt_landscape = Surface(
            self._opt_surface,
            resolution=(resolution_fa, resolution_fa),
            v_range=[-2, +2],
            u_range=[-2, +2]
        )
        
        # opt_landscape.scale(1, about_point=ORIGIN)
        opt_landscape.set_style(fill_opacity=0.5, stroke_color=GREEN)
        opt_landscape.set_fill_by_value(axes=self.axes, colors=self.level_color, axis=2)
        
        self.add(opt_landscape)

    
    def _trace_optimization (self):
        x1 = self.x10
        x2 = self.x20

        x1t = ValueTracker(x1)
        x2t = ValueTracker(x2)

        fx = self.fx
        
        def _x_updater (x):
            x1 = x1t.get_value()
            x2 = x2t.get_value()
            fx_val = fx(x1, x2)
            x.move_to(np.array([x1, x2, fx_val]))

        def _x2D_updater (x):
            x1 = x1t.get_value()
            x2 = x2t.get_value()
            x.move_to(np.array([x1, x2, self.fstar]))

        x = Dot3D(self._opt_surface(x1, x2), radius=0.05, color=BLUE)
        x_2D = Dot3D(np.array([x1, x2, self.fstar]), radius=0.05, color=BLUE)
        
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

        # self.begin_ambient_camera_rotation(
        #     rate=PI / 10, about="theta"
        # )  
        for i in range(self.max_iter):
            x1, x2 = self.update_step(x1, x2)
            self.play(x1t.animate.set_value(x1), x2t.animate.set_value(x2))

        # self.stop_ambient_camera_rotation()

    def _opt_surface(self, x1, x2):
        return np.array([x1, x2, self.fx(x1, x2)])


    def _test_core_attribute(self):
        try:
            self.max_iter
        except AttributeError:
            self.max_iter = 5

        try:
            self.mode
        except AttributeError:
            self.mode = '3d'
        
        try:
            self.num_level
        except AttributeError:
            self.num_level = 20
        
        try:
            self.fstar
        except AttributeError:
            self.fstar = None
        
        try:
            self.fmax
        except AttributeError:
            self.fmax = None
        
        try:
            self.x10
        except AttributeError:
            if np.random.rand() < 0.5:
                sign = -1
            else:
                sign = 1
            self.x10 = sign * (np.random.rand()*0.5 + 1.5)

        try:
            self.x20
        except AttributeError:
            if np.random.rand() < 0.5:
                sign = -1
            else:
                sign = 1
            self.x20 = sign * (np.random.rand()*0.5 + 1.5)
    

    def _test_derive_attribute(self):
        raise NotImplementedError('You need to overide update_step method')


    def fx(self, x1, x2):
        raise NotImplementedError('You need to overide fx method')

    
    def update_step (self, x1, x2):
        raise NotImplementedError('You need to overide update_step method')