import tensorflow as tf
import tensorflow_probability as tfp

from manim import *

# shortcuts
tfd = tfp.distributions
kernels = tfp.math.psd_kernels


class SampleTrajectory:

    def __init__(self, gp, n_index_points, random_state):
        self.u = random_state.randn(n_index_points)
        self.v = random_state.randn(n_index_points)
        self.gp = gp

    def __call__(self, theta):
        v_norm = np.linalg.norm(self.v, axis=None, ord=2)
        v_normed = np.true_divide(self.v, v_norm)

        c = np.sum(self.u * v_normed, axis=None)

        t = self.u - c * v_normed
        t_norm = np.linalg.norm(t, ord=2, axis=None)
        t_normed = np.true_divide(t, t_norm)

        eps = v_norm * (v_normed * np.cos(theta) + t_normed * np.sin(theta))
        gp_marginal = self.gp.get_marginal_distribution()
        return gp_marginal.bijector.forward(eps).numpy()


class GaussianProcessSamples(Scene):

    def make_line_graph(self, x, func, tracker, color):
        return self.ax.plot_line_graph(x_values=x.squeeze(axis=-1),
                                       y_values=func(tracker.get_value()),
                                       add_vertex_dots=False,
                                       line_color=color,
                                       vertex_dot_style=dict(fill_color=color),
                                       stroke_opacity=0.8)

    def construct(self):

        # colors = [BLUE, TEAL, GREEN, GOLD, RED, MAROON, PURPLE]
        colors = [RED, GREEN, BLUE]
        n_samples = len(colors)

        n_index_points = 512  # nbr of index points

        x_min, x_max = -.3, 3.
        X_grid = np.linspace(x_min, x_max, n_index_points).reshape(-1, 1)

        # kernel_cls = kernels.MaternFiveHalves
        kernel_cls = kernels.ExponentiatedQuadratic
        amplitude = 1.
        length_scale = .1

        seed = 8888
        random_state = np.random.RandomState(seed)

        kernel = kernel_cls(amplitude=tf.constant(amplitude, dtype="float64"), 
                            length_scale=tf.constant(length_scale, dtype="float64"))
        
        gp = tfd.GaussianProcess(kernel=kernel, index_points=X_grid)

        theta_tracker = ValueTracker(0.)

        self.ax = Axes(
            x_range=[x_min, x_max, 1.],
            y_range=[-4., 4., 1.],
            x_length=12,
            y_length=6,
            tips=False,
        )

        def make_updater(func, color):
            def updater(m):
                return m.become(self.make_line_graph(X_grid, func, theta_tracker, color))
            return updater

        graphs = VGroup()
        for color in colors:
            func = SampleTrajectory(gp, n_index_points, random_state)
            graph = self.make_line_graph(X_grid, func, theta_tracker, color)
            graph.add_updater(make_updater(func, color))
            graphs.add(graph)

        self.add(graphs)
        self.play(theta_tracker.animate.increment_value(TAU),
                  rate_func=rate_functions.linear, run_time=1.5)
