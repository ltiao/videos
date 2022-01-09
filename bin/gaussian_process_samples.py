import tensorflow as tf
import tensorflow_probability as tfp

from manim import *

# shortcuts
tfd = tfp.distributions
kernels = tfp.math.psd_kernels


class Test:

    def __init__(self, gp_marginal, n_index_points, random_state):
        self.u = random_state.randn(n_index_points)
        self.v = random_state.randn(n_index_points)
        self.gp_marginal = gp_marginal

    def __call__(self, theta):
        v_norm = np.linalg.norm(self.v, axis=None, ord=2)
        v_normed = np.true_divide(self.v, v_norm)

        c = np.sum(self.u * v_normed, axis=None)

        t = self.u - c * v_normed
        t_norm = np.linalg.norm(t, ord=2, axis=None)
        t_normed = np.true_divide(t, t_norm)

        eps = v_norm * (v_normed * np.cos(theta) + t_normed * np.sin(theta))
        return self.gp_marginal.bijector.forward(eps).numpy()


class GaussianProcessSamples(Scene):

    def construct(self):

        n_index_points = 512  # nbr of index points
        n_samples = 3
        # n_frames = 64

        x_min, x_max = -.5, 5.
        X_grid = np.linspace(x_min, x_max, n_index_points).reshape(-1, 1)

        # kernel_cls = kernels.MaternFiveHalves
        kernel_cls = kernels.ExponentiatedQuadratic
        amplitude = 1.
        length_scale = .6

        seed = 23
        random_state = np.random.RandomState(seed)

        # u = random_state.randn(n_samples, n_index_points)
        # v = random_state.randn(n_samples, n_index_points)

        # v_norm = np.linalg.norm(v, ord=2, axis=1, keepdims=True)
        # v_normed = np.true_divide(v, v_norm)

        # c = np.sum(u * v_normed, axis=1, keepdims=True)

        # t = u - c * v_normed
        # t_norm = np.linalg.norm(t, ord=2, axis=1, keepdims=True)
        # t_normed = np.true_divide(t, t_norm)

        kernel = kernel_cls(amplitude=tf.constant(amplitude, dtype="float64"), 
                            length_scale=tf.constant(length_scale, dtype="float64"))
        
        gp = tfd.GaussianProcess(kernel=kernel, index_points=X_grid)
        gp_marginal = gp.get_marginal_distribution()

        # def make_func(ind):
        #     def func(theta):
        #         eps = v_norm[ind] * (v_normed[ind] * np.cos(theta) + t_normed[ind] * np.sin(theta))
        #         return gp_marginal.bijector.forward(eps).numpy()
        #     return func

        theta_tracker = ValueTracker(0.)

        ax = Axes(
            x_range=[x_min, x_max, 1.],
            y_range=[-4., 4., 1.],
            x_length=12,
            y_length=6,
            # x_axis_config={
            # "numbers_to_include": np.arange(0, 5.1, 1),
            # "numbers_with_elongated_ticks": np.arange(-10, 10.01, 2),
            # },
            tips=False,
        )  # .add_coordinates()

        add_vertex_dots = False

        funcs = [Test(gp_marginal, n_index_points, random_state) for i in range(n_samples)]
        colors = [RED, GREEN, BLUE]

        def make_updater(func, color):
            def updater(m):
                return m.become(
                        ax.plot_line_graph(x_values=X_grid.squeeze(axis=-1),
                                           y_values=func(theta_tracker.get_value()),
                                           add_vertex_dots=add_vertex_dots,
                                           line_color=color,
                                           vertex_dot_style=dict(fill_color=color),
                                           stroke_opacity=0.8))
            return updater

        graphs = VGroup()
        for func, color in zip(funcs, colors):
            graph = ax.plot_line_graph(x_values=X_grid.squeeze(axis=-1),
                                       y_values=func(theta_tracker.get_value()),
                                       add_vertex_dots=add_vertex_dots,
                                       line_color=color,
                                       vertex_dot_style=dict(fill_color=color),
                                       stroke_opacity=0.8)
            graph.add_updater(make_updater(func, color))

            graphs.add(graph)

        self.add(ax, graphs)
        self.play(theta_tracker.animate.set_value(6. * TAU),
                  run_time=30., rate_func=rate_functions.linear)
