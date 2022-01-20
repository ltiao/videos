import tensorflow as tf
import tensorflow_probability as tfp

from scipy.stats import expon
from videos.linalg import safe_cholesky
from manim import *

# shortcuts
tfd = tfp.distributions
kernels = tfp.math.psd_kernels


def default_float():
    return "float64"


class State:

    def __init__(self, kernel, x_grid, xa, xb_tracker, ci=.95):
        self.kernel = kernel
        self.x_grid = x_grid  # shape (K, 1)
        self.xa = xa  # shape ()
        self.xb_tracker = xb_tracker
        self.ci = ci
        # cholesky decomposition of gram matrix over grid points; shape (K, K)
        self.scale_grid = safe_cholesky(self.kernel.matrix(x_grid, x_grid))

    def index_points(self):
        return np.vstack([self.xa, self.xb_tracker.get_value()])  # shape (2, 1)

    def scale(self):
        xs = self.index_points()  # shape (2, 1)
        Ks = self.kernel.matrix(xs, xs)  # shape (2, 2)
        Ks_grid = self.kernel.matrix(self.x_grid, xs)  # shape (K, 2)
        K_col = tf.concat([Ks_grid, Ks], axis=0)  # shape (K+2, 2)
        L = tfp.math.cholesky_concat(self.scale_grid, K_col)  # shape (K+2, K+2)
        return tf.linalg.LinearOperatorLowerTriangular(L)

    def _ellipse_parametric(self, t):
        xs = self.index_points()  # shape (2, 1)
        Ks = self.kernel.matrix(xs, xs)  # shape (2, 2)

        # compute 95% confidence interval using inverse cdf of 
        # chi-squared distribution with 2 degrees of freedom
        s = expon(scale=2.).ppf(q=self.ci)

        w, v = tf.linalg.eigh(Ks)
        U = tf.sqrt(s * w) * v

        z = tf.stack((tf.cos(t), tf.sin(t)), axis=-1)
        a = tf.matmul(U, tf.expand_dims(z, axis=-1)).numpy()
        return (*a, 0)

    def plot_ellipse(self, ax):
        return ax.plot_parametric_curve(self._ellipse_parametric,
                                        t_range=(0, TAU), 
                                        fill_opacity=.25) \
            .set_color(TEAL)


class SampleTrajectory:

    def __init__(self, state, theta_tracker, random_state):
        m = len(state.x_grid)
        self.u = random_state.randn(m+2)
        self.v = random_state.randn(m+2)
        self.state = state
        self.theta_tracker = theta_tracker

    def __call__(self, theta):
        v_norm = np.linalg.norm(self.v, axis=None, ord=2)
        v_normed = np.true_divide(self.v, v_norm)

        c = np.sum(self.u * v_normed, axis=None)

        t = self.u - c * v_normed
        t_norm = np.linalg.norm(t, ord=2, axis=None)
        t_normed = np.true_divide(t, t_norm)

        eps = v_norm * (v_normed * np.cos(theta) + t_normed * np.sin(theta))
        return self.state.scale().matmul(tf.expand_dims(eps, axis=-1)).numpy()

    def make_updater(self, ax, color, make_line_graph_fn):
        def updater(m):
            foo = self(self.theta_tracker.get_value())
            y_values = foo[:-2]
            return m.become(make_line_graph_fn(ax, self.state.x_grid, y_values, color))
        return updater

    def dot_updater(self, ax):
        def updater(m):
            foo = self(self.theta_tracker.get_value())
            y1, y2 = foo[-2:]
            return m.move_to(ax.coords_to_point(y1, y2))
        return updater

    def make_xa_updater(self, ax):
        def updater(m):
            foo = self(self.theta_tracker.get_value())
            x = self.state.xa
            y = foo[-2]
            return m.move_to(ax.coords_to_point(x, y))
        return updater

    def make_xb_updater(self, ax):
        def updater(m):
            foo = self(self.theta_tracker.get_value())
            x = self.state.xb_tracker.get_value()
            y = foo[-1]
            return m.move_to(ax.coords_to_point(x, y))
        return updater

    def tset(self, ya, yb, ax, z_index, color):
        return ax.get_lines_to_point(ax.coords_to_point(ya, yb), color=color) \
            .set_z_index(z_index)

    def make_lines_updater(self, ax, z_index, color):
        def updater(m):
            foo = self(self.theta_tracker.get_value())
            ya, yb = foo[-2:]
            return m.become(self.tset(ya, yb, ax, z_index, color))
        return updater


class GaussianProcessSamples(Scene):

    def make_line_graph(self, ax, x, y, color):
        x_values = x.squeeze(axis=-1)
        return ax.plot_line_graph(x_values=x_values,
                                  y_values=y,
                                  add_vertex_dots=False,
                                  line_color=color,
                                  # vertex_dot_style=dict(fill_color=color,
                                  #                       fill_opacity=0.8),
                                  stroke_opacity=0.9)

    def construct(self):

        # self.camera.background_color = WHITE

        seed = 23
        random_state = np.random.RandomState(seed)

        # colors = [BLUE, TEAL, GREEN, GOLD, RED, MAROON, PURPLE]
        colors = [RED, GREEN, BLUE]
        n_samples = len(colors)

        n_index_points = 512  # nbr of index points
        n_foo = 2

        y_min, y_max, y_step = -3.2, 3.2, .8
        x_min, x_max, x_step = -.1, 1., .1
        X_grid = np.linspace(x_min, x_max, n_index_points).reshape(-1, 1)

        # X_foo = random_state.uniform(low=x_min, high=x_max, size=(n_foo, 1))
        xa = 0.7
        xb = xa - 0.2
        # x2 = random_state.uniform(low=x_min, high=x_max)

        # kernel_cls = kernels.MaternFiveHalves
        kernel_cls = kernels.ExponentiatedQuadratic
        amplitude = 1.
        length_scale = .1

        kernel = kernel_cls(
            amplitude=tf.constant(amplitude, dtype=default_float()), 
            length_scale=tf.constant(length_scale, dtype=default_float())
        )

        # angle
        theta = 0.

        ax1 = Axes(
            x_range=[x_min, x_max, x_step],
            y_range=[y_min, y_max, y_step],
            x_length=7.,
            y_length=4.,
            tips=False,
        )
        ax2 = Axes(
            x_range=[y_min, y_max, y_step],
            y_range=[y_min, y_max, y_step],
            x_length=4.,
            y_length=4.,
            tips=False,
        )
        axes = VGroup(ax1, ax2).arrange(RIGHT, buff=LARGE_BUFF)

        ax1_label = ax1.get_axis_labels(y_label=r"f(x)")
        ax2_label = ax2.get_axis_labels(x_label=r"f(x_1)", y_label=r"f(x_2)")
        labels = VGroup(ax1_label, ax2_label)

        xb_tracker = ValueTracker(xb)
        length_scale_tracker = ValueTracker(length_scale)
        theta_tracker = ValueTracker(theta)

        state = State(kernel, X_grid, xa, xb_tracker)

        curve = state.plot_ellipse(ax2)
        curve.add_updater(lambda m: m.become(state.plot_ellipse(ax2)))

        graphs = VGroup()
        lines = VGroup()
        dots = VGroup()
        for i, color in enumerate(colors):
            traj = SampleTrajectory(state, theta_tracker, random_state)

            foo = traj(theta_tracker.get_value())
            *y_values, ya, yb = foo

            graph = self.make_line_graph(ax1, X_grid, y_values, color) \
                .set_z_index(i+1)
            graph.add_updater(traj.make_updater(ax1, color, self.make_line_graph))
            graphs.add(graph)

            dot_xa = Dot(ax1.coords_to_point(xa, ya),
                         fill_color=color, fill_opacity=0.9, stroke_width=1.5) \
                .set_z_index(i+1)
            dot_xa.add_updater(traj.make_xa_updater(ax1))

            dot_xb = Dot(ax1.coords_to_point(xb_tracker.get_value(), yb),
                         fill_color=color, fill_opacity=0.9, stroke_width=1.5) \
                .set_z_index(i+1)
            dot_xb.add_updater(traj.make_xb_updater(ax1))

            dot = Dot(ax2.coords_to_point(ya, yb),
                      fill_color=color, stroke_width=1.5) \
                .set_z_index(curve.z_index+i+1)
            dot.add_updater(traj.dot_updater(ax2))

            line = traj.tset(ya, yb, ax2, z_index=curve.z_index+i+1, color=color)
            line.add_updater(traj.make_lines_updater(ax2, z_index=curve.z_index+i+1, color=color))

            dots.add(dot, dot_xa, dot_xb)
            lines.add(line)

        line_a = ax1.get_vertical_line(ax1.coords_to_point(xa, .75 * y_min)) 
        line_b = ax1.get_vertical_line(ax1.coords_to_point(xb_tracker.get_value(), .75 * y_max))
        line_b.add_updater(lambda m: m.become(ax1.get_vertical_line(ax1.coords_to_point(xb_tracker.get_value(), .75 * y_max))))
        lines.add(line_a, line_b)

        label_a = MathTex("x_1").next_to(line_a, DOWN)
        label_b = MathTex("x_2").next_to(line_b, UP)
        label_b.add_updater(lambda m: m.next_to(line_b, UP))
        labels.add(label_a, label_b)

        logo = Text("@louistiao", font="Open Sans", font_size=20, color=BLUE_D).to_corner(DR)

        self.add(logo, axes, labels, graphs, dots, curve, lines)

        rotations = 1
        frequency = 1

        self.play(xb_tracker.animate.set_value(xa - 0.45))
        self.wait()
        self.animate_samples(theta_tracker, rotations, frequency)
        self.wait()

        self.next_section()
        self.play(xb_tracker.animate.set_value(xa + 0.2))
        self.wait()
        self.animate_samples(theta_tracker, rotations, frequency)
        self.wait()

        # self.next_section()
        # self.play(xb_tracker.animate.set_value(xa + .015))
        # self.wait()
        # self.animate_samples(theta_tracker, rotations, frequency)
        # self.wait()

        # self.next_section()
        # self.play(xb_tracker.animate.set_value(xb))
        # self.wait()
        # self.animate_samples(theta_tracker, rotations, frequency)
        # self.wait()

    def animate_samples(self, tracker, rotations, frequency,
                        rate_func=rate_functions.linear):
        self.play(tracker.animate.increment_value(rotations * TAU),
                  rate_func=rate_func, run_time=rotations / frequency)
