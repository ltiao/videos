from manim import *

from gpflow.kernels import SquaredExponential


def ppf(p=0.95):
    return - 2. * np.log(1. - p)


class Test(Scene):

    kernel = SquaredExponential()
    p = 0.95

    def construct(self):

        self.ax = Axes(
            x_range=[-0.5, 5., 1.],
            y_range=[-4., 4., 1.],
            x_length=7,
            y_length=5,
            # x_axis_config={
            # "numbers_to_include": np.arange(0, 5.1, 1),
            # "numbers_with_elongated_ticks": np.arange(-10, 10.01, 2),
            # },
            tips=False,
        ).to_corner(UL)
        label_ax1 = self.ax.get_axis_labels(y_label=r"f(x)")

        self.ax2 = Axes(
            x_range=[-4, 4., 1.],
            y_range=[-4, 4., 1.],
            x_length=5,
            y_length=5,
            tips=False,
        ).next_to(self.ax, RIGHT).to_edge(RIGHT)
        label_ax2 = self.ax2.get_axis_labels(x_label=r"f(x_1)", y_label=r"f(x_2)")

        # print("HHELLLO", tracker.get_value(), tracker.points, self.ax2.point_to_coords(tracker.points[0]))

        # initial values
        x1, y1 = 0.5, 1.
        x2, y2 = 3.5, 2.

        dot = Dot().add_updater(
            lambda x: x.move_to(self.ax2.coords_to_point(tracker.get_value().real,
                                                         tracker.get_value().imag))
        )

        lines = self.ax2.get_lines_to_point(self.ax2.coords_to_point(y1, y2))
        lines.add_updater(
            lambda m: m.become(self.ax2.get_lines_to_point(
                self.ax2.coords_to_point(tracker.get_value().real, 
                                         tracker.get_value().imag))))

        curve = self.create_parametric_ellipse(x1, x2)
        curve.add_updater(lambda m: m.become(self.create_parametric_ellipse(x1_tracker.get_value(), x2)))

        self.add(curve)
        self.add(dot, lines)

        tracker = ComplexValueTracker(y1 + y2 * 1j)

        x1_line = self.create_vertical_line(x1, y1)
        x1_line.add_updater(lambda m: m.become(self.create_vertical_line(x1_tracker.get_value(), tracker.get_value().real)))

        label1 = MathTex("x_1").add_updater(lambda m: m.next_to(x1_line, UP))

        x2_line = self.create_vertical_line(x2, y2)
        x2_line.add_updater(lambda m: m.become(self.create_vertical_line(x2, tracker.get_value().imag)))
        label2 = MathTex("x_2").add_updater(lambda m: m.next_to(x2_line, UP))

        x1_tracker = ValueTracker(x1)

        mat = self.create_covariance_matrix(x1, x2)
        mat.add_updater(lambda m: m.become(self.create_covariance_matrix(x1_tracker.get_value(), x2)))
        self.add(mat)

        # ellipse = self.create_ellipse(x1, x2)
        # ellipse.add_updater(lambda m: m.become(self.create_ellipse(x1_tracker.get_value(), x2)))
        # self.add(ellipse)

        self.add(self.ax, self.ax2, label1, label2, label_ax1, label_ax2, curve, x1_line, x2_line, label1, label2)

        self.play(x1_tracker.animate.set_value(4)),
        self.wait(0.5)
        # self.play(x1_tracker.animate.set_value(3))
        # self.play(x1_tracker.animate.increment_value(-2))
        # self.wait(0.5)
        # self.play(tracker.animate.set_value(2-2j))
        # self.wait()

    def compute_gram_matrix(self, *args):
        X = np.expand_dims(args, axis=-1)
        return self.kernel.K(X).numpy()

    def create_covariance_matrix(self, *args):
        K = self.compute_gram_matrix(*args)
        mat = DecimalMatrix(
            K,
            element_to_mobject_config={"num_decimal_places": 2},
            left_bracket="(", right_bracket=")"
        )
        eq = MathTex(r"\mathrm{cov}(f(x_1), f(x_2)) =").next_to(mat, LEFT, buff=0.2)
        return VGroup(mat, eq).to_corner(DL)

    def create_vertical_line(self, x, y):
        p = self.ax.coords_to_point(x, y)
        return self.ax.get_vertical_line(p)

    def create_ellipse(self, x1, x2):
        s = ppf(self.p)
        K = self.compute_gram_matrix(x1, x2)
        w, v = np.linalg.eigh(K)
        height, width = 2.*np.sqrt(s * w)
        angle = np.arctan2(v[-1, 1], v[-1, 0])
        e = Ellipse(width, height, color=TEAL, fill_opacity=0.2) \
            .rotate(angle) \
            .to_corner(DOWN + RIGHT)
        return e

    def create_parametric_ellipse(self, x1, x2):
        s = ppf(self.p)
        K = self.compute_gram_matrix(x1, x2)
        w, v = np.linalg.eigh(K)

        U = np.sqrt(s * w) * v

        def func(t):
            z = np.vstack((np.cos(t), np.sin(t)))
            a = U @ z
            return np.append(a, 0., axis=None)  # [a[0], a[1], 0]

        curve = self.ax2.plot_parametric_curve(func,
                                               t_range=np.array([0, TAU]), 
                                               fill_opacity=0.3) \
            .set_color(TEAL)
        return curve
