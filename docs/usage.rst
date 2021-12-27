=====
Usage
=====

To use Videos in a project::

    import videos


.. manim:: Subplots

    class Subplots(Scene):

        def construct(self):

            x_min, x_max = -3., 3.

            axes1 = Axes(
                x_range=[x_min, x_max, 1.],
                y_range=[-1., 10., 1.],
                x_length=10,
                tips=False,
            )
            axes_labels = axes1.get_axis_labels()

            graph1 = axes1.get_graph(lambda x: x**2, color=BLUE)
            label1 = axes1.get_graph_label(
                graph1, r"f(x)", x_val=2., direction=RIGHT
            )

            axes2 = Axes(
                x_range=[x_min, x_max, 1.],
                y_range=[-1., 10., 1.],
                x_length=10,
                tips=False,
            )
            graph2 = axes2.get_graph(np.cos, color=RED)

            group1 = VGroup(axes1, axes_labels, graph1, label1)  # .scale(0.6).to_corner(UP + LEFT)
            group2 = VGroup(axes2, graph2).scale(0.6).to_corner(DOWN + RIGHT)

            # self.add(group1, group2)

            self.play(Create(group1))
            self.play(group1.animate.scale(0.6))
            self.play(group1.animate.to_corner(DOWN + LEFT))
            self.wait()
            self.play(Create(group2))
            self.wait()


.. manim:: ManimCELogo
    :save_last_frame:
    :ref_classes: MathTex Circle Square Triangle

    class ManimCELogo(Scene):
        def construct(self):
            self.camera.background_color = "#ece6e2"
            logo_green = "#87c2a5"
            logo_blue = "#525893"
            logo_red = "#e07a5f"
            logo_black = "#343434"
            ds_m = MathTex(r"\mathbb{M}", fill_color=logo_black).scale(7)
            ds_m.shift(2.25 * LEFT + 1.5 * UP)
            circle = Circle(color=logo_green, fill_opacity=1).shift(LEFT)
            square = Square(color=logo_blue, fill_opacity=1).shift(UP)
            triangle = Triangle(color=logo_red, fill_opacity=1).shift(RIGHT)
            logo = VGroup(triangle, square, circle, ds_m)  # order matters
            logo.move_to(ORIGIN)
            self.add(logo)

.. manim:: PointWithTrace
    :ref_classes: Rotating
    :ref_methods: VMobject.set_points_as_corners Mobject.add_updater

    class PointWithTrace(Scene):
        def construct(self):
            path = VMobject()
            dot = Dot()
            path.set_points_as_corners([dot.get_center(), dot.get_center()])
            def update_path(path):
                previous_path = path.copy()
                previous_path.add_points_as_corners([dot.get_center()])
                path.become(previous_path)
            path.add_updater(update_path)
            self.add(path, dot)
            self.play(Rotating(dot, radians=PI, about_point=RIGHT, run_time=2))
            self.wait()
            self.play(dot.animate.shift(UP))
            self.play(dot.animate.shift(LEFT))
            self.wait()


.. manim:: PolarPlaneExample
    :save_last_frame:

    class PolarPlaneExample(Scene):
        def construct(self):
            polarplane_pi = PolarPlane(
                radius_max=2.,
                radius_step=.5,
                azimuth_units="PI radians",
                size=6,
                azimuth_label_font_size=33.6,
                radius_config={"font_size": 33.6},
            ).add_coordinates()
            func = lambda t: 1. + np.cos(t) * np.square(np.sin(t))
            polar_graph = polarplane_pi.plot_polar_graph(func, color=GREEN)
            self.add(polarplane_pi, polar_graph)
