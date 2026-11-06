use mathru::{
    algebra::linear::{matrix::General, vector::Vector},
    analysis::differential_equation::ordinary::{
        solver::implicit::runge_kutta::{ImplicitEuler, ImplicitFixedStepper},
        ImplicitInitialValueProblemBuilder, ImplicitODE,
    },
    elementary::Trigonometry,
    {matrix, vector},
};
use plotters::prelude::*;

fn main() {
    pub struct Euler {
        i1: f64,
        i2: f64,
        i3: f64,
    }
    /// ```math
    /// y_{1}^{'}(x) = (I_{2} - I_{3})/I_{1} * y_{2}(x) * y_{3}(x)
    /// y_{2}^{'}(x) = (I_{3} - I_{1})/I_{2} * y_{3}(x) * y_{1}(x)
    /// y_{3}^{'}(x) = (I_{1} - I_{2})/I_{3} * y_{1}(x) * y_{2}(x) + f(x)
    /// ```
    impl ImplicitODE<f64> for Euler {
        fn ode(&self, x: &f64, y: &Vector<f64>) -> Vector<f64> {
            let a: f64 = (self.i2 - self.i3) / self.i1;
            let b: f64 = (self.i3 - self.i1) / self.i2;
            let c: f64 = (self.i1 - self.i2) / self.i3;

            let y_1s: f64 = a * (y[1] * y[2]);
            let y_2s: f64 = b * (y[2] * y[0]);

            let f: f64 = if *x >= 3.0 * f64::pi() && *x <= 4.0 * f64::pi() {
                0.25 * x.sin() * x.sin()
            } else {
                0.0
            };

            let y_3s: f64 = c * (y[0] * y[1]) + f;
            vector![y_1s; y_2s; y_3s]
        }

        fn jacobian(&self, _x: &f64, y: &Vector<f64>) -> General<f64> {
            let a: f64 = (self.i2 - self.i3) / self.i1;
            let b: f64 = (self.i3 - self.i1) / self.i2;
            let c: f64 = (self.i1 - self.i2) / self.i3;

            matrix![0.0, a * y[2], a * y[1];
                        b * y[2], 0.0, b * y[0];
                        c * y[1], c * y[0], 0.0]
        }
    }
    let implicit_euler = Euler {
        i1: 0.5,
        i2: 2.0,
        i3: 3.0,
    };

    let x_start: f64 = 0.0;
    let x_end: f64 = 20.0;

    let problem = ImplicitInitialValueProblemBuilder::<f64, Euler>::new(
        &implicit_euler,
        x_start,
        vector![1.0; 0.0; 0.9],
    )
    .t_end(x_end)
    .build();

    let solver: ImplicitFixedStepper<f64> = ImplicitFixedStepper::new(0.0001);

    let (x, y): (Vec<f64>, Vec<Vector<f64>>) =
        solver.solve(&problem, &ImplicitEuler::default()).unwrap();

    //Create chart
    let mut graph_x1: Vec<(f64, f64)> = Vec::with_capacity(x.len());
    let mut graph_x2: Vec<(f64, f64)> = Vec::with_capacity(x.len());
    let mut graph_x3: Vec<(f64, f64)> = Vec::with_capacity(x.len());

    for i in 0..y.len() {
        let x_i = x[i];

        graph_x1.push((x_i, y[i][0]));
        graph_x2.push((x_i, y[i][1]));
        graph_x3.push((x_i, y[i][2]));
    }

    let root_area =
        BitMapBackend::new("./figures/ode_implicit_euler.png", (1200, 800)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    let mut ctx = ChartBuilder::on(&root_area)
        .margin(20)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption("ODE solved with implicit Euler", ("Arial", 40))
        .build_cartesian_2d(x_start..x_end, -1.0f64..1.5f64)
        .unwrap();

    ctx.configure_mesh()
        .x_desc("Time t")
        .axis_desc_style(("sans-serif", 25).into_font())
        .draw()
        .unwrap();

    ctx.draw_series(LineSeries::new(graph_x1, &BLACK)).unwrap();

    ctx.draw_series(LineSeries::new(graph_x2, &RED)).unwrap();

    ctx.draw_series(LineSeries::new(graph_x3, &BLUE)).unwrap();
}
