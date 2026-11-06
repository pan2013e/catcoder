use mathru::analysis::interpolation::spline::{CubicSpline, CubicSplineConstraint};
use plotters::prelude::*;

fn main() {
    let x = vec![-2.0, 0.0, 1.0];
    let y = vec![6.0, 2.0, 4.0];
    let cubic_spline = CubicSpline::interpolate(&x, &y, CubicSplineConstraint::Periodic);

    let x_start: f64 = -2.0;
    let x_end: f64 = 1.0;
    let length: usize = 500;

    let mut graph_1: Vec<(f64, f64)> = Vec::with_capacity(length);

    for i in 0..length {
        let x: f64 = (x_end - x_start) / (length as f64) * (i as f64) + x_start;
        graph_1.push((x, cubic_spline.eval(x)));
    }

    let root_area =
        BitMapBackend::new("./figures/periodic_cubic_spline.png", (400, 400)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    let mut ctx = ChartBuilder::on(&root_area)
        .margin(20)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption("Periodic cubic spline interpolation", ("Arial", 25))
        .build_cartesian_2d(x_start - 0.5..x_end + 0.5, 0.0f64..7.5f64)
        .unwrap();

    ctx.configure_mesh()
        .x_desc("x")
        .y_desc("y")
        .axis_desc_style(("sans-serif", 15).into_font())
        .draw()
        .unwrap();

    ctx.draw_series(LineSeries::new(graph_1, BLUE)).unwrap();

    let points: Vec<(f64, f64)> = x.into_iter().zip(y).map(|(x, y)| (x, y)).collect();
    ctx.draw_series(
        points
            .iter()
            .map(|point| Circle::new(*point, 2, BLUE.filled())),
    )
    .unwrap();
}
