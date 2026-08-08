use crate::{Position, numeric};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    #[must_use]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    #[must_use]
    pub fn distance(self, other: Self) -> f64 {
        numeric::length(self.x - other.x, self.y - other.y)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BoardSide {
    Left,
    Right,
    Top,
    Bottom,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EdgeSource {
    Board(BoardSide),
    Bisector { stone: usize, other: usize },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Edge {
    pub start: Point,
    pub end: Point,
    pub source: Option<EdgeSource>,
}

impl Edge {
    #[must_use]
    pub fn other_stone(&self) -> Option<usize> {
        match self.source {
            Some(EdgeSource::Bisector { other, .. }) => Some(other),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Cell {
    pub polygon: Vec<Point>,
    pub area: f64,
    pub edges: Vec<Edge>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GeometryDiagnostics {
    pub unclassified_edges: usize,
    pub degenerate_edges: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Geometry {
    pub cells: Vec<Cell>,
    pub adjacency: Vec<Vec<usize>>,
    pub groups: Vec<usize>,
    pub diagnostics: GeometryDiagnostics,
}

#[derive(Clone, Copy)]
struct Constraint {
    nx: f64,
    ny: f64,
    offset: f64,
    source: EdgeSource,
}

fn distance_squared(a: Point, b: Point) -> f64 {
    (a.x - b.x).mul_add(a.x - b.x, (a.y - b.y) * (a.y - b.y))
}

fn normalize_polygon(points: Vec<Point>) -> Vec<Point> {
    if points.len() < 3 {
        return points;
    }
    let mut deduped = Vec::with_capacity(points.len());
    for point in points {
        if deduped
            .last()
            .is_none_or(|last| distance_squared(point, *last) > numeric::EDGE_EPSILON.powi(2))
        {
            deduped.push(point);
        }
    }
    if deduped.len() > 1
        && distance_squared(deduped[0], *deduped.last().expect("nonempty"))
            <= numeric::EDGE_EPSILON.powi(2)
    {
        deduped.pop();
    }

    let mut polygon = deduped;
    loop {
        if polygon.len() < 3 {
            return polygon;
        }
        let mut changed = false;
        let mut clean = Vec::with_capacity(polygon.len());
        for index in 0..polygon.len() {
            let a = polygon[(index + polygon.len() - 1) % polygon.len()];
            let b = polygon[index];
            let c = polygon[(index + 1) % polygon.len()];
            let abx = b.x - a.x;
            let aby = b.y - a.y;
            let bcx = c.x - b.x;
            let bcy = c.y - b.y;
            let cross = abx.mul_add(bcy, -aby * bcx);
            let scale = numeric::length(abx, aby) + numeric::length(bcx, bcy);
            if cross.abs() <= numeric::COLLINEAR_EPSILON * scale.max(1.0) {
                changed = true;
            } else {
                clean.push(b);
            }
        }
        polygon = clean;
        if !changed {
            return polygon;
        }
    }
}

fn clip_half_plane(polygon: Vec<Point>, constraint: Constraint) -> Vec<Point> {
    let mut output = Vec::with_capacity(polygon.len() + 1);
    for index in 0..polygon.len() {
        let a = polygon[index];
        let b = polygon[(index + 1) % polygon.len()];
        let fa = constraint.nx.mul_add(a.x, constraint.ny * a.y) - constraint.offset;
        let fb = constraint.nx.mul_add(b.x, constraint.ny * b.y) - constraint.offset;
        let a_inside = fa <= numeric::COORDINATE_EPSILON;
        let b_inside = fb <= numeric::COORDINATE_EPSILON;
        if a_inside {
            output.push(a);
        }
        if a_inside != b_inside {
            let denominator = fa - fb;
            if denominator.abs() > f64::EPSILON {
                let t = fa / denominator;
                output.push(Point::new(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y)));
            }
        }
    }
    normalize_polygon(output)
}

fn polygon_area(polygon: &[Point]) -> f64 {
    let mut twice_area = 0.0;
    for index in 0..polygon.len() {
        let a = polygon[index];
        let b = polygon[(index + 1) % polygon.len()];
        twice_area += a.x.mul_add(b.y, -b.x * a.y);
    }
    twice_area.abs() / 2.0
}

fn board_constraint(side: BoardSide) -> Constraint {
    match side {
        BoardSide::Left => Constraint {
            nx: -1.0,
            ny: 0.0,
            offset: 0.0,
            source: EdgeSource::Board(side),
        },
        BoardSide::Right => Constraint {
            nx: 1.0,
            ny: 0.0,
            offset: 1.0,
            source: EdgeSource::Board(side),
        },
        BoardSide::Top => Constraint {
            nx: 0.0,
            ny: -1.0,
            offset: 0.0,
            source: EdgeSource::Board(side),
        },
        BoardSide::Bottom => Constraint {
            nx: 0.0,
            ny: 1.0,
            offset: 1.0,
            source: EdgeSource::Board(side),
        },
    }
}

fn bisector_constraint(position: &Position, stone: usize, other: usize) -> Constraint {
    let a = position.stones()[stone];
    let b = position.stones()[other];
    let nx = b.x - a.x;
    let ny = b.y - a.y;
    Constraint {
        nx,
        ny,
        offset: nx.mul_add((a.x + b.x) / 2.0, ny * (a.y + b.y) / 2.0),
        source: EdgeSource::Bisector { stone, other },
    }
}

fn residual(constraint: Constraint, point: Point) -> f64 {
    let length = numeric::length(constraint.nx, constraint.ny);
    if length == 0.0 {
        return f64::INFINITY;
    }
    (constraint.nx.mul_add(point.x, constraint.ny * point.y) - constraint.offset).abs() / length
}

fn edge_source(constraints: &[Constraint], start: Point, end: Point) -> Option<EdgeSource> {
    constraints
        .iter()
        .map(|constraint| {
            (
                residual(*constraint, start).max(residual(*constraint, end)),
                constraint.source,
            )
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .filter(|(error, _)| *error <= numeric::COORDINATE_EPSILON * 4.0)
        .map(|(_, source)| source)
}

#[must_use]
pub fn compute(position: &Position) -> Geometry {
    let count = position.stones().len();
    let mut cells = Vec::with_capacity(count);
    let mut adjacency = vec![Vec::new(); count];
    let mut diagnostics = GeometryDiagnostics::default();

    for (stone, neighbors) in adjacency.iter_mut().enumerate() {
        let mut polygon = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(0.0, 1.0),
        ];
        let mut constraints = vec![
            board_constraint(BoardSide::Left),
            board_constraint(BoardSide::Right),
            board_constraint(BoardSide::Top),
            board_constraint(BoardSide::Bottom),
        ];
        for other in 0..count {
            if other == stone || polygon.is_empty() {
                continue;
            }
            let constraint = bisector_constraint(position, stone, other);
            constraints.push(constraint);
            polygon = clip_half_plane(polygon, constraint);
        }
        polygon = normalize_polygon(polygon);
        let mut edges = Vec::with_capacity(polygon.len());
        for index in 0..polygon.len() {
            let start = polygon[index];
            let end = polygon[(index + 1) % polygon.len()];
            if start.distance(end) <= numeric::EDGE_EPSILON {
                diagnostics.degenerate_edges += 1;
                continue;
            }
            let source = edge_source(&constraints, start, end);
            if source.is_none() {
                diagnostics.unclassified_edges += 1;
            }
            if let Some(EdgeSource::Bisector { other, .. }) = source {
                neighbors.push(other);
            }
            edges.push(Edge { start, end, source });
        }
        cells.push(Cell {
            area: polygon_area(&polygon),
            polygon,
            edges,
        });
    }

    for stone in 0..count {
        let neighbors = adjacency[stone].clone();
        for other in neighbors {
            if !adjacency[other].contains(&stone) {
                adjacency[other].push(stone);
            }
        }
    }
    for neighbors in &mut adjacency {
        neighbors.sort_unstable();
        neighbors.dedup();
    }

    let mut parent: Vec<usize> = (0..count).collect();
    fn find(parent: &mut [usize], mut index: usize) -> usize {
        let mut root = index;
        while parent[root] != root {
            root = parent[root];
        }
        while parent[index] != index {
            let next = parent[index];
            parent[index] = root;
            index = next;
        }
        root
    }
    for (stone, neighbors) in adjacency.iter().enumerate() {
        for &other in neighbors {
            if position.stones()[stone].color == position.stones()[other].color {
                let a = find(&mut parent, stone);
                let b = find(&mut parent, other);
                if a != b {
                    parent[a.max(b)] = a.min(b);
                }
            }
        }
    }
    let groups = (0..count).map(|index| find(&mut parent, index)).collect();
    Geometry {
        cells,
        adjacency,
        groups,
        diagnostics,
    }
}

#[cfg(test)]
mod tests {
    use crate::{Color, Position, Stone};

    use super::{EdgeSource, compute};

    #[test]
    fn two_sites_split_the_board_and_retain_provenance() {
        let position = Position::new(
            0.05,
            vec![
                Stone::new(0.25, 0.5, Color::Black),
                Stone::new(0.75, 0.5, Color::White),
            ],
            Color::Black,
        );
        let geometry = compute(&position);
        assert!((geometry.cells[0].area - 0.5).abs() < 1.0e-12);
        assert!((geometry.cells[1].area - 0.5).abs() < 1.0e-12);
        assert_eq!(geometry.adjacency, vec![vec![1], vec![0]]);
        assert!(
            geometry.cells[0]
                .edges
                .iter()
                .any(|edge| { edge.source == Some(EdgeSource::Bisector { stone: 0, other: 1 }) })
        );
        assert_eq!(geometry.diagnostics.unclassified_edges, 0);
    }

    #[test]
    fn point_contacts_do_not_connect_diagonal_groups() {
        let position = Position::new(
            0.05,
            vec![
                Stone::new(0.25, 0.25, Color::Black),
                Stone::new(0.75, 0.25, Color::White),
                Stone::new(0.75, 0.75, Color::Black),
                Stone::new(0.25, 0.75, Color::White),
            ],
            Color::Black,
        );
        let geometry = compute(&position);
        assert!(!geometry.adjacency[0].contains(&2));
        assert_ne!(geometry.groups[0], geometry.groups[2]);
    }
}
