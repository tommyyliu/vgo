use vgo_core::Color;

use super::ChildSummary;

pub(super) fn preferred_child_indices(children: &[ChildSummary], to_move: Color) -> Vec<usize> {
    let perspective = if to_move == Color::Black { 1.0 } else { -1.0 };
    let mut indices = (0..children.len()).collect::<Vec<_>>();
    indices.sort_by(|left, right| {
        let left = *left;
        let right = *right;
        children[right]
            .visits
            .cmp(&children[left].visits)
            .then_with(|| {
                (perspective * children[right].black_value)
                    .total_cmp(&(perspective * children[left].black_value))
            })
            .then_with(|| left.cmp(&right))
    });
    indices
}

/// Order children by sampling without replacement from `visits^(1 / temperature)`.
///
/// Only children with visits participate in the draw; unvisited children keep
/// their deterministic ordering at the tail, so a caller falling back through the
/// list still degrades to visit order rather than to noise. `temperature <= 0`
/// never reaches here.
pub(super) fn sampled_child_indices(
    children: &[ChildSummary],
    to_move: Color,
    temperature: f64,
    rng_state: u64,
) -> Vec<usize> {
    let deterministic = preferred_child_indices(children, to_move);
    let exponent = 1.0 / temperature;
    // Scale by the maximum visit count before exponentiating: visits^(1/t) with a
    // small temperature overflows f64 for even modest visit counts (128^100), and
    // the distribution is unchanged by dividing through by the max first.
    let maximum = children
        .iter()
        .map(|child| child.visits)
        .max()
        .unwrap_or(0)
        .max(1);
    let mut pool = deterministic
        .iter()
        .copied()
        .filter(|&index| children[index].visits > 0)
        .map(|index| {
            let share = f64::from(children[index].visits) / f64::from(maximum);
            (index, share.powf(exponent))
        })
        .collect::<Vec<_>>();

    let mut rng = rng_state;
    let mut next = || {
        rng = crate::candidates::splitmix64(rng);
        crate::candidates::unit_f64(rng)
    };

    let mut order = Vec::with_capacity(children.len());
    while !pool.is_empty() {
        let total: f64 = pool.iter().map(|&(_, weight)| weight).sum();
        if total <= 0.0 || !total.is_finite() {
            // Degenerate weights (all zero, or non-finite): fall back to the
            // deterministic order for whatever remains.
            order.extend(pool.iter().map(|&(index, _)| index));
            break;
        }
        let mut target = next() * total;
        let mut chosen = pool.len() - 1;
        for (position, &(_, weight)) in pool.iter().enumerate() {
            target -= weight;
            if target < 0.0 {
                chosen = position;
                break;
            }
        }
        order.push(pool.remove(chosen).0);
    }
    // Unvisited children, in deterministic order, after every visited one.
    order.extend(
        deterministic
            .into_iter()
            .filter(|&index| children[index].visits == 0),
    );
    order
}
