#[inline]
pub fn entropy(distribution: &[usize]) -> f64 {
    let sum: usize = distribution.iter().sum();
    if sum == 0 {
        return 0.0;
    }

    let mut entropy = 0.0;
    for &count in distribution.iter() {
        if count > 0 {
            let probability = count as f64 / sum as f64;
            entropy -= probability * probability.log2();
        }
    }
    entropy
}

#[inline]
pub fn gini_index(
    parent_distribution: &[usize],
    left_distribution: &[usize],
    right_distribution: &[usize],
    _parent_entropy: f64,
) -> f64 {
    let total_samples = parent_distribution.iter().sum::<usize>();
    if total_samples == 0 {
        return 0.0;
    }

    let total_weight = total_samples as f64;
    let left_weight = left_distribution.iter().sum::<usize>() as f64;
    let right_weight = right_distribution.iter().sum::<usize>() as f64;

    let left_impurity = calculate_branch_impurity(left_distribution, left_weight);
    let right_impurity = calculate_branch_impurity(right_distribution, right_weight);

    ((left_weight * left_impurity) + (right_weight * right_impurity)) / total_weight
}

#[inline]
fn calculate_branch_impurity(distribution: &[usize], total: f64) -> f64 {
    if total < 1.0 {
        return 0.0; // Empty branch has zero impurity
    }

    1.0 - distribution
        .iter()
        .map(|&count| {
            let probability = count as f64 / total;
            probability * probability
        })
        .sum::<f64>()
}

#[inline]
pub fn information_gain(
    parent_distribution: &[usize],
    left_distribution: &[usize],
    right_distribution: &[usize],
    parent_entropy: f64,
) -> f64 {
    let total_count = parent_distribution.iter().sum::<usize>() as f64;
    if total_count < 1.0 {
        return 0.0;
    }

    let left_count = left_distribution.iter().sum::<usize>() as f64;
    let right_count = right_distribution.iter().sum::<usize>() as f64;

    let left_weight = left_count / total_count;
    let right_weight = right_count / total_count;

    let left_entropy = entropy(left_distribution);
    let right_entropy = entropy(right_distribution);
    parent_entropy - (left_weight * left_entropy + right_weight * right_entropy)
}

#[inline]
pub fn weighted_entropy(
    parent_distribution: &[usize],
    left_distribution: &[usize],
    right_distribution: &[usize],
    _parent_entropy: f64,
) -> f64 {
    let total_count = parent_distribution.iter().sum::<usize>() as f64;
    if total_count < 1.0 {
        return f64::INFINITY;
    }

    let left_count = left_distribution.iter().sum::<usize>() as f64;
    let right_count = right_distribution.iter().sum::<usize>() as f64;

    let left_weight = left_count / total_count;
    let right_weight = right_count / total_count;

    if left_weight < f64::EPSILON || right_weight < f64::EPSILON {
        return f64::INFINITY;
    }

    // Calculate entropy for each branch
    let left_entropy = entropy(&left_distribution);
    let right_entropy = entropy(&right_distribution);

    left_weight * left_entropy + right_weight * right_entropy
}
