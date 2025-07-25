def derived_max(x_vals, y_vals):
    max_idx = x_vals.index(max(x_vals))
    return y_vals[max_idx]

def derived_min(x_vals, y_vals):
    min_idx = x_vals.index(min(x_vals))
    return y_vals[min_idx]

def derived_adjustedmin(x_vals, y_vals):
    # Filter out negative x_vals
    filtered = [(x, y) for x, y in zip(x_vals, y_vals) if x > 0]
    if not filtered:
        return 0  # fallback
    sorted_pairs = sorted(filtered)
    return sorted_pairs[0][1]

def derived_mean(x_vals, y_vals):
    # Sort by x_vals
    sorted_pairs = sorted(zip(x_vals, y_vals))
    median_idx = len(sorted_pairs) >> 1
    return sorted_pairs[median_idx][1]

def derived_adjustedmean(x_vals, y_vals):
    # Filter out negative x_vals
    filtered = [(x, y) for x, y in zip(x_vals, y_vals) if x > 0]
    if not filtered:
        return 0  # fallback
    sorted_pairs = sorted(filtered)
    median_idx = len(sorted_pairs)  >> 1
    return sorted_pairs[median_idx][1]

__all__ = ['derived_max', 'derived_min', 'derived_adjustedmin', 'derived_mean', 'derived_adjustedmean']