def sort_by_values(d: dict, reverse=False) -> dict:
    """
    Sort the dictionary by its values.
    """
    ordered_pairs = sorted(d.items(), key=lambda x: x[1], reverse=reverse)
    return {k: v for k, v in ordered_pairs}
