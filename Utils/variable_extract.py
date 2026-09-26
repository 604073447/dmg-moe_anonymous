def get_unique_top_k_levels(paths, k=1):
    unique_prefixes = set()
    for p in paths:
        parts = p.split('.')
        top_k_parts = parts[:k]
        prefix = ".".join(top_k_parts)
        unique_prefixes.add(prefix)
    return sorted(list(unique_prefixes))