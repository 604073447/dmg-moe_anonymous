def value_fmt(x, f: str = 'f', decimal: int = 4):
    if isinstance(x, (float, int)):
        return f"{x:.{decimal}{f}}"
    return x
