def to_arrow(x, *args, **kwargs):
    return x


def to_pandas(x, *args, **kwargs):
    try:
        return x.to_pandas()
    except AttributeError:
        return x
