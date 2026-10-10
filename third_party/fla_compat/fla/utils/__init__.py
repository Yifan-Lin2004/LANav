"""The only FLA helper imported by the bundled NVIDIA kernels."""

import functools
import torch


def contiguous(fn):
    """Make tensor arguments contiguous before calling a decorated kernel."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        new_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                new_args.append(arg.contiguous() if not arg.is_contiguous() else arg)
            else:
                new_args.append(arg)

        new_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                new_kwargs[k] = v.contiguous() if not v.is_contiguous() else v
            else:
                new_kwargs[k] = v

        return fn(*new_args, **new_kwargs)

    return wrapper


__all__ = ["contiguous"]
