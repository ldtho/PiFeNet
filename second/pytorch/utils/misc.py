import torch
import numpy as np


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


# def unravel_indices(
#         indices: torch.LongTensor,
#         shape: tuple[int, ...],
# ) -> torch.LongTensor:
#     r"""Converts flat indices into unraveled coordinates in a target shape.
# 
#     Args:
#         indices: A tensor of (flat) indices, (*, N).
#         shape: The targeted shape, (D,).
# 
#     Returns:
#         The unraveled coordinates, (*, N, D).
#     """
# 
#     coord = []
# 
#     for dim in reversed(shape):
#         coord.append(indices % dim)
#         indices = indices // dim
# 
#     coord = torch.stack(coord[::-1], dim=-1)
# 
#     return coord


# def unravel_index(
#         indices: torch.LongTensor,
#         shape: tuple[int, ...],
# ) -> tuple[torch.LongTensor, ...]:
#     r"""
#     https://github.com/pytorch/pytorch/issues/35674
#     Converts flat indices into unraveled coordinates in a target shape.
#
#     This is a `torch` implementation of `numpy.unravel_index`.
#
#     Args:
#         indices: A tensor of (flat) indices, (N,).
#         shape: The targeted shape, (D,).
#
#     Returns:
#         A tuple of unraveled coordinate tensors of shape (D,).
#     """
#
#     coord = unravel_indices(indices, shape)
#     return coord


def angle2class(angle, num_angle_bin):
    """Convert continuous angle to discrete class
    [optinal] also small regression number from
    class center angle to current angle.

    angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
    returns class [0,1,...,N-1] and a residual number such that
        class*(2pi/N) + number = angle
    """
    num_class = num_angle_bin
    angle = angle % (2 * np.pi)
    assert angle >= 0 and angle <= 2 * np.pi
    angle_per_class = 2 * np.pi / float(num_class)
    shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
    class_id = int(shifted_angle / angle_per_class)
    residual_angle = shifted_angle - (
            class_id * angle_per_class + angle_per_class / 2
    )
    return class_id, residual_angle


if __name__ == '__main__':
    print(angle2class(1 * np.pi, 12))
