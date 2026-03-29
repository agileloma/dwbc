 uv run play Mjlab-Walk-Flat-B2Z1 --agent zero

 uv run play Mjlab-Walk-Flat-B2Z1 --agent random

 uv run play Mjlab-Walk-Flat-B2Z1 --wandb-run-path 

 uv run play Mjlab-Walk-Flat-B2Z1 --checkpoint-file

 uv run train Mjlab-Walk-Flat-B2Z1   --env.scene.num-envs 4096   --agent.max-iterations 15000



dwbc/.venv/lib/python3.13/site-packages/mjlab/utils/lab_api/math.py
@torch.jit.script
def quat_unique(quat: torch.Tensor) -> torch.Tensor:
    """Ensure the quaternion has a positive real part.

    Quaternions ``q`` and ``-q`` represent the same rotation. This function
    flips the sign of the quaternion if the real part is negative.

    Args:
        quat: The input quaternion in (w, x, y, z). Shape is (..., 4).

    Returns:
        Quaternion with non‑negative real part. Same shape.
    """
    shape = quat.shape
    quat = quat.reshape(-1, 4)
    # condition: w < 0
    mask = quat[:, 0] < 0
    quat[mask] = -quat[mask]
    return quat.view(shape)

@torch.jit.script
def quat_conj(quat: torch.Tensor) -> torch.Tensor:
    """Compute the conjugate of a quaternion.

    The conjugate of a quaternion :math:`q = (w, x, y, z)` is :math:`q^* = (w, -x, -y, -z)`.
    It represents the inverse rotation when the quaternion is normalized.

    Args:
        quat: The input quaternion in (w, x, y, z). Shape is (..., 4).

    Returns:
        The conjugate quaternion in (w, x, y, z). Same shape as input.
    """
    # store shape
    shape = quat.shape
    # reshape to (N, 4) for operation
    quat = quat.reshape(-1, 4)
    # conjugate: (w, -x, -y, -z)
    conj = torch.cat([quat[:, :1], -quat[:, 1:]], dim=-1)
    return conj.view(shape)




