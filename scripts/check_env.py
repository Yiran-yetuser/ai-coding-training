"""Check the training environment without creating exercises or model files."""

import platform
import sys
from importlib.metadata import version

import numpy as np
import torch


def check_device(device: str) -> None:
    x = torch.ones((2, 3), device=device, requires_grad=True)
    weights = torch.ones((3, 2), device=device)
    output = x @ weights
    torch.testing.assert_close(output, torch.full((2, 2), 3.0, device=device))
    output.sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 2.0))
    print(f"{device}: tensor operations and autograd OK")


def main() -> None:
    print(f"Python: {platform.python_version()} ({platform.machine()})")
    print(f"Interpreter: {sys.executable}")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            "Use the project Python 3.12 environment: uv run python scripts/check_env.py"
        )

    for package in ("numpy", "torch", "scikit-learn", "matplotlib", "pytest", "ruff"):
        print(f"{package}: {version(package)}")

    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    tensor = torch.from_numpy(array)
    np.testing.assert_array_equal(tensor.numpy(), array)
    print("NumPy/PyTorch conversion: OK")
    check_device("cpu")

    mps_available = torch.backends.mps.is_available()
    print(f"Apple MPS available: {mps_available}")
    if mps_available:
        check_device("mps")
    print("Environment ready.")


if __name__ == "__main__":
    main()
