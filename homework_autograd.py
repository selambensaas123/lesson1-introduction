"""Урок 1, задание 2: автоматическое дифференцирование PyTorch."""

from __future__ import annotations

import math
import torch


def three_variable_gradients(
    x_value: float = 1.5, y_value: float = -2.0, z_value: float = 0.5
) -> dict[str, object]:
    """Найти градиенты f=x²+y²+z²+2xyz и сравнить с аналитикой."""
    x = torch.tensor(float(x_value), requires_grad=True)
    y = torch.tensor(float(y_value), requires_grad=True)
    z = torch.tensor(float(z_value), requires_grad=True)
    f = x.square() + y.square() + z.square() + 2 * x * y * z
    f.backward()

    # ∂f/∂x=2x+2yz, ∂f/∂y=2y+2xz, ∂f/∂z=2z+2xy
    analytical = {
        "x": 2 * x_value + 2 * y_value * z_value,
        "y": 2 * y_value + 2 * x_value * z_value,
        "z": 2 * z_value + 2 * x_value * y_value,
    }
    autograd = {"x": x.grad.item(), "y": y.grad.item(), "z": z.grad.item()}
    for variable in analytical:
        assert math.isclose(autograd[variable], analytical[variable], rel_tol=1e-5, abs_tol=1e-6)
    return {"f": f.item(), "autograd": autograd, "analytical": analytical}


def mse_gradients() -> dict[str, object]:
    """Найти производные MSE по весу w и смещению b, проверить вручную."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y_true = torch.tensor([3.0, 5.0, 7.0, 9.0])
    w = torch.tensor(1.5, requires_grad=True)
    b = torch.tensor(0.5, requires_grad=True)

    prediction = w * x + b
    errors = prediction - y_true
    loss = torch.mean(errors.square())
    loss.backward()

    # dMSE/dw = (2/n)Σ[(w*x_i+b-y_i)*x_i]
    # dMSE/db = (2/n)Σ[w*x_i+b-y_i]
    analytical_w = (2 * torch.mean(errors.detach() * x)).item()
    analytical_b = (2 * torch.mean(errors.detach())).item()
    assert math.isclose(w.grad.item(), analytical_w, abs_tol=1e-6)
    assert math.isclose(b.grad.item(), analytical_b, abs_tol=1e-6)
    return {
        "mse": loss.item(),
        "autograd": {"w": w.grad.item(), "b": b.grad.item()},
        "analytical": {"w": analytical_w, "b": analytical_b},
    }


def chain_rule_gradient(x_value: float = 0.7) -> dict[str, float]:
    """Найти df/dx для f=sin(x²+1); проверить backward и autograd.grad."""
    x = torch.tensor(float(x_value), dtype=torch.float64, requires_grad=True)
    f = torch.sin(x.square() + 1)
    derivative = torch.autograd.grad(f, x, retain_graph=True)[0]
    f.backward()
    analytical = 2 * x_value * math.cos(x_value**2 + 1)
    assert math.isclose(derivative.item(), analytical, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(x.grad.item(), analytical, rel_tol=1e-12, abs_tol=1e-12)
    return {
        "f": f.item(),
        "autograd_grad": derivative.item(),
        "backward": x.grad.item(),
        "analytical": analytical,
    }


def main() -> None:
    """Запустить три эксперимента с градиентами."""
    print("=== 2.1 f(x,y,z) и аналитические градиенты ===")
    print(three_variable_gradients())
    print("\n=== 2.2 Градиенты MSE по w и b ===")
    print(mse_gradients())
    print("\n=== 2.3 Цепное правило для sin(x²+1) ===")
    print(chain_rule_gradient())
    print("\nВсе градиенты совпадают с аналитическими значениями.")


if __name__ == "__main__":
    main()
