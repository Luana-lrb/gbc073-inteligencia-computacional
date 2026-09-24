import math
import torch

LAST_FACTOR = 0.50


def ativacao(x):
    return torch.tanh(x)


@torch.no_grad()
def inicializar(W, b, fan_in, fan_out, camada, n_camadas):
    # Xavier/Glorot (Glorot & Bengio, 2010) — feita sob medida para ativações
    # SATURANTES como tanh/sigmoid. Diferente da técnica de Monte Carlo
    # (E[f(z)^2] com z~N(0,1)), que só é válida para ativações homogêneas
    # (ReLU, leaky ReLU: f(c·z)=c·f(z)), Xavier já balanceia fan_in e fan_out
    # justamente para não empurrar demais a pré-ativação pra saturação logo
    # nas primeiras camadas — o erro que causou a queda em L4/L16/cifar.
    std = math.sqrt(2.0 / (fan_in + fan_out))

    if camada == n_camadas:
        std *= LAST_FACTOR

    W.normal_(0.0, std)
    b.zero_()