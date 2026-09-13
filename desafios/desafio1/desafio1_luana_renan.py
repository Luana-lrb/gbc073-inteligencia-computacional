"""
desafio1_aluno.py — Desafio 1: Mapa de características para o Perceptron
GBC073 — Inteligência Computacional (FACOM/UFU)


"""
import math
import time
import torch

DIM_MAX = 64
SEMENTES = (0, 1, 2)

# =============================================================================
# >>> SUA SUBMISSÃO — edite apenas esta classe <<<
# =============================================================================
class Submissao:
    DIM_MAX = DIM_MAX

    def fit(self, X: torch.Tensor) -> None:
        d = X.shape[1] # num features
        self.mu = X.mean(0) # media
        self.sd = X.std(0) + 1e-8 # desvio padrao
        Xs = (X - self.mu) / self.sd

        # pares (i, j) só entram se couberem no orçamento
        n_pares = d * (d - 1) // 2 # qnt pares de colunas
        # se d >= 2 e n_pares <= 20, gera combinações de pares (i,j) 
        self.pair_idx = torch.combinations(torch.arange(d), r=2) if d >= 2 and n_pares <= 20 else None
        # numero de linhas de pair_idx = quantidade de pares de colunas (novas features)
        pair_dim = self.pair_idx.shape[0] if self.pair_idx is not None else 0

        
        usado = 2 * d + pair_dim # quantas dimensoes usadas
        self.n_freq = max((self.DIM_MAX - usado) // 2, 1) # quantas frequencia de fourier ainda pode usar
 
        g = torch.Generator().manual_seed(0) # gerador de numero aleatorio
        amostra = Xs[:min(len(Xs), 300)] # pega 300 primeiras amostras de Xs
        dists = torch.pdist(amostra) # distancia entre todos pares da amostra
        sigma_base = dists.median().item() if dists.numel() else 1.0 # mediana das distancias
        sigma_base = sigma_base if sigma_base > 1e-8 else 1.0 

        log_min, log_max = math.log(0.12 * sigma_base), math.log(6.0 * sigma_base) # limites de intervalo
        # sorteia numeros entre 0 e 1
        # transforma os numeros para a escala do intervalo
        # exp converte de volta de log para linear
        escala_por_freq = torch.exp(log_min + torch.rand(self.n_freq, generator=g) * (log_max - log_min))

        # gera pesos aleatorios para fourier
        self.W = torch.randn(d, self.n_freq, generator=g) / escala_por_freq

    def phi(self, X: torch.Tensor) -> torch.Tensor:
        # X com mesmos valores de media e desvio padrao do fit
        Xs = (X - self.mu) / self.sd
        # features
        partes = [Xs, Xs ** 2]

        if self.pair_idx is not None:
            # separa todos pares ([0,1] -> [0], [1])
            i, j = self.pair_idx[:, 0], self.pair_idx[:, 1]
            # calcula produto entre cada par de elementos xi * xj
            partes.append(Xs[:, i] * Xs[:, j])

        # multiplica Xs @ W
        proj = Xs.double() @ self.W.double()
        # calcula cos e sen (fourier)
        partes += [torch.cos(proj).float(), torch.sin(proj).float()]
        # concatena as colunas, formando vetor de features
        return torch.cat(partes, dim=1)


# =============================================================================
# Harness (não edite daqui para baixo)
# =============================================================================
def _luas(n, g):
    t = torch.rand(n // 2, generator=g) * math.pi
    X = torch.cat([torch.stack([torch.cos(t), torch.sin(t)], 1),
                   torch.stack([1 - torch.cos(t), 0.5 - torch.sin(t)], 1)])
    y = torch.cat([torch.zeros(n // 2), torch.ones(n // 2)])
    return X + 0.15 * torch.randn(n, 2, generator=g), y

def _circulos(n, g):
    t = torch.rand(n, generator=g) * 2 * math.pi
    r = torch.where(torch.arange(n) < n // 2, 1.0, 0.45)
    X = torch.stack([r * torch.cos(t), r * torch.sin(t)], 1)
    return X + 0.08 * torch.randn(n, 2, generator=g), (torch.arange(n) >= n // 2).float()

def _xor(n, g):
    X = torch.rand(n, 2, generator=g) * 2 - 1
    y = (X[:, 0] * X[:, 1] < 0).float()
    return X + 0.15 * torch.randn(n, 2, generator=g), y

def _espiral(n, g):
    t = torch.sqrt(torch.rand(n // 2, generator=g)) * 3 * math.pi
    a = torch.stack([t * torch.cos(t), t * torch.sin(t)], 1) / 10
    y = torch.cat([torch.zeros(n // 2), torch.ones(n // 2)])
    return torch.cat([a, -a]) + 0.05 * torch.randn(n, 2, generator=g), y

def _esfera(n, g, d=10):
    X = torch.randn(n, d, generator=g)
    r2 = (X ** 2).sum(1)
    return X, (r2 > r2.median()).float()

TAREFAS = {"xor": lambda g: _xor(600, g), "duas_luas": lambda g: _luas(600, g),
           "circulos": lambda g: _circulos(600, g), "espiral": lambda g: _espiral(800, g),
           "esfera_10d": lambda g: _esfera(800, g)}


@torch.no_grad()
def perceptron_pocket(Z, y, epocas=50, eta=1.0, semente=0):
    Zb = torch.cat([Z, torch.ones(len(Z), 1)], 1)
    yb = 2 * y - 1
    w = torch.zeros(Zb.shape[1]); melhor_w, melhor_acc = w.clone(), -1.0
    g = torch.Generator().manual_seed(semente)
    for _ in range(epocas):
        for i in torch.randperm(len(Zb), generator=g).tolist():
            if yb[i] * (Zb[i] @ w) <= 0:
                w += eta * yb[i] * Zb[i]
        acc = ((Zb @ w) * yb > 0).float().mean().item()
        if acc > melhor_acc:
            melhor_acc, melhor_w = acc, w.clone()
    return melhor_w


def acuracia_balanceada(y, yhat):
    return torch.stack([(yhat[y == c] == c).float().mean() for c in y.unique()]).mean().item()


@torch.no_grad()
def rodar(sub, gerador, semente, checar=True):
    g = torch.Generator().manual_seed(semente)
    X, y = gerador(g)
    idx = torch.randperm(len(X), generator=g); ntr = int(0.6 * len(X))
    Xtr, ytr, Xte, yte = X[idx[:ntr]], y[idx[:ntr]], X[idx[ntr:]], y[idx[ntr:]]

    torch.manual_seed(semente)
    sub.fit(Xtr)
    t0 = time.perf_counter(); Ztr = sub.phi(Xtr); dt = time.perf_counter() - t0
    Zte = sub.phi(Xte)
    if checar:
        d, dl = Xtr.shape[1], Ztr.shape[1]
        assert Ztr.ndim == 2 and Zte.shape[1] == dl, "phi deve devolver (n, d')"
        assert d < dl <= DIM_MAX, f"exige d < d' <= {DIM_MAX}; recebi d={d}, d'={dl}"
        assert torch.isfinite(Ztr).all() and torch.isfinite(Zte).all(), "NaN/Inf na saída de phi"
        assert torch.allclose(sub.phi(Xtr[:20]), Ztr[:20]), "phi não é determinística"
        assert dt * (10_000 / len(Xtr)) < 2.0, "phi lenta demais (limite: 10^4 pontos em 2 s)"

    w = perceptron_pocket(Ztr, ytr, semente=semente)
    yhat = (torch.cat([Zte, torch.ones(len(Zte), 1)], 1) @ w > 0).float()
    return acuracia_balanceada(yte, yhat)


class _Identidade:
    def fit(self, X): pass
    def phi(self, X): return X

class _RFF:
    def fit(self, X):
        g = torch.Generator().manual_seed(0)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        Xs = (X - self.mu) / self.sd
        d2 = torch.cdist(Xs[:300], Xs[:300]) ** 2
        sigma = math.sqrt(d2[d2 > 0].median().item() / 2)
        self.W = torch.randn(X.shape[1], DIM_MAX, generator=g) / sigma
        self.b = torch.rand(DIM_MAX, generator=g) * 2 * math.pi
    def phi(self, X):
        return math.sqrt(2 / DIM_MAX) * torch.cos(((X - self.mu) / self.sd) @ self.W + self.b)


def _mediana(cls, gerador, checar=True):
    vals = [rodar(cls(), gerador, sem, checar) for sem in SEMENTES]
    return float(torch.tensor(vals).median())


def avaliar():
    print(f"{'tarefa':<12}{'baseline':>10}{'referência':>12}{'você':>8}{'s_t':>7}")
    s = []
    for nome, gen in TAREFAS.items():
        b = _mediana(_Identidade, gen, checar=False)
        r = max(_mediana(_RFF, gen, checar=False), b + 1e-3)
        try:
            m = _mediana(Submissao, gen); erro = ""
        except AssertionError as e:
            m, erro = b, f"   <- {e}"
        st = min(max((m - b) / (r - b), 0.0), 1.25); s.append(st)
        print(f"{nome:<12}{b:>10.3f}{r:>12.3f}{m:>8.3f}{st:>7.2f}{erro}")
    S = 100 * (0.7 * sum(s) / len(s) + 0.3 * min(s))
    print(f"\nESCORE S = {S:.1f}   (0 = baseline, 100 = referência, até 125 com bônus)")
    return S


if __name__ == "__main__":
    avaliar()
