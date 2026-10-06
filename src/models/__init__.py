"""Exactly symmetric output-linear residual networks."""
import torch
from torch import nn


class ResidualNetwork(nn.Module):
    def __init__(self, embedding, genes, config, output_basis=None):
        super().__init__()
        self.config = config
        self.family = config["family"]
        self.embedding = nn.Parameter(torch.as_tensor(embedding, dtype=torch.float64).clone(),
                                      requires_grad=config.get("finetune", False))
        m = self.embedding.shape[1]
        h = config.get("features", m)
        k = min(config.get("rank", 4), genes)
        if config.get("unrestricted", False):
            k = genes
        self.rank = k
        self.unrestricted = config.get("unrestricted", False)
        if self.family in ["bilinear", "network"]:
            self.projection = nn.Linear(m, h, bias=False, dtype=torch.float64)
        if self.family == "independent":
            self.forms = nn.Parameter(torch.randn(k, m, m, dtype=torch.float64) / m)
            self.coefficient = nn.Linear(k, k, bias=False, dtype=torch.float64)
        elif self.family == "network" or self.family == "sum_network":
            width = config.get("width", 64)
            inputs = 2 * m + (h if self.family == "network" else 0)
            layers = [nn.Linear(inputs, width, dtype=torch.float64), nn.Tanh()]
            for _ in range(config.get("depth", 1) - 1):
                layers += [nn.Linear(width, width, dtype=torch.float64), nn.Tanh()]
            self.hidden = nn.Sequential(*layers)
            self.coefficient = nn.Linear(width, k, bias=False, dtype=torch.float64)
        else:
            self.coefficient = nn.Linear(h if self.family == "bilinear" else m, k, bias=False, dtype=torch.float64)
        if k == genes and config.get("unrestricted", False):
            self.register_buffer("basis", torch.empty(genes, 0, dtype=torch.float64))
        else:
            basis = torch.randn(genes, k, dtype=torch.float64) / genes ** 0.5 if output_basis is None else torch.as_tensor(output_basis[:, :k], dtype=torch.float64)
            self.basis = nn.Parameter(basis.clone(), requires_grad=not config.get("fixed_basis", False))
        self.offset = nn.Parameter(torch.zeros(genes, dtype=torch.float64), requires_grad=config.get("offset", True))
        if not config.get("random_init", False):
            nn.init.zeros_(self.coefficient.weight)

    def forward(self, pairs):
        a, b = self.embedding[pairs[:, 0]], self.embedding[pairs[:, 1]]
        if self.family == "coordinate":
            features = a * b
        elif self.family == "bilinear":
            features = self.projection(a) * self.projection(b)
        elif self.family == "independent":
            w = (self.forms + self.forms.transpose(1, 2)) / 2
            # Averaging evaluates identical arithmetic under pair reversal.
            features = (torch.einsum("ni,kij,nj->nk", a, w, b) + torch.einsum("ni,kij,nj->nk", b, w, a)) / 2
        elif self.family in ["network", "sum_network"]:
            parts = [a + b, torch.abs(a - b)]
            if self.family == "network":
                parts.append(self.projection(a) * self.projection(b))
            features = self.hidden(torch.cat(parts, dim=1))
        else:
            raise ValueError("Unknown neural family")
        coefficients = self.coefficient(features)
        return coefficients + self.offset if self.unrestricted else coefficients @ self.basis.T + self.offset
