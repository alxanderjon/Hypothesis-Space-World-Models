"""Trial 9: observable-only supervision. Context and training query targets are noisy central differences of the
trajectory; evaluation scores against clean central differences at clean query states. Ground-truth derivatives are
never read. Reuses trial8's training and evaluation loops with the episode function and sampler replaced.

  python trial9.py train --seed 1 --device cuda:0
  python trial9.py eval --device cuda:0
  python trial9.py analyze
"""

import argparse

import torch

import systems as S
import trial8 as T8
from trial1 import N_QUERY

OUT = S.ROOT / "outputs/accumulating_learner/trial9"
ARM = "seqobs400"


def central(x_prev, x_next, h):
    return (x_next - x_prev) / (2 * h)


def episode_obs(x, dx, K, gen, device, h):
    """Evaluation episode. x: (1, T, 3); dx is accepted for interface compatibility and never used."""
    T = x.shape[1]
    ci = (1 + torch.randperm(T // 2 - 2, generator=gen)[:K]).to(device)
    qi = (T // 2 + 1 + torch.randperm(T // 2 - 2, generator=gen)[:N_QUERY]).to(device)
    noise = torch.randn(3, K, 3, generator=gen).to(device) * T8.NOISE_X * x[0].std(0)
    xp, xm, xn = x[0, ci - 1] + noise[0], x[0, ci] + noise[1], x[0, ci + 1] + noise[2]
    yq = central(x[0, qi - 1], x[0, qi + 1], h)  # clean expected observation
    return T8.standardize(xm, central(xp, xn, h), x[0, qi], yq)


class ObsBank(T8.TrajBank):
    """Training sampler: noisy context and noisy query states/targets (all observable)."""

    def sample(self, B, K, gen):
        d, half = self.device, self.T // 2
        sys_ = torch.randint(len(self.n_inst), (B,), generator=gen).to(d)
        inst = (torch.rand(B, generator=gen).to(d) * self.n_inst[sys_]).long()
        ci = (1 + torch.rand(B, half - 2, generator=gen).argsort(1)[:, :K]).to(d)
        qi = (half + 1 + torch.rand(B, half - 2, generator=gen).argsort(1)[:, :N_QUERY]).to(d)
        std = T8.NOISE_X * self.std[sys_, inst].unsqueeze(1)
        nc = torch.randn(3, B, K, 3, generator=gen).to(d) * std
        nq = torch.randn(3, B, N_QUERY, 3, generator=gen).to(d) * std
        tx = self.x[sys_, inst]
        g = lambda t, i: torch.gather(t, 1, i.unsqueeze(-1).expand(-1, -1, 3))
        hh = self.h[sys_].view(B, 1, 1)
        xp, xm, xn = g(tx, ci - 1) + nc[0], g(tx, ci) + nc[1], g(tx, ci + 1) + nc[2]
        qp, qm, qn = g(tx, qi - 1) + nq[0], g(tx, qi) + nq[1], g(tx, qi + 1) + nq[2]
        return T8.standardize(xm, central(xp, xn, hh), qm, central(qp, qn, hh))


def patch():
    T8.OUT, T8.ARM, T8.TrajBank, T8.episode_traj = OUT, ARM, ObsBank, episode_obs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval", "analyze"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    patch()
    if a.cmd == "analyze":
        import analyze_trial8
        analyze_trial8.main()
    else:
        {"train": T8.train, "eval": T8.evaluate}[a.cmd](a)
