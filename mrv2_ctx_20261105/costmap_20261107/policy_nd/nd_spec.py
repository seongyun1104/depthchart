"""Fixed parameters of the narrow execution diagnostic (POLICY_ND_PREREG.md): D prompts only, b32·1k and b8·1k,
K5 / K3 / S5 / PBS, three repetitions."""

from policy_p1 import p1_spec as P1

SHAPES = ((32, 1024), (8, 1024))
BATCHES = P1.BATCHES
CONFIGS = ("K5", "K3", "S5", "PBS")
REPS = 3
PLAN_SEED = 20261113
ORDER_SEED = 20261113
MAX_K = 5
# (bs_start, bs_end, ctx_start, ctx_end, K); one context column per batch-size range = no context rule.
TABLES = {"S5": [[1, 64, 1, 1, 5]], "PBS": [[1, 16, 1, 1, 5], [17, 64, 1, 1, 3]]}
PRIMARY = ("PBS", "K5")
PAIRS = (("PBS", "K5"), ("K3", "K5"), ("S5", "K5"), ("PBS", "S5"), ("PBS", "K3"))
TARGET_REL = -0.02
TARGET_SHAPE = (32, 1024)


def table_k(config, bs):
    for bs_start, bs_end, _, _, k in TABLES[config]:
        if bs_start <= bs <= bs_end:
            return min(k, MAX_K)
    raise ValueError(f"no row for bs {bs} in {config}")


def cell_k(config, bs):
    """K the configuration should select at this batch size (static K ignores bs)."""
    return table_k(config, bs) if config in TABLES else int(config[1:])


def cell_seed(bs, ctx, batch):
    """Same seed as the Phase 1 cell with the same prompts."""
    return 30000 + 10 * P1.SHAPES.index((bs, ctx)) + batch


def launch_order():
    """Rep r = the seeded order rotated by r; reps run one after another."""
    import random

    a = list(CONFIGS)
    random.Random(ORDER_SEED).shuffle(a)
    return [c for r in range(REPS) for c in a[r:] + a[:r]]
