from __future__ import annotations

from .schema import ControllerConfig

Tier = tuple[int, int]  # (bs_range_idx, ctx_range_idx)


class LUT2D:
    def __init__(self, config: ControllerConfig):
        rows = config.schedule_2d
        bs_pairs = sorted({(r.bs_lo, r.bs_hi) for r in rows})
        ctx_pairs = sorted({(r.ctx_lo, r.ctx_hi) for r in rows})
        self.bs_ranges: list[tuple[int, int]] = bs_pairs
        self.ctx_ranges: list[tuple[int, int]] = ctx_pairs
        self.max_bs: int = bs_pairs[-1][1]
        n_b, n_c = len(bs_pairs), len(ctx_pairs)
        self._grid: list[list[int]] = [[0] * n_c for _ in range(n_b)]
        bs_index = {p: i for i, p in enumerate(bs_pairs)}
        ctx_index = {p: j for j, p in enumerate(ctx_pairs)}
        for r in rows:
            i = bs_index[(r.bs_lo, r.bs_hi)]
            j = ctx_index[(r.ctx_lo, r.ctx_hi)]
            self._grid[i][j] = r.k

    def _bs_idx(self, bs: int) -> int:
        bs_c = max(1, min(bs, self.max_bs))
        for i, (lo, hi) in enumerate(self.bs_ranges):
            if lo <= bs_c <= hi:
                return i
        return len(self.bs_ranges) - 1

    def _ctx_idx(self, ctx: int) -> int:
        ctx_c = max(1, ctx)
        for j, (lo, hi) in enumerate(self.ctx_ranges):
            if lo <= ctx_c <= hi:
                return j
        return len(self.ctx_ranges) - 1  # ctx above top bucket clamps to top

    def tier_of(self, bs: int, ctx: int) -> Tier:
        return (self._bs_idx(bs), self._ctx_idx(ctx))

    def lookup(self, bs: int, ctx: int) -> int:
        i, j = self.tier_of(bs, ctx)
        return self._grid[i][j]

    def k_at(self, tier: Tier) -> int:
        i, j = tier
        return self._grid[i][j]

    def tier_ranges(self, tier: Tier) -> tuple[tuple[int, int], tuple[int, int]]:
        i, j = tier
        return self.bs_ranges[i], self.ctx_ranges[j]
