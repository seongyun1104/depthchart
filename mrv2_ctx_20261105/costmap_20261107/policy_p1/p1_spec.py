"""Fixed parameters of the Phase 1 policy-room experiment (POLICY_MIN_EXP_DESIGN_20261111.md)."""

SHAPES = ((1, 1024), (8, 1024), (32, 1024), (8, 4096), (8, 16384))
BATCHES = 2
CONFIGS = ("OFF", "K1", "K3", "K5")
ORDER_SEED = 20261112
PLAN_SEED = 20261112
SELECT_SEED = 20261112
OUT_LEN = 512
MIN_WINDOW_STEPS = 64

DATASET = {
    "repo": "launch/gov_report",
    "revision": "32feeaede49fed993aef070bc4da09263fd0429a",
    "file": "data/gao_test.jsonl",
    "sha256": "5285ee273ae430003c1e8fdb45ed7c6757349c114d2cae45160845ef60a49a94",
    "license": "cc-by-4.0",
}
TOKENIZER = {
    "repo": "prithivMLmods/gemma-4-31B-it-qat-FP8",
    "tokenizer_json_sha256": "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
    "chat_template_sha256": "94899c0f917d93f6fe81c95744d1e8ddab2d21d39228d2e4aec1fb2a25bff413",
    "transformers": "5.22.1",
}
PRE = ("Below is an excerpt from a report by the U.S. Government Accountability Office (GAO). "
       "The excerpt may end mid-sentence.\n\n<report>\n")
POST = ("\n</report>\n\nWrite a summary of the excerpt above in plain prose, about 300 words long. "
        "Cover what was examined, the main findings and the evidence behind them, and any recommendations.")


def cell_id(bs, ctx, batch):
    return f"nat_b{bs}_c{ctx}_k{batch}"


def launch_order():
    """A = seeded shuffle of CONFIGS, B = A reversed; each config twice."""
    import random

    a = list(CONFIGS)
    random.Random(ORDER_SEED).shuffle(a)
    return a + a[::-1]
