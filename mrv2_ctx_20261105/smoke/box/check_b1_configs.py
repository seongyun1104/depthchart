#!/usr/bin/env python3
"""Config contract, before weights. Copied from e1_prep for vLLM main 7ecc83a987,
which reads per-layer head sizes the same way (gemma4.py:600, gemma4_mtp.py:269).
Original note for the B1 tree (PR #48944 head on 9f1c78388):

The B1 tree reads Gemma4 head sizes per layer
(gemma4.py, gemma4_mtp.py: config.per_layer_config[i].head_dim), so the contract
is the opposite of the 1479bd9e9 tree's: per_layer_config must exist and each
layer's (head_dim, num_key_value_heads) must equal the raw config.json
(sliding = head_dim / num_key_value_heads, full = global_head_dim /
num_global_key_value_heads). The draft is checked after the same model_type
rewrite SpeculativeConfig applies (gemma4_assistant -> gemma4_mtp). The engine's
own convertor must then give head_size = max over layers.

  python check_b1_configs.py <target_dir> <draft_dir>   # prints CONFIG_CONTRACT_OK last
"""
import collections
import json
import sys

import transformers
import vllm.config  # noqa: F401  (import order: avoids a circular import in the convertor)
from transformers import AutoConfig
from vllm.transformers_utils.model_arch_config_convertor import MODEL_ARCH_CONFIG_CONVERTORS

WINDOW = ((5, 16, 1), (5, 23, 0))  # main 7ecc83a987 requirements/common.txt


def truth(raw):
    t = raw.get("text_config", raw)
    out = []
    for kind in t["layer_types"]:
        if kind == "full_attention" and t.get("global_head_dim"):
            out.append((t["global_head_dim"], t.get("num_global_key_value_heads") or t["num_key_value_heads"]))
        else:
            out.append((t["head_dim"], t["num_key_value_heads"]))
    return out


def check(path, is_draft):
    raw = json.load(open(f"{path}/config.json"))
    cfg = AutoConfig.from_pretrained(path)
    if is_draft and cfg.model_type in ("gemma4_assistant", "gemma4_unified_assistant"):
        cfg.model_type = "gemma4_mtp"
        tc = getattr(cfg, "text_config", cfg)
        if hasattr(tc, "num_kv_shared_layers"):
            tc.num_kv_shared_layers = 0
        cfg.update({"n_predict": 1, "architectures": ["Gemma4MTPModel"]})
    tc = cfg.get_text_config()
    want = truth(raw)
    pl = getattr(tc, "per_layer_config", None)
    res = {"path": path, "model_type": cfg.model_type, "layers": len(want),
           "layer_types": dict(collections.Counter(raw.get("text_config", raw)["layer_types"]))}
    if pl is None:
        res["error"] = "per_layer_config absent: this transformers does not give the B1 tree per-layer values"
        return res, False
    got = [(c.head_dim, c.num_key_value_heads) for c in pl]
    res["mismatched_layers"] = [i for i, (g, w) in enumerate(zip(got, want)) if g != w]
    res["length_ok"] = len(got) == len(want)
    conv = MODEL_ARCH_CONFIG_CONVERTORS[cfg.model_type](cfg, tc).convert()
    res["convert_head_size"] = conv.head_size
    res["max_layer_head_dim"] = max(h for h, _ in want)
    ok = res["length_ok"] and not res["mismatched_layers"] and conv.head_size == res["max_layer_head_dim"]
    return res, ok


def main():
    v = tuple(int(x) for x in transformers.__version__.split(".")[:3])
    report = {"transformers": transformers.__version__,
              "in_tree_window": WINDOW[0] <= v < WINDOW[1]}
    oks = [report["in_tree_window"]]
    for path, is_draft in ((sys.argv[1], False), (sys.argv[2], True)):
        r, ok = check(path, is_draft)
        report["draft" if is_draft else "target"] = r
        oks.append(ok)
    print(json.dumps(report, indent=2))
    print("CONFIG_CONTRACT_OK" if all(oks) else "CONFIG_CONTRACT_FAIL")
    sys.exit(0 if all(oks) else 1)


if __name__ == "__main__":
    main()
