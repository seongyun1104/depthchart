#!/usr/bin/env python3
"""Which source and which binaries does this Python actually load?

Imports the modules E1 depends on and checks that each file lies under the B1
tree. On Linux it also reads /proc/self/maps after loading the compiled ops and
checks that every mapped vLLM shared object is under the tree and has the sha256
recorded by wheel_provenance.py. Elsewhere the mapping check is reported as not
run, never as passed.

  python check_import_paths.py <tree> <wheel_provenance.json|-> [--no-ops]
"""
import hashlib
import importlib
import json
import os
import sys

MODULES = (
    "vllm",
    "vllm.config.speculative",
    "vllm.v1.core.sched.scheduler",
    "vllm.v1.spec_decode.dynamic.utils",
    "vllm.model_executor.models.gemma4",
    "vllm.model_executor.models.gemma4_mtp",
    "vllm.v1.worker.gpu.cudagraph_utils",
    "vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils",
)


def main():
    tree = os.path.realpath(sys.argv[1])
    prov = None if sys.argv[2] == "-" else json.load(open(sys.argv[2]))
    out, ok = {"tree": tree, "modules": {}}, True
    for name in MODULES:
        f = os.path.realpath(importlib.import_module(name).__file__)
        inside = f.startswith(tree + os.sep)
        out["modules"][name] = {"file": f, "under_tree": inside}
        ok &= inside
    if "--no-ops" not in sys.argv:
        importlib.import_module("vllm._custom_ops")
    if os.path.exists("/proc/self/maps"):
        mapped = sorted({line.split()[-1] for line in open("/proc/self/maps")
                         if line.rstrip().endswith(".so") and "/vllm/" in line})
        rows = {}
        for p in mapped:
            rp = os.path.realpath(p)
            rel = os.path.relpath(rp, tree) if rp.startswith(tree + os.sep) else None
            sha = hashlib.sha256(open(rp, "rb").read()).hexdigest()
            want = (prov or {}).get("sha256", {}).get(rel) if rel else None
            rows[rp] = {"under_tree": rel is not None, "sha256_matches_wheel": want == sha}
            ok &= rel is not None and want == sha
        out["mapped_vllm_objects"] = rows
        ok &= bool(rows) or "--no-ops" in sys.argv
    else:
        out["mapped_vllm_objects"] = "not checked: /proc/self/maps unavailable on this OS"
    print(json.dumps(out, indent=2))
    print("IMPORT_PATHS_OK" if ok else "IMPORT_PATHS_FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
