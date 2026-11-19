#!/usr/bin/env python3
"""Every compiled object in the tree must be byte-identical to the wheel's copy.

  python wheel_provenance.py <wheel> <tree>   # exit 1 on any mismatch or no objects
"""
import glob, hashlib, json, os, sys, zipfile
wheel, tree = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(wheel)
inside = {n: hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.endswith(".so")}
meta = [n for n in z.namelist() if n.endswith(".dist-info/METADATA")]
version = next((l.split(":", 1)[1].strip() for l in z.read(meta[0]).decode().splitlines()
                if l.startswith("Version:")), None) if meta else None
local = {}
for so in glob.glob(os.path.join(tree, "vllm", "**", "*.so"), recursive=True):
    rel = os.path.relpath(so, tree)
    local[rel] = hashlib.sha256(open(so, "rb").read()).hexdigest()
mismatch = sorted(r for r, h in local.items() if inside.get(r) != h)
print(json.dumps({"wheel": os.path.basename(wheel), "wheel_version": version,
                  "objects_in_tree": len(local), "objects_in_wheel": len(inside),
                  "not_identical_to_wheel": mismatch, "sha256": local}, indent=2))
sys.exit(1 if mismatch or not local else 0)
