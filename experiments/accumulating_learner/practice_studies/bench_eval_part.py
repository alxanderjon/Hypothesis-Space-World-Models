import sys, os, glob
sys.path.insert(0, "/tmp/claude-1007/-home-alex-Hypothesis-Space-World-Models/9a225413-5adf-5fdf-bc92-9ffd33e7d712/scratchpad")
import bench_eval as B
k, K = int(sys.argv[1]), int(sys.argv[2])
B.PYSR = B.SP + "/bench/pysr"; B.OUT = B.SP + "/bench/eval"
os.makedirs(B.OUT, exist_ok=True)
files = sorted(glob.glob(B.PYSR + "/*.json"))
for idx, p in enumerate(files):
    if idx % K != k or os.path.exists(B.OUT + "/" + os.path.basename(p)):
        continue
    r = B.finish(os.path.basename(p)[:-5])
    print(f"{r['tag']}: {r['status']} val {r.get('val', float('nan')):.2e} exact {r['exact']}", flush=True)
