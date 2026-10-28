#!/usr/bin/env python3
import argparse, importlib.util, numpy as np

def gen_p3(rng):
    m=rng.normal(size=(4,4)); return m/max(np.linalg.norm(m),1e-12)*2.0

def apply_p3(x,m):
    p3=(5.0*x**3-3.0*x)/2.0
    return p3@m

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--driver',required=True); ap.add_argument('--plan',required=True); ap.add_argument('--out',required=True); a=ap.parse_args()
    spec=importlib.util.spec_from_file_location('frozen_common_panel_driver',a.driver); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    m.WORLD_SEED=2026104500; m.gen_net=gen_p3; m.apply_net=apply_p3
    m.main(argparse.Namespace(plan=a.plan,out=a.out))

if __name__=='__main__': main()
