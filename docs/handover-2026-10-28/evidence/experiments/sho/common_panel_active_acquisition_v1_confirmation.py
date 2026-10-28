#!/usr/bin/env python3
import argparse,importlib.util
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--driver',required=True); ap.add_argument('--plan',required=True); ap.add_argument('--out',required=True); a=ap.parse_args()
 spec=importlib.util.spec_from_file_location('frozen_active_driver',a.driver); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); m.WORLD=2026105100; m.main(argparse.Namespace(plan=a.plan,out=a.out))
if __name__=='__main__': main()
