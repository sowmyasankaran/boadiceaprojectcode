

import argparse
import os

from pedigree_data import load_vignettes
from exp1_validity import run_experiment1
from exp2_verification import run_experiment2
from exp4_extraction_robustness import run_experiment4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results")
    parser.add_argument("--submit", action="store_true",
                        help="make live CanRisk calls where relevant (exp1)")
    parser.add_argument("--n-repeats-exp4", type=int, default=1)
    parser.add_argument("--depth-repeats-exp1", type=int, default=0)
    parser.add_argument("--all-vignettes-exp4", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("=== Experiment 1: Validity + Equivalence (+ Exp5 reliability) ===")
    run_experiment1(load_vignettes(tag="exp1"), submit_live=args.submit,
                    depth_repeats=args.depth_repeats_exp1, out_dir=args.out)

    print("\n=== Experiment 2: Verification ===")
    run_experiment2(out_dir=args.out)

    print("\n=== Experiment 4: Robustness of Extraction ===")
    run_experiment4(load_vignettes(tag=None if args.all_vignettes_exp4 else "exp4"),
                    n_repeats=args.n_repeats_exp4, out_dir=args.out)

    print(f"\nAll done. Results in {args.out}/.")


if __name__ == "__main__":
    main()
