"""
Statistical significance of baseline vs yolo_guided ACT under corruption.

Reads the per-rollout results written by scripts/evaluate.py and, for each
severity, reports:

  * success rate + Wilson 95% CI for each mode
  * the paired difference (yolo_guided - baseline)
  * McNemar's exact test when rollouts are paired (same per-rollout seeds, which
    is how run_robustness.sh evaluates), otherwise a two-proportion z-test

Pairing matters here: evaluate.py reseeds each rollout to `seed + rollout_id`,
so both modes face the *same* scene layouts. McNemar's test conditions on that
pairing and is the correct, more powerful test — it only looks at scenes where
the two modes disagree.

Usage:
    python scripts/significance.py
    python scripts/significance.py --mode_a baseline --mode_b yolo_guided
"""
import argparse
import json
import math
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_PROJECT_ROOT, 'data', 'eval_results')


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _binom_two_sided_p(b, c):
    """Exact McNemar two-sided p-value: Binomial(b; b+c, 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    from math import comb
    def cdf_le(x):
        return sum(comb(n, i) for i in range(0, x + 1)) / (2 ** n)
    k = min(b, c)
    p = 2 * cdf_le(k)
    return min(1.0, p)


def mcnemar(a_succ, b_succ):
    """Paired test. a_succ/b_succ are aligned 0/1 lists. Returns (b, c, p)."""
    b = sum(1 for x, y in zip(a_succ, b_succ) if y == 1 and x == 0)  # B better
    c = sum(1 for x, y in zip(a_succ, b_succ) if y == 0 and x == 1)  # A better
    return b, c, _binom_two_sided_p(b, c)


def two_proportion_z(ka, na, kb, nb):
    """Unpaired two-proportion z-test, two-sided p (normal approx)."""
    if na == 0 or nb == 0:
        return 1.0
    pa, pb = ka / na, kb / nb
    p = (ka + kb) / (na + nb)
    se = math.sqrt(p * (1 - p) * (1 / na + 1 / nb))
    if se == 0:
        return 1.0
    z = (pb - pa) / se
    # two-sided normal tail
    return math.erfc(abs(z) / math.sqrt(2))


def load_results(mode):
    """Return {severity: result_dict} for a mode."""
    out = {}
    if not os.path.isdir(RESULTS_DIR):
        return out
    for fn in os.listdir(RESULTS_DIR):
        if not (fn.startswith(f'results_{mode}_') and fn.endswith('.json')):
            continue
        with open(os.path.join(RESULTS_DIR, fn)) as f:
            res = json.load(f)
        if res.get('mode') == mode:
            out[res['severity']] = res
    return out


def paired_align(ra, rb):
    """Align two result dicts by rollout seed. Returns (a_succ, b_succ) or None."""
    sa, sb = ra.get('rollout_seeds'), rb.get('rollout_seeds')
    pa, pb = ra.get('per_rollout_success'), rb.get('per_rollout_success')
    if not (sa and sb and pa and pb):
        return None
    da = dict(zip(sa, pa))
    db = dict(zip(sb, pb))
    common = sorted(set(da) & set(db))
    if not common:
        return None
    return [da[s] for s in common], [db[s] for s in common]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode_a', default='baseline')
    ap.add_argument('--mode_b', default='yolo_guided')
    ap.add_argument('--alpha', type=float, default=0.05)
    args = ap.parse_args()

    A = load_results(args.mode_a)
    B = load_results(args.mode_b)
    severities = sorted(set(A) & set(B))
    if not severities:
        print(f"No overlapping severities for {args.mode_a} / {args.mode_b} in {RESULTS_DIR}")
        print("Run scripts/run_robustness.sh first (and ensure both modes were evaluated).")
        sys.exit(1)

    rows = []
    header = (f"{'sev':>3} | {args.mode_a:>10} (95% CI)      | "
              f"{args.mode_b:>11} (95% CI)      | {'Δ (B-A)':>8} | test         | "
              f"{'p':>7} | sig")
    print(header)
    print('-' * len(header))

    summary = []
    for sev in severities:
        ra, rb = A[sev], B[sev]
        na, nb = ra['num_rollouts'], rb['num_rollouts']
        ka = ra.get('success_count', round(ra['success_rate'] * na))
        kb = rb.get('success_count', round(rb['success_rate'] * nb))
        pa, pb = ka / na, kb / nb
        cia = wilson_ci(ka, na)
        cib = wilson_ci(kb, nb)

        aligned = paired_align(ra, rb)
        if aligned is not None:
            a_succ, b_succ = aligned
            bb, cc, pval = mcnemar(a_succ, b_succ)
            test = f"McNemar n={len(a_succ)}"
        else:
            pval = two_proportion_z(ka, na, kb, nb)
            test = "2-prop z"

        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < args.alpha else 'ns'
        print(f"{sev:>3} | {pa:>6.3f} [{cia[0]:.2f},{cia[1]:.2f}] | "
              f"{pb:>6.3f} [{cib[0]:.2f},{cib[1]:.2f}] | {pb - pa:>+8.3f} | "
              f"{test:<12} | {pval:>7.4f} | {sig}")
        summary.append({
            'severity': sev, 'n_a': na, 'n_b': nb,
            'success_rate_a': pa, 'success_rate_b': pb,
            'ci_a': cia, 'ci_b': cib, 'delta': pb - pa,
            'test': test, 'p_value': pval, 'significant': pval < args.alpha,
        })

    print("\nLegend: *** p<0.001  ** p<0.01  * p<0.05  ns = not significant")
    out_path = os.path.join(RESULTS_DIR, f'significance_{args.mode_a}_vs_{args.mode_b}.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
