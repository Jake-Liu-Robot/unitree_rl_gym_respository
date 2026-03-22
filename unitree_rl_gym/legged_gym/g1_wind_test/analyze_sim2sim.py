"""Sim2Sim analysis: compare Isaac Gym vs MuJoCo evaluation results.

Usage:
    python legged_gym/g1_wind_test/analyze_sim2sim.py [--verbose]

Reads JSON result files from test_results/ and test_results/mujoco/ and prints
4 comparison tables to stdout.

Note on MuJoCo coverage:
    MuJoCo JSONs were run with test_levels=[3], so Suites B-F are only evaluated
    at L3. Suite A covers all levels (L0-L5). The "L5 survival" for MuJoCo is
    derived from A_level5 only (a single scenario), while Isaac Gym L5 is averaged
    across all 26 L5 scenarios (Suites B-F). Keep this asymmetry in mind when
    interpreting the comparison.
"""

import argparse
import json
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

IG_FILES = {
    "Exp1": os.path.join(REPO_ROOT, "test_results", "exp1_baseline", "eval_baseline_all.json"),
    "Exp2": os.path.join(REPO_ROOT, "test_results", "exp2_push_only", "eval_all.json"),
    "Exp3": os.path.join(REPO_ROOT, "test_results", "exp3_run8_full_method", "full_eval_run8.json"),
    "Exp4": os.path.join(REPO_ROOT, "test_results", "exp4_no_curriculum", "eval_all.json"),
    "Exp5": os.path.join(REPO_ROOT, "test_results", "exp5_no_reward", "eval_all.json"),
}

MJ_FILES = {
    "Exp1": os.path.join(REPO_ROOT, "test_results", "mujoco", "test_results", "mujoco", "exp1_baseline_mujoco.json"),
    "Exp2": os.path.join(REPO_ROOT, "test_results", "mujoco", "test_results", "mujoco", "exp2_push_only_mujoco.json"),
    "Exp3": os.path.join(REPO_ROOT, "test_results", "mujoco", "test_results", "mujoco", "exp3_full_method_mujoco.json"),
    "Exp4": os.path.join(REPO_ROOT, "test_results", "mujoco", "test_results", "mujoco", "exp4_no_curriculum_mujoco.json"),
    "Exp5": os.path.join(REPO_ROOT, "test_results", "mujoco", "test_results", "mujoco", "exp5_no_reward_mujoco.json"),
}

EXP_LABELS = {
    "Exp1": "Exp1 (baseline)",
    "Exp2": "Exp2 (push only)",
    "Exp3": "Exp3 (full)",
    "Exp4": "Exp4 (no curric)",
    "Exp5": "Exp5 (no reward)",
}

PASS_THRESHOLD = 0.90
VERDICT_THRESHOLD = 10.0  # percentage points


# ──────────────────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_ig(path):
    """Load Isaac Gym result JSON. Returns flat dict of scenario→metrics."""
    with open(path) as f:
        data = json.load(f)
    return data["results"]["policy_a"]


def load_mj(path):
    """Load MuJoCo result JSON. Returns flat dict of scenario→metrics."""
    with open(path) as f:
        data = json.load(f)
    return data["results"]


def ig_summary(results):
    """Compute summary metrics from Isaac Gym results dict."""
    all_keys = list(results.keys())
    l3 = [k for k in all_keys if k.endswith("_L3")]
    l4 = [k for k in all_keys if k.endswith("_L4")]
    l5 = [k for k in all_keys if k.endswith("_L5")]
    # Suite A (A_levelX) does not have _L3/_L4/_L5 suffix; included in fail count
    l3_pass = sum(1 for k in l3 if results[k]["survival_rate"] >= PASS_THRESHOLD)
    l4_pass = sum(1 for k in l4 if results[k]["survival_rate"] >= PASS_THRESHOLD)
    l5_pass = sum(1 for k in l5 if results[k]["survival_rate"] >= PASS_THRESHOLD)
    l5_surv = (sum(results[k]["survival_rate"] for k in l5) / len(l5)) if l5 else float("nan")
    l5_trk  = (sum(results[k]["mean_tracking_error"] for k in l5) / len(l5)) if l5 else float("nan")
    fails   = sum(1 for k in all_keys if results[k]["survival_rate"] < PASS_THRESHOLD)
    return {
        "l3_pass": l3_pass, "l3_total": len(l3),
        "l4_pass": l4_pass, "l4_total": len(l4),
        "l5_pass": l5_pass, "l5_total": len(l5),
        "l5_surv": l5_surv, "l5_trk": l5_trk,
        "fails": fails, "total": len(all_keys),
    }


def mj_summary(results):
    """Compute summary metrics from MuJoCo results dict.

    MuJoCo was run with test_levels=[3] only:
    - Suite A: A_level0..A_level5 (6 scenarios, all levels)
    - Suites B-F: _L3 suffix only (26 scenarios)
    L5 survival/tracking are from A_level5 only.
    """
    all_keys = list(results.keys())
    l3_bf = [k for k in all_keys if k.endswith("_L3")]   # Suites B-F at L3
    a_keys = sorted(k for k in all_keys if k.startswith("A_level"))

    l3_pass = sum(1 for k in l3_bf if results[k]["survival_rate"] >= PASS_THRESHOLD)
    # L4/L5 from Suite A only
    a5 = results.get("A_level5", {})
    a4 = results.get("A_level4", {})
    l5_surv_a = a5.get("survival_rate", float("nan"))
    l5_trk_a  = a5.get("mean_tracking_error", float("nan"))
    l4_surv_a = a4.get("survival_rate", float("nan"))

    fails = sum(1 for k in all_keys if results[k]["survival_rate"] < PASS_THRESHOLD)
    return {
        "l3_pass": l3_pass, "l3_total": len(l3_bf),
        "l4_surv_a": l4_surv_a,  # Suite A L4 only
        "l5_surv":  l5_surv_a,   # Suite A L5 only
        "l5_trk":   l5_trk_a,
        "fails":    fails,
        "total":    len(all_keys),
        "a_keys":   a_keys,
    }


def suite_breakdown(results, suite_letter):
    """Return list of (key, survival_rate, tracking_error) for one suite."""
    items = []
    for k, v in results.items():
        if k.startswith(suite_letter + "_") or k.startswith(suite_letter.upper() + "_"):
            items.append((k, v["survival_rate"], v["mean_tracking_error"]))
    items.sort(key=lambda x: x[0])
    return items


# ──────────────────────────────────────────────────────────────────────────────
# Print helpers
# ──────────────────────────────────────────────────────────────────────────────

W = 62

def section(title):
    print()
    print("=" * W)
    print(f"  {title}")
    print("=" * W)


def rule():
    print("  " + "─" * (W - 2))


# ──────────────────────────────────────────────────────────────────────────────
# Table 1: MuJoCo summary
# ──────────────────────────────────────────────────────────────────────────────

def print_table1(mj_data):
    section("Table 1 — Sim2Sim (MuJoCo) Evaluation Results")
    print("  32 scenarios per experiment (Suite A L0-L5 + Suites B-F at L3)")
    print("  Note: Suites B-F evaluated at L3 only. L5 survival from Suite A only.")
    print()
    header = f"  {'Exp':<20} {'L3 B-F':>8} {'A-L4':>7} {'A-L5':>7} {'L5 trk':>9} {'Fails':>7}"
    print(header)
    rule()
    for exp in ["Exp1", "Exp2", "Exp3", "Exp4", "Exp5"]:
        r = mj_data[exp]
        label = EXP_LABELS[exp]
        l3_str  = f"{r['l3_pass']}/{r['l3_total']}"
        l4_str  = f"{r['l4_surv_a']*100:.0f}%"
        l5_str  = f"{r['l5_surv']*100:.0f}%"
        trk_str = f"{r['l5_trk']:.3f}"
        fail_str = f"{r['fails']}/{r['total']}"
        print(f"  {label:<20} {l3_str:>8} {l4_str:>7} {l5_str:>7} {trk_str:>9} {fail_str:>7}")
    rule()
    print("  Pass threshold: survival >= 90%")
    print("  L3 B-F = Suites B-F at L3 (26 scenarios); A-L4/A-L5 = Suite A single scenario")


# ──────────────────────────────────────────────────────────────────────────────
# Table 2: Survival comparison
# ──────────────────────────────────────────────────────────────────────────────

def print_table2(ig_data, mj_data):
    section("Table 2 — Isaac Gym vs MuJoCo: L5 Survival Rate")
    print("  IG L5 = mean across 26 L5 scenarios (Suites B-F)")
    print("  MJ L5 = Suite A A_level5 only (single scenario)")
    print()
    header = f"  {'Exp':<20} {'Isaac Gym':>10} {'MuJoCo':>9} {'Delta':>8}   Verdict"
    print(header)
    rule()
    for exp in ["Exp1", "Exp2", "Exp3", "Exp4", "Exp5"]:
        ig = ig_data[exp]
        mj = mj_data[exp]
        ig_s = ig["l5_surv"] * 100
        mj_s = mj["l5_surv"] * 100
        delta = mj_s - ig_s
        verdict = "match" if abs(delta) <= VERDICT_THRESHOLD else "gap"
        sign = "+" if delta >= 0 else ""
        label = EXP_LABELS[exp]
        print(f"  {label:<20} {ig_s:>9.0f}%  {mj_s:>7.0f}%  {sign}{delta:>6.0f}pp   [{verdict}]")
    rule()
    print(f"  Verdict: |delta| <= {VERDICT_THRESHOLD:.0f}pp → 'match', > {VERDICT_THRESHOLD:.0f}pp → 'gap'")


# ──────────────────────────────────────────────────────────────────────────────
# Table 3: Tracking error comparison
# ──────────────────────────────────────────────────────────────────────────────

def print_table3(ig_data, mj_data):
    section("Table 3 — Isaac Gym vs MuJoCo: L5 Tracking Error")
    print("  IG L5 = mean across 26 L5 scenarios | MJ L5 = A_level5 only")
    print()
    header = f"  {'Exp':<20} {'Isaac Gym':>10} {'MuJoCo':>9} {'Delta':>9}   Verdict"
    print(header)
    rule()
    for exp in ["Exp1", "Exp2", "Exp3", "Exp4", "Exp5"]:
        ig = ig_data[exp]
        mj = mj_data[exp]
        ig_t = ig["l5_trk"]
        mj_t = mj["l5_trk"]
        delta = mj_t - ig_t
        sign = "+" if delta >= 0 else ""
        # Verdict: within 20% of ig value
        verdict = "match" if abs(delta) / max(ig_t, 1e-6) <= 0.20 else "gap"
        label = EXP_LABELS[exp]
        print(f"  {label:<20} {ig_t:>10.3f}  {mj_t:>9.3f}  {sign}{delta:>8.3f}   [{verdict}]")
    rule()
    print("  Verdict: |delta/IG| <= 20% → 'match', > 20% → 'gap'")


# ──────────────────────────────────────────────────────────────────────────────
# Table 4: Exp3 per-suite breakdown
# ──────────────────────────────────────────────────────────────────────────────

def print_table4(ig_results_exp3, mj_results_exp3, verbose):
    section("Table 4 — Exp3 (Full Method) Per-Suite Breakdown: IG vs MuJoCo")
    print("  IG: L3 and L5 averages across suite scenarios")
    print("  MJ: L3 only (Suites B-F); Suite A shown separately")
    print()

    suites = [
        ("A", "Wind levels (L0-L5)"),
        ("B", "Wind modes (steady/turb/gusts/full/pure)"),
        ("C", "Wind directions (front/side/back/diag/rand)"),
        ("D", "OU extremes (calm/turb/locked/erratic/default)"),
        ("E", "OOD patterns (step/periodic/reversal)"),
        ("F", "Command variations (stand/slow/norm/fast/lat/turn/hw/tw)"),
    ]

    for letter, desc in suites:
        ig_items = suite_breakdown(ig_results_exp3, letter)
        mj_items = suite_breakdown(mj_results_exp3, letter)

        # Split IG by level
        ig_l3 = [(k, s, t) for k, s, t in ig_items if "_L3" in k or "level3" in k]
        ig_l5 = [(k, s, t) for k, s, t in ig_items if "_L5" in k or "level5" in k]
        mj_l3 = [(k, s, t) for k, s, t in mj_items if "_L3" in k]
        mj_all = mj_items  # Suite A has all levels

        if letter == "A":
            ig_avg_s = sum(s for _, s, _ in ig_items) / len(ig_items) if ig_items else 0
            ig_avg_t = sum(t for _, _, t in ig_items) / len(ig_items) if ig_items else 0
            mj_avg_s = sum(s for _, s, _ in mj_all) / len(mj_all) if mj_all else 0
            mj_avg_t = sum(t for _, _, t in mj_all) / len(mj_all) if mj_all else 0
            print(f"  Suite {letter}: {desc}")
            print(f"    IG all (n={len(ig_items)}): surv={ig_avg_s*100:.0f}%  trk={ig_avg_t:.4f}")
            print(f"    MJ all (n={len(mj_all)}): surv={mj_avg_s*100:.0f}%  trk={mj_avg_t:.4f}")
        else:
            ig_l3_s = sum(s for _, s, _ in ig_l3) / len(ig_l3) if ig_l3 else float("nan")
            ig_l3_t = sum(t for _, _, t in ig_l3) / len(ig_l3) if ig_l3 else float("nan")
            ig_l5_s = sum(s for _, s, _ in ig_l5) / len(ig_l5) if ig_l5 else float("nan")
            ig_l5_t = sum(t for _, _, t in ig_l5) / len(ig_l5) if ig_l5 else float("nan")
            mj_l3_s = sum(s for _, s, _ in mj_l3) / len(mj_l3) if mj_l3 else float("nan")
            mj_l3_t = sum(t for _, _, t in mj_l3) / len(mj_l3) if mj_l3 else float("nan")
            print(f"  Suite {letter}: {desc}")
            print(f"    IG L3 (n={len(ig_l3)}): surv={ig_l3_s*100:.0f}%  trk={ig_l3_t:.4f}"
                  f"   |   IG L5 (n={len(ig_l5)}): surv={ig_l5_s*100:.0f}%  trk={ig_l5_t:.4f}")
            print(f"    MJ L3 (n={len(mj_l3)}): surv={mj_l3_s*100:.0f}%  trk={mj_l3_t:.4f}"
                  f"   |   MJ L4/L5: not evaluated")

        if verbose:
            print(f"    IG scenarios:")
            for k, s, t in ig_items:
                print(f"      {k}: surv={s:.2f} trk={t:.4f}")
            if mj_items:
                print(f"    MJ scenarios:")
                for k, s, t in mj_items:
                    print(f"      {k}: surv={s:.2f} trk={t:.4f}")
        print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sim2Sim results comparison: Isaac Gym vs MuJoCo")
    parser.add_argument("--verbose", action="store_true", help="Print per-scenario detail")
    args = parser.parse_args()

    # Load all results
    ig_raw, mj_raw = {}, {}
    for exp in ["Exp1", "Exp2", "Exp3", "Exp4", "Exp5"]:
        ig_path = IG_FILES[exp]
        mj_path = MJ_FILES[exp]
        if not os.path.exists(ig_path):
            print(f"[WARN] Isaac Gym result not found: {ig_path}")
            ig_raw[exp] = {}
        else:
            ig_raw[exp] = load_ig(ig_path)
        if not os.path.exists(mj_path):
            print(f"[WARN] MuJoCo result not found: {mj_path}")
            mj_raw[exp] = {}
        else:
            mj_raw[exp] = load_mj(mj_path)

    ig_summ = {exp: ig_summary(ig_raw[exp]) for exp in ig_raw if ig_raw[exp]}
    mj_summ = {exp: mj_summary(mj_raw[exp]) for exp in mj_raw if mj_raw[exp]}

    print_table1(mj_summ)
    print_table2(ig_summ, mj_summ)
    print_table3(ig_summ, mj_summ)
    print_table4(ig_raw["Exp3"], mj_raw["Exp3"], args.verbose)


if __name__ == "__main__":
    main()
