#!/usr/bin/env python3
"""Reduce results/bench_*.jsonl and results/sanitizer_*.json into the tables that
go into the #28090 and #27792 comments. Nothing here is typed by hand.

  python reduce.py --results <dir> [--md out.md]

Per architecture and model file it prints, for every ne11, the median t/s of each
binary across rounds, the ratio force_mmq / selected, the null_rebuild / selected
ratio (the noise floor), and the first ne11 at which force_mmq beats selected by
more than the noise floor. Rows with ne11 > 8 are a built-in control: both
binaries take MMQ there so the ratio must sit at 1.0 within noise.
"""

import argparse
import glob
import json
import os
import statistics
from collections import defaultdict


def load(results):
    rows = []
    for p in sorted(glob.glob(os.path.join(results, "bench_*.jsonl"))):
        for line in open(p):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def reduce_bench(rows):
    # (tag, model, quant) -> ne11 -> binary -> [t/s per round]
    table = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    meta = {}
    for r in rows:
        key = (r["tag"], r["model"], r["quant"])
        table[key][r["ne11"]][r["binary"]].append(r["avg_ts"])
        m = meta.setdefault(key, {"gpu": r["gpu"], "cc": r["compute_cap"], "rounds": set(), "sm": [], "temp": []})
        m["rounds"].add(r["round"])
        t = r.get("telemetry") or {}
        if t.get("sm_mhz_mean"):
            m["sm"].append(t["sm_mhz_mean"])
        if t.get("temp_c_max"):
            m["temp"].append(t["temp_c_max"])
    out = {}
    for key, per_ne in table.items():
        rows_out = []
        noise = 0.0
        for ne11 in sorted(per_ne):
            b = per_ne[ne11]
            if not all(k in b for k in ("selected", "force_mmq", "null_rebuild")):
                continue
            sel = statistics.median(b["selected"])
            fm = statistics.median(b["force_mmq"])
            nr = statistics.median(b["null_rebuild"])
            rows_out.append({
                "ne11": ne11, "selected": sel, "force_mmq": fm, "null_rebuild": nr,
                "ratio": fm / sel, "null_ratio": nr / sel,
                "n": min(len(b["selected"]), len(b["force_mmq"]), len(b["null_rebuild"])),
                # per-round ratio, so "wins in every round" can be stated or not
                "wins": sum(1 for s, f in zip(b["selected"], b["force_mmq"]) if f > s),
            })
            if 2 <= ne11 <= 8:
                noise = max(noise, abs(nr / sel - 1.0))
        # ne11 = 1 is excluded: force_mmq still returns MMVQ there (ne11 <= 1), so both
        # binaries run the same kernel and any difference at 1 is noise, not a crossover.
        crossover = None
        for r in rows_out:
            if 2 <= r["ne11"] <= 8 and r["ratio"] > 1.0 + noise:
                crossover = r["ne11"]
                break
        control = [r["ratio"] for r in rows_out if r["ne11"] > 8]
        m = meta[key]
        out[key] = {
            "gpu": m["gpu"], "cc": m["cc"], "rounds": len(m["rounds"]),
            "sm_mhz_mean": statistics.fmean(m["sm"]) if m["sm"] else None,
            "temp_c_max": max(m["temp"]) if m["temp"] else None,
            "noise_floor": noise, "crossover": crossover,
            "control_ratio_range": (min(control), max(control)) if control else None,
            "rows": rows_out,
        }
    return out


def md_bench(red):
    lines = []
    for (tag, model, quant), d in sorted(red.items()):
        lines.append(f"### {d['gpu']} ({tag}), {model} pure {quant}, {d['rounds']} rounds, "
                     f"SM clock mean {d['sm_mhz_mean'] and round(d['sm_mhz_mean'])} MHz, max temp {d['temp_c_max']} C")
        lines.append("")
        lines.append("| ne11 | selected | force_mmq | null_rebuild | force_mmq / selected | null / selected | rounds force_mmq won |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for r in d["rows"]:
            mark = ""
            lines.append(f"| {r['ne11']} | {r['selected']:.1f} | {r['force_mmq']:.1f} | {r['null_rebuild']:.1f} | "
                         f"{r['ratio']:.4f}{mark.strip() and ''} | {r['null_ratio']:.4f} | {r['wins']}/{r['n']} |")
        cr = d["crossover"]
        lines.append("")
        lines.append(f"noise floor (max |null/selected - 1| over ne11 1..8): {d['noise_floor']*100:.2f}%; "
                     f"first ne11 where force_mmq beats selected by more than the noise floor: "
                     f"{cr if cr is not None else 'none up to 8'}; "
                     f"control ne11 9..16 ratio range: "
                     + (f"{d['control_ratio_range'][0]:.4f} to {d['control_ratio_range'][1]:.4f}" if d['control_ratio_range'] else "n/a"))
        lines.append("")
    return "\n".join(lines)


def md_summary(red):
    """One row per (arch, model, quant): the crossover table the issue is asking for."""
    lines = ["| arch | GPU | model | quant | ne11=8 force_mmq/selected | noise floor | first winning ne11 (<=8) |",
             "|---|---|---|---|---:|---:|---:|"]
    for (tag, model, quant), d in sorted(red.items()):
        r8 = next((r for r in d["rows"] if r["ne11"] == 8), None)
        lines.append(f"| {tag} | {d['gpu']} | {model} | {quant} | "
                     f"{r8['ratio']:.4f} | {d['noise_floor']*100:.2f}% | "
                     f"{d['crossover'] if d['crossover'] is not None else 'none'} |" if r8 else
                     f"| {tag} | {d['gpu']} | {model} | {quant} | n/a | | |")
    return "\n".join(lines)


def md_sanitizer(results):
    lines = []
    for p in sorted(glob.glob(os.path.join(results, "sanitizer_*.json"))):
        d = json.load(open(p))
        lines.append(f"### {d['gpu']} ({d['tag']})")
        lines.append("")
        lines.append("| binary | filter | invalid global reads (memcheck summary) | kernel: reads, max bytes past, allocation sizes hit | test-backend-ops |")
        lines.append("|---|---|---:|---|---|")
        for label, r in d["runs"].items():
            variant, _, filt = label.partition(":")
            ks = "; ".join(f"{k}: {v['count']}, {v['max_after']} B past, allocs {v['allocs']}" for k, v in r["by_kernel"].items()) or "none"
            lines.append(f"| {variant} | {filt} | {r.get('summary_errors')} | {ks} | "
                         f"ok {r.get('tests_ok', '?')} fail {r.get('tests_fail', '?')}"
                         + (" (sanitizer unsupported on this host)" if r.get("sanitizer_unsupported") else "") + " |")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    rows = load(a.results)
    red = reduce_bench(rows)
    parts = []
    if red:
        parts.append("## Crossover summary\n\n" + md_summary(red) + "\n\n## Per-architecture tables\n\n" + md_bench(red))
    san = md_sanitizer(a.results)
    if san:
        parts.append("## compute-sanitizer, MUL_MAT_ID under an exact-size pool\n\n" + san)
    text = "\n".join(parts)
    print(text)
    if a.md:
        open(a.md, "w").write(text)
    json.dump({"|".join(k): v for k, v in red.items()}, open(os.path.join(a.results, "reduced.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
