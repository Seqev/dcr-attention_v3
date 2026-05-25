"""
Phase 5.5 — final merge of v2.0 main.tex with Phase 5.1-5.4 .tex rewrites.

A.1/A.2: replace ranges in v2.0 main.tex with new content
A.3: label renames (thm:scaling, sec:hero)
B.1: §6.2 trained-indexer rewording (applied directly to section_6_discussion.tex)

Outputs: paper/main.tex (merged), with all \input{} directives for the
phase-5 .tex files. Compile is done in a separate step.
"""
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path("/home/user/dcr-attention")
PAPER = ROOT / "paper"
MAIN_TEX = PAPER / "main.tex"


def main():
    src = MAIN_TEX.read_text(encoding="utf-8").splitlines(keepends=True)
    out_lines = []
    i = 0
    n = len(src)

    # State: track which range we're in
    # v2.0 line numbers (1-based):
    #   78-113   abstract block
    #   226-319  §3 Theoretical analysis (entire)
    #   322-390  §4 sec:hero header + Experimental setup + Quality scaling
    #   415-440  sec:hero4 (Pareto frontier + stat significance) — kept with footnote
    #            (sec:pca, sec:topk, sec:pareto retained as-is)
    #   631      §5 section header (rewrite the title only; keep label sec:systems)
    #   706-748  sec:latency_results subsection (replace with new §5 body)
    #   880-1082 §7 Related work + §8 Discussion + §9 Conclusion (replace)

    # We use 1-based line numbers; the array is 0-based.
    def lineno(idx0):  # idx0 is 0-based
        return idx0 + 1

    while i < n:
        ln = lineno(i)

        # Abstract block 78-113 → \input
        if ln == 78:
            out_lines.append("\\input{sections/abstract.tex}\n")
            # skip to line 114
            while i < n and lineno(i) <= 113:
                i += 1
            continue

        # §3 226-319 → \input section_3_theory (which inputs X, Y)
        if ln == 226:
            out_lines.append("\\input{sections/section_3_theory.tex}\n")
            while i < n and lineno(i) <= 319:
                i += 1
            continue

        # §4 322-390 → \input section_4_results
        # (preserves 391+: sec:pca, sec:hero4, sec:topk, sec:pareto as further subsections)
        if ln == 322:
            out_lines.append("\\input{sections/section_4_results.tex}\n")
            out_lines.append("\n")
            out_lines.append("% Preserved from v2.0 §4: sec:pca, sec:hero4, sec:topk, sec:pareto\n")
            out_lines.append("% remain as further subsections of the new \\section{Results}.\n")
            out_lines.append("% sec:hero4 carries the now-superseded single-seed +0.308\\% claim;\n")
            out_lines.append("% Phase 5.2 acceptance flagged this for Phase 5.5 micro-edit, but the\n")
            out_lines.append("% architect's 5.5 prompt did not enumerate it among B-tasks. Surfaced\n")
            out_lines.append("% in PHASE_5_5_ACCEPTANCE_REPORT.md §D for architect resolution.\n")
            while i < n and lineno(i) <= 390:
                i += 1
            continue

        # §5 section header (line 631): rewrite title, keep label sec:systems
        if ln == 631 and "Systems engineering" in src[i]:
            out_lines.append(
                "\\section{Systems: latency landscape and the path to speedup}\\label{sec:systems}\n"
            )
            i += 1
            continue

        # sec:latency_results 706-748 → \input section_5_systems_body (strip section header)
        if ln == 706:
            out_lines.append(
                "% \\input below is the BODY of section_5_systems.tex (skipping the\n"
                "% top-level \\section header — the v2.0 \\section{Systems...} above is\n"
                "% the host section).\n"
            )
            out_lines.append("\\input{sections/section_5_systems_body.tex}\n")
            while i < n and lineno(i) <= 748:
                i += 1
            continue

        # §7 + §8 + §9 (880-1082) → \input section_6_discussion
        if ln == 880:
            out_lines.append("\\input{sections/section_6_discussion.tex}\n")
            while i < n and lineno(i) <= 1082:
                i += 1
            continue

        # default: copy line
        out_lines.append(src[i])
        i += 1

    merged = "".join(out_lines)

    # A.3 — label renames (project-wide in main.tex)
    # thm:scaling → thm:empirical-scaling
    n_thm_scaling = len(re.findall(r"\bthm:scaling\b", merged))
    merged = re.sub(r"\bthm:scaling\b", "thm:empirical-scaling", merged)

    # sec:hero → sec:hero-deployment  (but NOT sec:hero4 — that label survives)
    # Use negative lookahead: sec:hero not followed by 4
    n_sec_hero = len(re.findall(r"\bsec:hero(?!\w)", merged))
    merged = re.sub(r"\bsec:hero(?!\w)", "sec:hero-deployment", merged)

    MAIN_TEX.write_text(merged, encoding="utf-8")

    print(f"Merge complete:")
    print(f"  thm:scaling → thm:empirical-scaling : {n_thm_scaling} occurrences")
    print(f"  sec:hero → sec:hero-deployment      : {n_sec_hero} occurrences")
    print(f"  output: {MAIN_TEX} ({len(merged.splitlines())} lines)")


if __name__ == "__main__":
    main()
