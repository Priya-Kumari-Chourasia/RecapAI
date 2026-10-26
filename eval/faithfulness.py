"""Faithfulness evaluation: how much of what Recap AI writes is backed by the source?

Runs the fact-checker over every meeting/document summary in a workspace and
reports how many summary points are supported, partly supported or unsupported.

    python -m eval.faithfulness                     # workspace from RECAP_WORKSPACE / DEFAULT_WORKSPACE
    python -m eval.faithfulness --project Thesis    # one project
    python -m eval.faithfulness --recheck           # re-run checks even if already stored
    python -m eval.faithfulness --csv claims.csv    # every claim with its verdict, for inspection

Faithfulness = (supported + 0.5 * partial) / all claims  (micro-averaged over claims).
"""
import argparse
import csv
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from core import vector_store  # noqa: E402
from core.factcheck import check_source_summary, score  # noqa: E402


def evaluate(workspace: str, project: str | None = None, recheck: bool = False, limit: int | None = None, log=print):
    sources = vector_store.list_sources(workspace, project=project)[:limit]
    rows, all_claims = [], []
    for n, src in enumerate(sources, 1):
        report = None if recheck else src.get("fact_check")
        if not report:
            log(f"[{n}/{len(sources)}] checking “{src['title']}”…")
            report = check_source_summary(workspace, src)
        claims = report["claims"]
        all_claims += [{**c, "source": src["title"], "date": src.get("date"), "type": src.get("source_type")} for c in claims]
        rows.append({"title": src["title"], "date": src.get("date", ""), "type": src.get("source_type", ""), **score(claims)})
    return rows, all_claims, score(all_claims)


def print_report(rows, overall, all_claims, out=sys.stdout):
    def pct(x):
        return "  —  " if x is None else f"{100 * x:5.1f}%"

    out.write(f"\n{'Date':<11} {'Type':<8} {'Claims':>6} {'✅':>4} {'⚠️':>4} {'❌':>4} {'Faithful':>9}  Title\n")
    out.write("-" * 90 + "\n")
    for r in rows:
        out.write(f"{r['date']:<11} {r['type']:<8} {r['total']:>6} {r['supported']:>4} {r['partial']:>4} "
                  f"{r['unsupported']:>4} {pct(r['faithfulness']):>9}  {r['title'][:40]}\n")
    out.write("-" * 90 + "\n")
    out.write(f"{'OVERALL':<20} {overall['total']:>6} {overall['supported']:>4} {overall['partial']:>4} "
              f"{overall['unsupported']:>4} {pct(overall['faithfulness']):>9}  ({len(rows)} summaries)\n")

    flagged = [c for c in all_claims if c["status"] == "unsupported"]
    if flagged:
        out.write(f"\nUnsupported points ({len(flagged)}) — read these to see what the summarizer gets wrong:\n")
        for c in flagged[:15]:
            out.write(f"  ❌ [{c['source'][:30]}] {c['claim'][:100]}" + (f"  ({c['note']})" if c.get("note") else "") + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", default=os.getenv("RECAP_WORKSPACE", os.getenv("DEFAULT_WORKSPACE", "demo")),
                        help="workspace key (same as in the app)")
    parser.add_argument("--project", help="limit to one project")
    parser.add_argument("--recheck", action="store_true", help="re-run the check even where results are stored")
    parser.add_argument("--limit", type=int, help="only the N most recent summaries")
    parser.add_argument("--csv", help="write every claim and verdict to this CSV file")
    args = parser.parse_args(argv)

    ws = vector_store.workspace_id(args.workspace)
    rows, all_claims, overall = evaluate(ws, args.project, args.recheck, args.limit)
    if not rows:
        print("No meetings or documents found in this workspace/project.")
        return 1
    print_report(rows, overall, all_claims)
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            fields = ["source", "date", "type", "status", "claim", "evidence", "note", "start", "page"]
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_claims)
        print(f"\nWrote {len(all_claims)} claims to {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
