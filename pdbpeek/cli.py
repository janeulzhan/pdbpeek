"""Command line entry point: `python -m pdbpeek 1CRN`."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .contacts import DEFAULT_CUTOFF, contact_map
from .fetch import FetchError, fetch_cif, fetch_entry
from .mmcif import MmcifError, parse_file, parse_mmcif, select
from .sasa import DEFAULT_N_POINTS, compute_sasa
from .summary import summarize_entry


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pdbpeek",
        description="Parse a PDB structure and measure its solvent accessibility.",
    )
    p.add_argument("target", help="a four-character PDB id, or a path to an mmCIF file")
    p.add_argument("--model", type=int, default=None, help="model number (default: first)")
    p.add_argument("--chain", default=None, help="restrict to one chain")
    p.add_argument(
        "--n-points",
        type=int,
        default=DEFAULT_N_POINTS,
        help=f"SASA test points per atom (default: {DEFAULT_N_POINTS})",
    )
    p.add_argument(
        "--cutoff",
        type=float,
        default=DEFAULT_CUTOFF,
        help=f"contact map CB-CB cutoff in angstroms (default: {DEFAULT_CUTOFF})",
    )
    p.add_argument("--keep-waters", action="store_true", help="keep HETATM records")
    p.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    p.add_argument(
        "--no-summary",
        action="store_true",
        help="skip the entry metadata lookup, so the run needs no network",
    )
    p.add_argument("--cache-dir", default=None, help="directory to cache fetched mmCIF in")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    target = args.target

    try:
        if os.path.exists(target):
            structure = parse_file(target)
            pdb_id = structure.entry_id.upper()
        else:
            pdb_id = target.strip().upper()
            structure = parse_mmcif(fetch_cif(pdb_id, cache_dir=args.cache_dir))
        selected = select(
            structure,
            model=args.model,
            chain=args.chain,
            drop_hetatm=not args.keep_waters,
        )
    except (MmcifError, FetchError, ValueError) as exc:
        print(f"pdbpeek: {exc}", file=sys.stderr)
        return 2

    sasa = compute_sasa(selected, n_points=args.n_points)
    cmap = contact_map(selected, cutoff=args.cutoff)
    burial = sasa.burial() or []
    buried_fraction = burial.count("buried") / len(burial) if burial else 0.0

    measured = {
        "n_atoms": len(selected.atoms),
        "n_residues": len(sasa.residue_keys),
        "n_chains": len(selected.chains),
        "total_sasa": sasa.total,
        "buried_fraction": buried_fraction,
    }

    summary = ""
    if not args.no_summary and pdb_id:
        try:
            summary = summarize_entry(fetch_entry(pdb_id), measured=measured)
        except (FetchError, ValueError) as exc:
            print(f"pdbpeek: metadata lookup skipped ({exc})", file=sys.stderr)

    if args.json:
        rel = sasa.relative_residue_sasa
        payload = {
            "pdb_id": pdb_id,
            "summary": summary,
            "measured": measured,
            "models_in_file": structure.models,
            "sasa": {
                "probe_radius": sasa.probe,
                "n_points": sasa.n_points,
                "total": round(sasa.total, 3),
                "unknown_elements": sasa.unknown_elements,
            },
            "contacts": {
                "cutoff": cmap.cutoff,
                "n_residues": cmap.n_residues,
                "n_contacts": cmap.n_contacts,
                "mean_contact_order": round(cmap.contact_order(), 3),
            },
            "residues": [
                {
                    "chain": key[0],
                    "seq": key[1],
                    "name": key[2],
                    "sasa": round(float(sasa.residue_sasa[i]), 3),
                    "relative_sasa": round(float(rel[i]), 4) if rel is not None else None,
                    "burial": burial[i] if burial else None,
                }
                for i, key in enumerate(sasa.residue_keys)
            ],
        }
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    print(f"pdbpeek {pdb_id or target}")
    if summary:
        print()
        print(summary)
    print()
    print(f"  atoms              {len(selected.atoms)}")
    print(f"  residues           {len(sasa.residue_keys)}")
    print(f"  chains             {len(selected.chains)} ({', '.join(selected.chains)})")
    print(f"  models in file     {len(structure.models)}")
    print(
        f"  total SASA         {sasa.total:,.1f} A^2  "
        f"(probe {sasa.probe} A, {sasa.n_points} points/atom)"
    )
    print(f"  buried residues    {burial.count('buried')} of {len(burial)}")
    print(
        f"  contacts           {cmap.n_contacts} pairs under {cmap.cutoff:g} A, "
        f"mean sequence separation {cmap.contact_order():.1f}"
    )
    if sasa.unknown_elements:
        print(
            "  note               fallback radius used for: "
            + ", ".join(sasa.unknown_elements)
        )

    rel = sasa.relative_residue_sasa
    if rel is not None:
        order = sorted(range(len(rel)), key=lambda i: -rel[i])[:8]
        print()
        print("  most exposed residues")
        for i in order:
            chain, seq, name = sasa.residue_keys[i]
            print(
                f"    {name} {chain}{seq:<5} "
                f"{sasa.residue_sasa[i]:7.1f} A^2   rel {rel[i]:.2f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
