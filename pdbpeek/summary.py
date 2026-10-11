"""Plain-English entry summary built by template, with no language model.

Everything below is string formatting over fields from the RCSB entry
endpoint. There is no model, no API key, and no generation step, so the output
cannot hallucinate: a sentence appears only when the field it reads is present
and well formed, and every number in the text is copied from the entry record
rather than predicted. The cost of that guarantee is that the prose is plainer
than a language model's would be. That is the trade I wanted.

Every field is treated as optional. A missing or null value drops its clause
instead of rendering "None", which is what `test_summary.py` pins down.
"""

from __future__ import annotations

from typing import Any

#: Experimental methods as deposited, mapped to readable phrasing.
METHOD_PHRASES = {
    "X-RAY DIFFRACTION": "X-ray crystallography",
    "SOLUTION NMR": "solution NMR",
    "SOLID-STATE NMR": "solid-state NMR",
    "ELECTRON MICROSCOPY": "cryo-electron microscopy",
    "ELECTRON CRYSTALLOGRAPHY": "electron crystallography",
    "NEUTRON DIFFRACTION": "neutron diffraction",
    "FIBER DIFFRACTION": "fiber diffraction",
    "SOLUTION SCATTERING": "solution scattering",
    "POWDER DIFFRACTION": "powder diffraction",
    "THEORETICAL MODEL": "computational modelling",
}

#: Methods for which "resolution" is not a meaningful quantity.
RESOLUTIONLESS = {"SOLUTION NMR", "SOLID-STATE NMR", "THEORETICAL MODEL"}


def _get(obj: Any, *path: str, default: Any = None) -> Any:
    """Walk nested dicts, returning `default` on any missing or null link."""
    cur = obj
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


def _clean_text(value: Any) -> str | None:
    """Normalise a free-text field, or `None` if it carries no information."""
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    if not text or text in (".", "?"):
        return None
    return text


def _sentence_case(text: str) -> str:
    """Soften the ALL CAPS titles that older PDB entries use.

    Lowercasing is only applied when the text is entirely uppercase, so
    modern mixed-case titles and embedded acronyms are left alone.
    """
    letters = [c for c in text if c.isalpha()]
    if letters and all(c.isupper() for c in letters):
        text = text.lower()
    if not text:
        return text
    chars = list(text)
    capitalise_next = True
    for i, ch in enumerate(chars):
        if capitalise_next and ch.isalpha():
            chars[i] = ch.upper()
            capitalise_next = False
        elif ch in ".!?":
            capitalise_next = True
    return "".join(chars)


def _methods(entry: dict) -> list[str]:
    raw = entry.get("exptl")
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        method = _clean_text(_get(item, "method"))
        if method and method.upper() not in [m.upper() for m in out]:
            out.append(method)
    return out


def _join(parts: list[str]) -> str:
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def _keywords(entry: dict) -> list[str]:
    """Distinct keywords, preferring the curated `pdbx_keywords` field."""
    seen: list[str] = []
    lowered: set[str] = set()
    for field in ("pdbx_keywords", "text"):
        text = _clean_text(_get(entry, "struct_keywords", field))
        if not text:
            continue
        for piece in text.split(","):
            kw = piece.strip().lower()
            if kw and kw not in lowered:
                lowered.add(kw)
                seen.append(kw)
    return seen


def _citation(entry: dict) -> str | None:
    cit = entry.get("rcsb_primary_citation")
    if not isinstance(cit, dict):
        return None
    authors = cit.get("rcsb_authors")
    first = None
    if isinstance(authors, list) and authors:
        first = _clean_text(authors[0])
    if first and "," in first:
        first = first.split(",")[0].strip()

    journal = _clean_text(cit.get("rcsb_journal_abbrev")) or _clean_text(
        cit.get("journal_abbrev")
    )
    year = cit.get("year")
    year = str(year) if isinstance(year, (int, float)) else _clean_text(year)

    bits: list[str] = []
    if first:
        bits.append(f"{first} et al." if isinstance(authors, list) and len(authors) > 1 else first)
    if journal:
        bits.append(journal)
    if year:
        bits.append(year)
    if not bits:
        return None

    text = "Published by " + ", ".join(bits) if first else "Published in " + ", ".join(bits)
    doi = _clean_text(cit.get("pdbx_database_id_DOI"))
    if doi:
        text += f" (doi:{doi})"
    return text + "."


def summarize_entry(entry: dict, measured: dict | None = None) -> str:
    """Render a paragraph describing an RCSB entry record.

    `entry` is the JSON body of `data.rcsb.org/rest/v1/core/entry/<id>`.
    `measured` optionally carries values this repo computed from the
    coordinate file, which are labelled as computed so a reader can tell them
    apart from the deposited metadata.
    """
    if not isinstance(entry, dict):
        raise TypeError("entry must be the decoded entry JSON object")

    pdb_id = (
        _clean_text(entry.get("rcsb_id"))
        or _clean_text(_get(entry, "entry", "id"))
        or ""
    ).upper()

    sentences: list[str] = []

    title = _clean_text(_get(entry, "struct", "title"))
    label = pdb_id or "This entry"
    if title:
        sentences.append(f"{label}: {_sentence_case(title)}.")
    else:
        sentences.append(f"{label} is a Protein Data Bank entry.")

    methods = _methods(entry)
    res_list = _get(entry, "rcsb_entry_info", "resolution_combined")
    resolution = None
    if isinstance(res_list, list) and res_list:
        try:
            resolution = float(res_list[0])
        except (TypeError, ValueError):
            resolution = None

    if methods:
        phrases = [METHOD_PHRASES.get(m.upper(), m.lower()) for m in methods]
        clause = f"Determined by {_join(phrases)}"
        resolutionless = all(m.upper() in RESOLUTIONLESS for m in methods)
        if resolution is not None and not resolutionless:
            clause += f" at {resolution:g} A resolution"
        sentences.append(clause + ".")
    elif resolution is not None:
        sentences.append(f"Refined to {resolution:g} A resolution.")

    models = _get(entry, "rcsb_entry_info", "deposited_model_count")
    if isinstance(models, int) and models > 1:
        sentences.append(
            f"The deposition contains {models} models; analysis here uses the first."
        )

    n_poly = _get(entry, "rcsb_entry_info", "polymer_entity_count")
    n_nonpoly = _get(entry, "rcsb_entry_info", "nonpolymer_entity_count")
    counts: list[str] = []
    if isinstance(n_poly, int) and n_poly > 0:
        counts.append(f"{n_poly} polymer {'entity' if n_poly == 1 else 'entities'}")
    if isinstance(n_nonpoly, int) and n_nonpoly > 0:
        counts.append(
            f"{n_nonpoly} non-polymer {'entity' if n_nonpoly == 1 else 'entities'}"
        )
    if counts:
        sentences.append(f"It holds {_join(counts)}.")

    keywords = _keywords(entry)
    if keywords:
        sentences.append(f"Classified as {_join(keywords[:3])}.")

    if measured:
        sentences.append(_measured_sentence(measured))

    citation = _citation(entry)
    if citation:
        sentences.append(citation)

    return " ".join(s for s in sentences if s)


def _measured_sentence(measured: dict) -> str:
    """One sentence for values this repo computed, kept clearly separate."""
    bits: list[str] = []
    atoms = measured.get("n_atoms")
    residues = measured.get("n_residues")
    chains = measured.get("n_chains")
    if isinstance(atoms, int):
        bits.append(f"{atoms} atoms")
    if isinstance(residues, int):
        bits.append(f"{residues} residues")
    if isinstance(chains, int):
        bits.append(f"{chains} {'chain' if chains == 1 else 'chains'}")

    total = measured.get("total_sasa")
    buried = measured.get("buried_fraction")
    tail: list[str] = []
    if isinstance(total, (int, float)):
        tail.append(f"a total solvent-accessible surface of {total:,.0f} square angstroms")
    if isinstance(buried, (int, float)):
        tail.append(f"{buried * 100:.0f}% of residues buried")

    lead = "Computed from the coordinate file: "
    parts = []
    if bits:
        parts.append(_join(bits))
    parts.extend(tail)
    if not parts:
        return ""
    return lead + _join(parts) + "."


def summarize_pdbe(summary: dict, pdb_id: str) -> str:
    """Same idea against the EBI PDBe summary endpoint.

    Kept as a fallback because PDBe and RCSB occasionally disagree on which
    entries are reachable, and because having two independent shapes of input
    proves the templating is not tied to one vendor's schema.
    """
    key = pdb_id.lower()
    records = summary.get(key) if isinstance(summary, dict) else None
    if not isinstance(records, list) or not records:
        return f"No PDBe summary record for {pdb_id.upper()}."
    rec = records[0]

    sentences: list[str] = []
    title = _clean_text(rec.get("title"))
    if title:
        sentences.append(f"{pdb_id.upper()}: {_sentence_case(title)}.")
    methods = rec.get("experimental_method")
    if isinstance(methods, list) and methods:
        phrases = [
            METHOD_PHRASES.get(str(m).upper(), str(m).lower())
            for m in methods
            if _clean_text(m)
        ]
        if phrases:
            sentences.append(f"Determined by {_join(phrases)}.")
    entities = rec.get("number_of_entities")
    if isinstance(entities, dict):
        poly = entities.get("polypeptide")
        if isinstance(poly, int) and poly > 0:
            sentences.append(
                f"It holds {poly} polypeptide {'entity' if poly == 1 else 'entities'}."
            )
    date = _clean_text(rec.get("release_date"))
    if date and len(date) >= 4:
        sentences.append(f"Released {date[:4]}.")
    return " ".join(sentences)
