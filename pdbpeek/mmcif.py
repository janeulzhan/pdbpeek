"""A minimal mmCIF reader for the `atom_site` loop.

This is deliberately not a general CIF library. It implements the subset of
the mmCIF/STAR grammar that deposited PDB coordinate files actually use:

* `data_XXXX` block header
* single-line `_category.item value` pairs
* `loop_` headers followed by whitespace-separated value rows
* single-quoted, double-quoted and semicolon-delimited text fields
* `.` (inapplicable) and `?` (unknown) null markers

Column order inside `_atom_site` is read from the loop header rather than
assumed, because it is not fixed by the format.
"""

from __future__ import annotations

from dataclasses import dataclass


class MmcifError(ValueError):
    """Raised when the input is not mmCIF we can read."""


NULL_TOKENS = (".", "?")


@dataclass(frozen=True)
class Atom:
    """One row of `_atom_site`, with the fields SASA and contacts need."""

    serial: int
    element: str
    atom_name: str
    alt_id: str | None
    res_name: str
    chain: str
    res_seq: str
    ins_code: str | None
    x: float
    y: float
    z: float
    occupancy: float
    b_factor: float
    model: int
    is_hetatm: bool

    @property
    def residue_key(self) -> tuple[str, str, str]:
        """Identity of the residue this atom belongs to."""
        return (self.chain, self.res_seq + (self.ins_code or ""), self.res_name)


@dataclass
class Structure:
    """Parsed coordinates plus the non-loop items we kept."""

    entry_id: str
    atoms: list[Atom]
    items: dict[str, str]
    models: list[int]

    def __len__(self) -> int:
        return len(self.atoms)

    @property
    def chains(self) -> list[str]:
        out: list[str] = []
        for a in self.atoms:
            if a.chain not in out:
                out.append(a.chain)
        return out

    def residues(self) -> list[tuple[tuple[str, str, str], list[Atom]]]:
        """Atoms grouped by residue, in file order."""
        order: list[tuple[str, str, str]] = []
        groups: dict[tuple[str, str, str], list[Atom]] = {}
        for a in self.atoms:
            key = a.residue_key
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(a)
        return [(k, groups[k]) for k in order]

    def by_chain(self) -> dict[str, list[Atom]]:
        out: dict[str, list[Atom]] = {}
        for a in self.atoms:
            out.setdefault(a.chain, []).append(a)
        return out


# --------------------------------------------------------------------------
# tokenizing
# --------------------------------------------------------------------------


def _split_line(line: str) -> list[str]:
    """Split one CIF data line, honouring single and double quotes.

    A quote only opens a string at the start of a token, and only closes it
    when followed by whitespace or end of line. That is what the CIF 1.1 rule
    says, and it is why `C1'` and `O5'` atom names parse correctly.
    """
    tokens: list[str] = []
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if ch in " \t":
            i += 1
            continue
        if ch == "#":
            break
        if ch in "'\"":
            quote = ch
            i += 1
            start = i
            while True:
                if i >= n:
                    raise MmcifError(f"unterminated quoted value: {line!r}")
                if line[i] == quote and (i + 1 >= n or line[i + 1] in " \t"):
                    break
                i += 1
            tokens.append(line[start:i])
            i += 1
            continue
        start = i
        while i < n and line[i] not in " \t":
            i += 1
        tokens.append(line[start:i])
    return tokens


def _read_semicolon_block(lines: list[str], i: int) -> tuple[str, int]:
    """Read a multi-line `;`-delimited text field starting at lines[i]."""
    body = [lines[i][1:]]
    i += 1
    while i < len(lines):
        if lines[i].startswith(";"):
            return "\n".join(body).strip(), i + 1
        body.append(lines[i])
        i += 1
    raise MmcifError("unterminated semicolon text field")


def _tokens_for_row(lines: list[str], i: int, want: int) -> tuple[list[str], int]:
    """Collect `want` tokens starting at lines[i], spanning lines if needed."""
    row: list[str] = []
    while len(row) < want:
        if i >= len(lines):
            raise MmcifError(
                f"loop row ended early: wanted {want} values, got {len(row)}"
            )
        line = lines[i]
        if line.startswith(";"):
            value, i = _read_semicolon_block(lines, i)
            row.append(value)
            continue
        row.extend(_split_line(line))
        i += 1
    if len(row) > want:
        raise MmcifError(f"loop row has {len(row)} values, expected {want}")
    return row, i


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

#: Non-loop items worth keeping so an offline file can still be described.
KEEP_ITEMS = (
    "_struct.title",
    "_exptl.method",
    "_refine.ls_d_res_high",
    "_struct_keywords.pdbx_keywords",
    "_struct_keywords.text",
)


def _coord(value: str, field: str, serial: str) -> float:
    if value in NULL_TOKENS:
        raise MmcifError(f"atom {serial} has no {field} coordinate")
    try:
        return float(value)
    except ValueError as exc:
        raise MmcifError(f"atom {serial} has non-numeric {field}: {value!r}") from exc


def _number(value: str, default: float) -> float:
    if value in NULL_TOKENS:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def parse_mmcif(text: str) -> Structure:
    """Parse mmCIF text into a `Structure`.

    Every model and every altloc in the file is kept. Use `select` to reduce
    to a single model and a single conformer before computing anything.
    """
    if not text or not text.strip():
        raise MmcifError("empty input")

    lines = text.splitlines()
    entry_id = ""
    items: dict[str, str] = {}
    atoms: list[Atom] = []
    models: list[int] = []
    seen_models: set[int] = set()
    saw_atom_site_loop = False

    i = 0
    n = len(lines)
    while i < n:
        raw = lines[i]
        stripped = raw.strip()

        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        low = stripped.lower()

        if low.startswith("data_"):
            entry_id = stripped[5:].strip()
            i += 1
            continue

        if low == "loop_":
            i += 1
            tags: list[str] = []
            while i < n:
                s = lines[i].strip()
                if not s or s.startswith("#"):
                    i += 1
                    continue
                if not s.startswith("_"):
                    break
                tags.append(s.split()[0])
                i += 1

            if not tags:
                raise MmcifError("loop_ with no column headers")

            is_atom_site = tags[0].lower().startswith("_atom_site.")
            if not is_atom_site:
                # Skip the body of loops we do not care about.
                while i < n:
                    s = lines[i].strip()
                    if s.startswith("_") or s.lower() == "loop_" or s.lower().startswith("data_"):
                        break
                    if lines[i].startswith(";"):
                        _, i = _read_semicolon_block(lines, i)
                        continue
                    i += 1
                continue

            saw_atom_site_loop = True
            col = {t.lower(): k for k, t in enumerate(tags)}
            want = len(tags)

            def need(name: str) -> int:
                key = "_atom_site." + name
                if key not in col:
                    raise MmcifError(f"_atom_site loop is missing {key}")
                return col[key]

            ix, iy, iz = need("cartn_x"), need("cartn_y"), need("cartn_z")
            i_elem = col.get("_atom_site.type_symbol")
            i_name = col.get("_atom_site.label_atom_id", col.get("_atom_site.auth_atom_id"))
            i_alt = col.get("_atom_site.label_alt_id")
            i_comp = col.get("_atom_site.label_comp_id", col.get("_atom_site.auth_comp_id"))
            i_asym = col.get("_atom_site.auth_asym_id", col.get("_atom_site.label_asym_id"))
            i_seq = col.get("_atom_site.auth_seq_id", col.get("_atom_site.label_seq_id"))
            i_ins = col.get("_atom_site.pdbx_pdb_ins_code")
            i_occ = col.get("_atom_site.occupancy")
            i_b = col.get("_atom_site.b_iso_or_equiv")
            i_model = col.get("_atom_site.pdbx_pdb_model_num")
            i_group = col.get("_atom_site.group_pdb")
            i_serial = col.get("_atom_site.id")

            if i_comp is None or i_asym is None or i_seq is None:
                raise MmcifError("_atom_site loop lacks residue or chain identity")

            while i < n:
                s = lines[i].strip()
                if not s:
                    i += 1
                    continue
                if s.startswith("#"):
                    i += 1
                    continue
                if s.startswith("_") or s.lower() == "loop_" or s.lower().startswith("data_"):
                    break
                row, i = _tokens_for_row(lines, i, want)

                serial_txt = row[i_serial] if i_serial is not None else str(len(atoms) + 1)
                alt = row[i_alt] if i_alt is not None else "."
                ins = row[i_ins] if i_ins is not None else "."
                model = int(_number(row[i_model], 1.0)) if i_model is not None else 1
                name = row[i_name] if i_name is not None else ""
                element = row[i_elem] if i_elem is not None else ""
                if not element or element in NULL_TOKENS:
                    element = _element_from_name(name)

                try:
                    serial = int(serial_txt)
                except ValueError:
                    serial = len(atoms) + 1

                atoms.append(
                    Atom(
                        serial=serial,
                        element=element.upper(),
                        atom_name=name,
                        alt_id=None if alt in NULL_TOKENS else alt,
                        res_name=row[i_comp],
                        chain=row[i_asym],
                        res_seq=row[i_seq],
                        ins_code=None if ins in NULL_TOKENS else ins,
                        x=_coord(row[ix], "x", serial_txt),
                        y=_coord(row[iy], "y", serial_txt),
                        z=_coord(row[iz], "z", serial_txt),
                        occupancy=_number(row[i_occ], 1.0) if i_occ is not None else 1.0,
                        b_factor=_number(row[i_b], 0.0) if i_b is not None else 0.0,
                        model=model,
                        is_hetatm=(
                            row[i_group].upper() == "HETATM" if i_group is not None else False
                        ),
                    )
                )
                if model not in seen_models:
                    seen_models.add(model)
                    models.append(model)
            continue

        if stripped.startswith("_"):
            parts = _split_line(stripped)
            tag = parts[0]
            if len(parts) > 1:
                value = " ".join(parts[1:])
                i += 1
            else:
                i += 1
                while i < n and not lines[i].strip():
                    i += 1
                if i < n and lines[i].startswith(";"):
                    value, i = _read_semicolon_block(lines, i)
                elif i < n:
                    value = " ".join(_split_line(lines[i]))
                    i += 1
                else:
                    value = "."
            if tag.lower() in KEEP_ITEMS and value not in NULL_TOKENS:
                items[tag.lower()] = value
            continue

        i += 1

    if not saw_atom_site_loop:
        raise MmcifError("no _atom_site loop found; this is not a coordinate file")
    if not atoms:
        raise MmcifError("_atom_site loop contained no atoms")

    return Structure(entry_id=entry_id, atoms=atoms, items=items, models=models or [1])


def _element_from_name(name: str) -> str:
    """Last-resort element guess for files with no `type_symbol` column."""
    cleaned = "".join(c for c in name if c.isalpha())
    if not cleaned:
        return ""
    two = cleaned[:2].upper()
    if two in ("CL", "BR", "ZN", "FE", "MG", "MN", "NA", "CA", "CU", "SE", "NI", "CO"):
        return two
    return cleaned[0].upper()


def parse_file(path: str) -> Structure:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return parse_mmcif(fh.read())


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------


def select(
    structure: Structure,
    model: int | None = None,
    chain: str | None = None,
    drop_hydrogens: bool = True,
    drop_hetatm: bool = False,
    altloc: str = "occupancy",
) -> Structure:
    """Reduce a structure to one model and one conformer per atom.

    `altloc="occupancy"` keeps, for each (residue, atom name), the alternate
    conformer with the highest occupancy, breaking ties on the alternate id so
    the choice is deterministic. `altloc="first"` keeps whichever came first in
    the file. `altloc="all"` keeps every conformer, which double counts atoms
    and is only useful for inspecting a file.
    """
    if altloc not in ("occupancy", "first", "all"):
        raise ValueError(f"unknown altloc policy: {altloc!r}")

    chosen_model = model if model is not None else structure.models[0]
    if chosen_model not in structure.models:
        raise MmcifError(
            f"model {chosen_model} not in file (models present: {structure.models})"
        )

    atoms = [a for a in structure.atoms if a.model == chosen_model]
    if chain is not None:
        atoms = [a for a in atoms if a.chain == chain]
        if not atoms:
            raise MmcifError(f"chain {chain!r} not in model {chosen_model}")
    if drop_hydrogens:
        atoms = [a for a in atoms if a.element not in ("H", "D")]
    if drop_hetatm:
        atoms = [a for a in atoms if not a.is_hetatm]

    if altloc != "all":
        best: dict[tuple[tuple[str, str, str], str], Atom] = {}
        order: list[tuple[tuple[str, str, str], str]] = []
        for a in atoms:
            key = (a.residue_key, a.atom_name)
            if key not in best:
                best[key] = a
                order.append(key)
            elif altloc == "occupancy":
                cur = best[key]
                if (a.occupancy, a.alt_id or "") > (cur.occupancy, cur.alt_id or ""):
                    best[key] = a
        atoms = [best[k] for k in order]

    if not atoms:
        raise MmcifError("selection left no atoms")

    return Structure(
        entry_id=structure.entry_id,
        atoms=atoms,
        items=dict(structure.items),
        models=[chosen_model],
    )
