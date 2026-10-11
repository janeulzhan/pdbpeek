// Minimal mmCIF atom_site reader, the JS twin of pdbpeek/mmcif.py.
//
// Same grammar subset and the same selection rules, so the two
// implementations can be compared row for row. Runs unchanged in the browser
// and under node, which is how tests/test_parity.py drives it.

export class MmcifError extends Error {}

const NULL_TOKENS = new Set(['.', '?']);

// The 20 standard amino acids, plus selenomethionine and the standard
// nucleotides. Inside these residues the PDB atom naming convention puts the
// element in the first character of every heavy atom name, which is what lets
// elementFromName resolve CA to carbon rather than calcium.
const STANDARD_RESIDUES = new Set(
  ('ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP '
    + 'TYR VAL MSE SEC PYL ASX GLX UNK A C G U I DA DC DG DT DI DU N').split(' '),
);

const TWO_LETTER = new Set(
  'CL BR ZN FE MG MN NA CA CU SE NI CO K P S F I W MO CD HG PT AU AG'.split(' '),
);

// Split one CIF data line. A quote only opens a string at the start of a
// token and only closes it before whitespace, which is what keeps nucleic
// acid atom names like C1' intact.
export function splitLine(line) {
  const tokens = [];
  let i = 0;
  const n = line.length;
  while (i < n) {
    const ch = line[i];
    if (ch === ' ' || ch === '\t') { i += 1; continue; }
    if (ch === '#') break;
    if (ch === "'" || ch === '"') {
      const quote = ch;
      i += 1;
      const start = i;
      for (;;) {
        if (i >= n) throw new MmcifError(`unterminated quoted value: ${line}`);
        const next = line[i + 1];
        if (line[i] === quote && (i + 1 >= n || next === ' ' || next === '\t')) break;
        i += 1;
      }
      tokens.push(line.slice(start, i));
      i += 1;
      continue;
    }
    const start = i;
    while (i < n && line[i] !== ' ' && line[i] !== '\t') i += 1;
    tokens.push(line.slice(start, i));
  }
  return tokens;
}

function readSemicolonBlock(lines, i) {
  const body = [lines[i].slice(1)];
  i += 1;
  while (i < lines.length) {
    if (lines[i].startsWith(';')) return [body.join('\n').trim(), i + 1];
    body.push(lines[i]);
    i += 1;
  }
  throw new MmcifError('unterminated semicolon text field');
}

function tokensForRow(lines, i, want) {
  const row = [];
  while (row.length < want) {
    if (i >= lines.length) {
      throw new MmcifError(`loop row ended early: wanted ${want}, got ${row.length}`);
    }
    const line = lines[i];
    if (line.startsWith(';')) {
      const [value, next] = readSemicolonBlock(lines, i);
      row.push(value);
      i = next;
      continue;
    }
    for (const t of splitLine(line)) row.push(t);
    i += 1;
  }
  if (row.length > want) {
    throw new MmcifError(`loop row has ${row.length} values, expected ${want}`);
  }
  return [row, i];
}

// Two letter symbols are only considered outside standard polymer residues.
// Inside them the first character is authoritative, so the alpha carbon CA of
// an alanine reads as carbon and not calcium, and CD, CE, ND, NE do not turn
// into cadmium, cerium, neodymium and neon.
function elementFromName(name, resName = '') {
  const cleaned = (name || '').replace(/[^A-Za-z]/g, '');
  if (!cleaned) return '';
  if (STANDARD_RESIDUES.has(String(resName).trim().toUpperCase())) {
    return cleaned[0].toUpperCase();
  }
  const two = cleaned.slice(0, 2).toUpperCase();
  if (cleaned.length >= 2 && TWO_LETTER.has(two)) return two;
  return cleaned[0].toUpperCase();
}

const KEEP_ITEMS = new Set([
  '_struct.title',
  '_exptl.method',
  '_refine.ls_d_res_high',
  '_struct_keywords.pdbx_keywords',
  '_struct_keywords.text',
]);

function numberOr(value, fallback) {
  if (value === undefined || NULL_TOKENS.has(value)) return fallback;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function coord(value, field, serial) {
  if (value === undefined || NULL_TOKENS.has(value)) {
    throw new MmcifError(`atom ${serial} has no ${field} coordinate`);
  }
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    throw new MmcifError(`atom ${serial} has non-numeric ${field}: ${value}`);
  }
  return parsed;
}

export function parseMmcif(text) {
  if (!text || !text.trim()) throw new MmcifError('empty input');

  const lines = text.split(/\r\n|\r|\n/);
  const atoms = [];
  const items = {};
  const models = [];
  const seenModels = new Set();
  let entryId = '';
  let sawAtomSite = false;

  let i = 0;
  const n = lines.length;
  while (i < n) {
    const stripped = lines[i].trim();
    if (!stripped || stripped.startsWith('#')) { i += 1; continue; }
    const low = stripped.toLowerCase();

    if (low.startsWith('data_')) {
      entryId = stripped.slice(5).trim();
      i += 1;
      continue;
    }

    if (low === 'loop_') {
      i += 1;
      const tags = [];
      while (i < n) {
        const s = lines[i].trim();
        if (!s || s.startsWith('#')) { i += 1; continue; }
        if (!s.startsWith('_')) break;
        tags.push(s.split(/\s+/)[0]);
        i += 1;
      }
      if (tags.length === 0) throw new MmcifError('loop_ with no column headers');

      if (!tags[0].toLowerCase().startsWith('_atom_site.')) {
        while (i < n) {
          const s = lines[i].trim();
          const sl = s.toLowerCase();
          if (s.startsWith('_') || sl === 'loop_' || sl.startsWith('data_')) break;
          if (lines[i].startsWith(';')) { i = readSemicolonBlock(lines, i)[1]; continue; }
          i += 1;
        }
        continue;
      }

      sawAtomSite = true;
      const col = {};
      tags.forEach((t, k) => { col[t.toLowerCase()] = k; });
      const want = tags.length;
      const need = (name) => {
        const key = `_atom_site.${name}`;
        if (!(key in col)) throw new MmcifError(`_atom_site loop is missing ${key}`);
        return col[key];
      };
      const ix = need('cartn_x');
      const iy = need('cartn_y');
      const iz = need('cartn_z');
      const pick = (...names) => {
        for (const name of names) {
          const key = `_atom_site.${name}`;
          if (key in col) return col[key];
        }
        return undefined;
      };
      const iElem = pick('type_symbol');
      const iName = pick('label_atom_id', 'auth_atom_id');
      const iAlt = pick('label_alt_id');
      const iComp = pick('label_comp_id', 'auth_comp_id');
      const iAsym = pick('auth_asym_id', 'label_asym_id');
      const iSeq = pick('auth_seq_id', 'label_seq_id');
      const iIns = pick('pdbx_pdb_ins_code');
      const iOcc = pick('occupancy');
      const iB = pick('b_iso_or_equiv');
      const iModel = pick('pdbx_pdb_model_num');
      const iGroup = pick('group_pdb');
      const iSerial = pick('id');

      if (iComp === undefined || iAsym === undefined || iSeq === undefined) {
        throw new MmcifError('_atom_site loop lacks residue or chain identity');
      }

      while (i < n) {
        const s = lines[i].trim();
        if (!s || s.startsWith('#')) { i += 1; continue; }
        const sl = s.toLowerCase();
        if (s.startsWith('_') || sl === 'loop_' || sl.startsWith('data_')) break;
        const [row, next] = tokensForRow(lines, i, want);
        i = next;

        const serialTxt = iSerial !== undefined ? row[iSerial] : String(atoms.length + 1);
        const alt = iAlt !== undefined ? row[iAlt] : '.';
        const ins = iIns !== undefined ? row[iIns] : '.';
        const model = iModel !== undefined ? Math.trunc(numberOr(row[iModel], 1)) : 1;
        const atomName = iName !== undefined ? row[iName] : '';
        let element = iElem !== undefined ? row[iElem] : '';
        if (!element || NULL_TOKENS.has(element)) {
          element = elementFromName(atomName, row[iComp]);
        }
        const serial = Number.parseInt(serialTxt, 10);

        atoms.push({
          serial: Number.isFinite(serial) ? serial : atoms.length + 1,
          element: element.toUpperCase(),
          atomName,
          altId: NULL_TOKENS.has(alt) ? null : alt,
          resName: row[iComp],
          chain: row[iAsym],
          resSeq: row[iSeq],
          insCode: NULL_TOKENS.has(ins) ? null : ins,
          x: coord(row[ix], 'x', serialTxt),
          y: coord(row[iy], 'y', serialTxt),
          z: coord(row[iz], 'z', serialTxt),
          occupancy: iOcc !== undefined ? numberOr(row[iOcc], 1.0) : 1.0,
          bFactor: iB !== undefined ? numberOr(row[iB], 0.0) : 0.0,
          model,
          isHetatm: iGroup !== undefined ? row[iGroup].toUpperCase() === 'HETATM' : false,
        });

        if (!seenModels.has(model)) { seenModels.add(model); models.push(model); }
      }
      continue;
    }

    if (stripped.startsWith('_')) {
      const parts = splitLine(stripped);
      const tag = parts[0];
      let value;
      if (parts.length > 1) {
        value = parts.slice(1).join(' ');
        i += 1;
      } else {
        i += 1;
        while (i < n && !lines[i].trim()) i += 1;
        if (i < n && lines[i].startsWith(';')) {
          const [block, next] = readSemicolonBlock(lines, i);
          value = block;
          i = next;
        } else if (i < n) {
          value = splitLine(lines[i]).join(' ');
          i += 1;
        } else {
          value = '.';
        }
      }
      const key = tag.toLowerCase();
      if (KEEP_ITEMS.has(key) && !NULL_TOKENS.has(value)) items[key] = value;
      continue;
    }

    i += 1;
  }

  if (!sawAtomSite) {
    throw new MmcifError('no _atom_site loop found; this is not a coordinate file');
  }
  if (atoms.length === 0) throw new MmcifError('_atom_site loop contained no atoms');

  return { entryId, atoms, items, models: models.length ? models : [1] };
}

export function residueKey(atom) {
  return `${atom.chain}|${atom.resSeq}${atom.insCode || ''}|${atom.resName}`;
}

// Reduce to one model and one conformer per atom. Mirrors mmcif.select.
export function select(structure, options = {}) {
  const {
    model = null,
    chain = null,
    dropHydrogens = true,
    dropHetatm = false,
    altloc = 'occupancy',
  } = options;

  if (!['occupancy', 'first', 'all'].includes(altloc)) {
    throw new Error(`unknown altloc policy: ${altloc}`);
  }

  const chosen = model === null ? structure.models[0] : model;
  if (!structure.models.includes(chosen)) {
    throw new MmcifError(`model ${chosen} not in file (models: ${structure.models})`);
  }

  let atoms = structure.atoms.filter((a) => a.model === chosen);
  if (chain !== null) {
    atoms = atoms.filter((a) => a.chain === chain);
    if (atoms.length === 0) throw new MmcifError(`chain ${chain} not in model ${chosen}`);
  }
  if (dropHydrogens) atoms = atoms.filter((a) => a.element !== 'H' && a.element !== 'D');
  if (dropHetatm) atoms = atoms.filter((a) => !a.isHetatm);

  if (altloc !== 'all') {
    const best = new Map();
    for (const a of atoms) {
      const key = `${residueKey(a)}|${a.atomName}`;
      const cur = best.get(key);
      if (cur === undefined) {
        best.set(key, a);
      } else if (altloc === 'occupancy') {
        const better = a.occupancy > cur.occupancy
          || (a.occupancy === cur.occupancy && (a.altId || '') > (cur.altId || ''));
        if (better) best.set(key, a);
      }
    }
    atoms = [...best.values()];
  }

  if (atoms.length === 0) throw new MmcifError('selection left no atoms');
  return { entryId: structure.entryId, atoms, items: { ...structure.items }, models: [chosen] };
}

// Atoms grouped by residue, in file order.
export function residues(structure) {
  const order = [];
  const groups = new Map();
  for (const a of structure.atoms) {
    const key = residueKey(a);
    if (!groups.has(key)) { groups.set(key, []); order.push(key); }
    groups.get(key).push(a);
  }
  return order.map((key) => {
    const atoms = groups.get(key);
    const [chain, seq, name] = key.split('|');
    return { key, chain, seq, name, atoms };
  });
}

export function chains(structure) {
  const out = [];
  for (const a of structure.atoms) if (!out.includes(a.chain)) out.push(a.chain);
  return out;
}
