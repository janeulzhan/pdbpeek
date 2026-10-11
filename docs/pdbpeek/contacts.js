// Residue contact map from beta-carbon positions, the twin of
// pdbpeek/contacts.py.
//
// Two residues are in contact when their CB atoms fall within a cutoff,
// conventionally 8 A, and glycine substitutes CA because it has no CB. The
// map is built on real atoms only: a residue with neither CB nor CA, which
// happens for waters, ions and many ligands, is reported in `skipped` rather
// than given an invented position.

import { residues as residueGroups } from './mmcif.js';

export const DEFAULT_CUTOFF = 8.0;

function representative(atoms) {
  let ca = null;
  for (const a of atoms) {
    const name = a.atomName.toUpperCase();
    if (name === 'CB') return a;
    if (name === 'CA' && ca === null) ca = a;
  }
  return ca;
}

/**
 * Dense contact map for a selected structure.
 *
 * @returns {{residueKeys: string[], labels: object[], distances: Float64Array,
 *   contacts: Uint8Array, n: number, cutoff: number, skipped: string[],
 *   nContacts: number}}
 *   `distances` and `contacts` are row-major n by n.
 */
export function contactMap(structure, options = {}) {
  const { cutoff = DEFAULT_CUTOFF, includeHetatm = false } = options;
  if (!(cutoff > 0)) throw new Error('cutoff must be positive');

  const labels = [];
  const points = [];
  const skipped = [];

  for (const group of residueGroups(structure)) {
    if (!includeHetatm && group.atoms.every((a) => a.isHetatm)) {
      skipped.push(group.key);
      continue;
    }
    const rep = representative(group.atoms);
    if (rep === null) { skipped.push(group.key); continue; }
    labels.push({ key: group.key, chain: group.chain, seq: group.seq, name: group.name });
    points.push([rep.x, rep.y, rep.z]);
  }

  const n = labels.length;
  const distances = new Float64Array(n * n);
  const contacts = new Uint8Array(n * n);
  let nContacts = 0;

  for (let i = 0; i < n; i += 1) {
    for (let j = i + 1; j < n; j += 1) {
      const dx = points[i][0] - points[j][0];
      const dy = points[i][1] - points[j][1];
      const dz = points[i][2] - points[j][2];
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
      distances[i * n + j] = d;
      distances[j * n + i] = d;
      if (d < cutoff) {
        contacts[i * n + j] = 1;
        contacts[j * n + i] = 1;
        nContacts += 1;
      }
    }
  }

  return {
    residueKeys: labels.map((l) => l.key),
    labels,
    distances,
    contacts,
    n,
    cutoff,
    skipped,
    nContacts,
  };
}

/**
 * Mean sequence separation of contacting pairs inside a single chain.
 *
 * Low means the fold is dominated by local contacts, a helix bundle for
 * instance. High means long-range pairing, such as a beta sheet built from
 * strands that are far apart in sequence. Cross-chain pairs are excluded
 * because sequence separation is undefined across chains.
 */
export function contactOrder(map) {
  let sum = 0;
  let count = 0;
  for (let i = 0; i < map.n; i += 1) {
    for (let j = i + 1; j < map.n; j += 1) {
      if (!map.contacts[i * map.n + j]) continue;
      if (map.labels[i].chain !== map.labels[j].chain) continue;
      sum += j - i;
      count += 1;
    }
  }
  return count ? sum / count : 0.0;
}

/** Number of contacts per residue, excluding the diagonal. */
export function neighbourCounts(map) {
  const out = new Int32Array(map.n);
  for (let i = 0; i < map.n; i += 1) {
    let c = 0;
    for (let j = 0; j < map.n; j += 1) c += map.contacts[i * map.n + j];
    out[i] = c;
  }
  return out;
}
