// Node entry point for the JS/Python parity test.
//
// Run the browser modules, unmodified, over a local mmCIF file and print the
// results as JSON on stdout so tests/test_parity.py can compare them against
// the Python implementation's numbers. Nothing here is test-only logic: the
// imports below are exactly the files docs/index.html loads.
//
//   node docs/pdbpeek/parity_runner.js tests/fixtures/1CRN.cif --n-points 256
//
// Options:
//   --n-points N   SASA test points per atom (default 256)
//   --cutoff A     contact map CB-CB cutoff in angstroms (default 8)
//   --model N      model number (default: first in file)
//   --keep-waters  keep HETATM records
//   --per-atom     include every per-atom area, for a per-atom comparison

import { readFileSync } from 'node:fs';
import { parseMmcif, select, chains, residues } from './mmcif.js';
import { computeSasa, classifyBurial } from './sasa.js';
import { contactMap, contactOrder } from './contacts.js';

function parseArgs(argv) {
  const out = {
    path: null,
    nPoints: 256,
    cutoff: 8.0,
    model: null,
    keepWaters: false,
    perAtom: false,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--n-points') { out.nPoints = Number.parseInt(argv[i += 1], 10); continue; }
    if (arg === '--cutoff') { out.cutoff = Number.parseFloat(argv[i += 1]); continue; }
    if (arg === '--model') { out.model = Number.parseInt(argv[i += 1], 10); continue; }
    if (arg === '--keep-waters') { out.keepWaters = true; continue; }
    if (arg === '--per-atom') { out.perAtom = true; continue; }
    if (arg.startsWith('--')) throw new Error(`unknown option: ${arg}`);
    out.path = arg;
  }
  if (out.path === null) {
    throw new Error('usage: parity_runner.js <file.cif> [--n-points N]');
  }
  if (!Number.isInteger(out.nPoints) || out.nPoints < 1) {
    throw new Error('--n-points must be a positive integer');
  }
  return out;
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const raw = parseMmcif(readFileSync(args.path, 'utf8'));
  const sel = select(raw, { model: args.model, dropHetatm: !args.keepWaters });

  const sasa = computeSasa(sel, { nPoints: args.nPoints });
  const burial = classifyBurial(sasa.residues);
  const map = contactMap(sel, { cutoff: args.cutoff });

  const payload = {
    entry_id: raw.entryId,
    models_in_file: raw.models,
    n_atoms: sel.atoms.length,
    n_residues: residues(sel).length,
    n_chains: chains(sel).length,
    chains: chains(sel),
    sasa: {
      probe: sasa.probe,
      n_points: sasa.nPoints,
      total: sasa.total,
      unknown_elements: sasa.unknownElements,
    },
    contacts: {
      cutoff: map.cutoff,
      n_residues: map.n,
      n_contacts: map.nContacts,
      mean_contact_order: contactOrder(map),
    },
    residues: sasa.residues.map((r, i) => ({
      chain: r.chain,
      seq: r.seq,
      name: r.name,
      sasa: r.sasa,
      relative_sasa: r.relative,
      burial: burial[i],
    })),
  };

  if (args.perAtom) {
    payload.per_atom_sasa = Array.from(sasa.atomSasa);
  }

  process.stdout.write(`${JSON.stringify(payload)}\n`);
}

main();
