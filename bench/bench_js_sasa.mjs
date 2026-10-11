// Time the JavaScript SASA, for the cross-language comparison.
//
// Driven by bench/bench_cross_language.py, which runs the Python side with
// the same structures, the same point count and the same run count, so the
// two halves are measured the same way. Emits JSON on stdout.
//
//   node bench/bench_js_sasa.mjs --n-points 256 --runs 10 --warmup 2

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = dirname(HERE);
const FIXTURES = join(ROOT, 'tests', 'fixtures');

const { parseMmcif, select } = await import(
  join(ROOT, 'docs', 'pdbpeek', 'mmcif.js')
);
const { sasaFromArrays, radiusFor } = await import(
  join(ROOT, 'docs', 'pdbpeek', 'sasa.js')
);

function parseArgs(argv) {
  const out = { nPoints: 256, runs: 10, warmup: 2, entries: ['1CRN', '1UBQ', '2MBW', '4HHB'] };
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === '--n-points') { out.nPoints = Number.parseInt(argv[i += 1], 10); continue; }
    if (argv[i] === '--runs') { out.runs = Number.parseInt(argv[i += 1], 10); continue; }
    if (argv[i] === '--warmup') { out.warmup = Number.parseInt(argv[i += 1], 10); continue; }
    if (argv[i] === '--entries') { out.entries = argv[i += 1].split(','); continue; }
    throw new Error(`unknown option: ${argv[i]}`);
  }
  return out;
}

function mean(xs) { return xs.reduce((a, b) => a + b, 0) / xs.length; }

function stdev(xs) {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  return Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / (xs.length - 1));
}

const args = parseArgs(process.argv.slice(2));
const rows = [];

for (const entry of args.entries) {
  const structure = select(
    parseMmcif(readFileSync(join(FIXTURES, `${entry}.cif`), 'utf8')),
    { dropHetatm: true },
  );
  const n = structure.atoms.length;
  const coords = new Float64Array(n * 3);
  const radii = new Float64Array(n);
  structure.atoms.forEach((a, i) => {
    coords[i * 3] = a.x;
    coords[i * 3 + 1] = a.y;
    coords[i * 3 + 2] = a.z;
    radii[i] = radiusFor(a.element);
  });

  for (let k = 0; k < args.warmup; k += 1) {
    sasaFromArrays(coords, radii, { nPoints: args.nPoints });
  }

  const timings = [];
  let total = 0;
  for (let k = 0; k < args.runs; k += 1) {
    const started = performance.now();
    const areas = sasaFromArrays(coords, radii, { nPoints: args.nPoints });
    timings.push((performance.now() - started) / 1000);
    total = areas.reduce((a, b) => a + b, 0);
  }

  rows.push({
    entry,
    n_atoms: n,
    n_points: args.nPoints,
    mean_seconds: mean(timings),
    stdev_seconds: stdev(timings),
    atoms_per_second: n / mean(timings),
    total_sasa: Number(total.toFixed(6)),
  });
}

process.stdout.write(`${JSON.stringify({
  runtime: `node ${process.versions.node}`,
  v8: process.versions.v8,
  rows,
}, null, 2)}\n`);
