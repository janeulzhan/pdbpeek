// Shrake-Rupley SASA in JavaScript, the twin of pdbpeek/sasa.py.
//
// The two implementations must agree, and tests/test_parity.py enforces that
// by running this file under node and comparing against the Python totals on
// the same fixture. Agreement is only possible because both sides generate
// the identical golden-spiral point set from the identical closed form and
// use an exact neighbour search, so nothing here is an approximation that
// the Python side does not also make.
//
// Python uses a scipy KD-tree; this uses a uniform grid of cells sized to the
// interaction cutoff. Both return every neighbour inside the cutoff and
// neither returns an atom outside it, so the choice does not affect results.

export const PROBE_RADIUS = 1.4;
export const DEFAULT_N_POINTS = 256;
export const DEFAULT_RADIUS = 1.8;

// Bondi (1964) van der Waals radii, identical table to pdbpeek/radii.py.
export const VDW_RADII = {
  H: 1.20,
  C: 1.70,
  N: 1.55,
  O: 1.52,
  F: 1.47,
  P: 1.80,
  S: 1.80,
  CL: 1.75,
  SE: 1.90,
  BR: 1.85,
  I: 1.98,
  NA: 2.27,
  MG: 1.73,
  K: 2.75,
  CA: 2.31,
  MN: 2.05,
  FE: 2.04,
  CO: 2.00,
  NI: 1.97,
  CU: 1.96,
  ZN: 2.01,
};

export function radiusFor(element) {
  const key = String(element || '').trim().toUpperCase();
  return Object.prototype.hasOwnProperty.call(VDW_RADII, key)
    ? VDW_RADII[key]
    : DEFAULT_RADIUS;
}

export function unknownElements(elements) {
  const out = new Set();
  for (const e of elements) {
    const key = String(e || '').trim().toUpperCase();
    if (key && !Object.prototype.hasOwnProperty.call(VDW_RADII, key)) out.add(key);
  }
  return [...out].sort();
}

// n roughly equidistant unit vectors, flattened to [x0,y0,z0, x1,...].
// Point k: y = 1 - (2k+1)/n, azimuth k * pi * (3 - sqrt 5).
export function goldenSpiralPoints(n) {
  if (!Number.isInteger(n) || n < 1) throw new Error('nPoints must be a positive integer');
  const out = new Float64Array(n * 3);
  const phi = Math.PI * (3.0 - Math.sqrt(5.0));
  for (let k = 0; k < n; k += 1) {
    const y = 1.0 - (2.0 * k + 1.0) / n;
    const r = Math.sqrt(Math.max(0.0, 1.0 - y * y));
    const theta = k * phi;
    out[k * 3] = Math.cos(theta) * r;
    out[k * 3 + 1] = y;
    out[k * 3 + 2] = Math.sin(theta) * r;
  }
  return out;
}

// Uniform grid over the coordinates, giving exact neighbour lists.
function buildGrid(coords, nAtoms, cell) {
  let minX = Infinity; let minY = Infinity; let minZ = Infinity;
  let maxX = -Infinity; let maxY = -Infinity; let maxZ = -Infinity;
  for (let i = 0; i < nAtoms; i += 1) {
    const x = coords[i * 3];
    const y = coords[i * 3 + 1];
    const z = coords[i * 3 + 2];
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }
  const nx = Math.max(1, Math.floor((maxX - minX) / cell) + 1);
  const ny = Math.max(1, Math.floor((maxY - minY) / cell) + 1);
  const nz = Math.max(1, Math.floor((maxZ - minZ) / cell) + 1);
  const buckets = new Map();
  const keyOf = (ax, ay, az) => (ax * ny + ay) * nz + az;
  const cellOf = (i) => [
    Math.min(nx - 1, Math.floor((coords[i * 3] - minX) / cell)),
    Math.min(ny - 1, Math.floor((coords[i * 3 + 1] - minY) / cell)),
    Math.min(nz - 1, Math.floor((coords[i * 3 + 2] - minZ) / cell)),
  ];
  for (let i = 0; i < nAtoms; i += 1) {
    const [ax, ay, az] = cellOf(i);
    const key = keyOf(ax, ay, az);
    const bucket = buckets.get(key);
    if (bucket === undefined) buckets.set(key, [i]); else bucket.push(i);
  }
  return { buckets, keyOf, cellOf, nx, ny, nz };
}

/**
 * Per-atom SASA for flat coordinates and van der Waals radii.
 *
 * @param {Float64Array|number[]} coords flattened xyz, length 3*N, angstroms
 * @param {Float64Array|number[]} radii van der Waals radii, length N, no probe
 * @returns {Float64Array} per-atom areas in square angstroms
 */
export function sasaFromArrays(coords, radii, options = {}) {
  const { probe = PROBE_RADIUS, nPoints = DEFAULT_N_POINTS } = options;
  const xyz = coords instanceof Float64Array ? coords : Float64Array.from(coords);
  const vdw = radii instanceof Float64Array ? radii : Float64Array.from(radii);
  const nAtoms = vdw.length;
  if (xyz.length !== nAtoms * 3) {
    throw new Error(`coords length ${xyz.length} does not match ${nAtoms} radii`);
  }
  const out = new Float64Array(nAtoms);
  if (nAtoms === 0) return out;

  const ext = new Float64Array(nAtoms);
  let maxExt = 0;
  for (let i = 0; i < nAtoms; i += 1) {
    ext[i] = vdw[i] + probe;
    if (ext[i] > maxExt) maxExt = ext[i];
  }

  const sphere = goldenSpiralPoints(nPoints);
  const areaPerPoint = (4.0 * Math.PI) / nPoints;
  const cutoff = maxExt * 2.0;
  const grid = buildGrid(xyz, nAtoms, cutoff);

  const alive = new Uint8Array(nPoints);
  const test = new Float64Array(nPoints * 3);
  const candIdx = [];
  const candDist = [];

  for (let i = 0; i < nAtoms; i += 1) {
    const ri = ext[i];
    const xi = xyz[i * 3];
    const yi = xyz[i * 3 + 1];
    const zi = xyz[i * 3 + 2];

    candIdx.length = 0;
    candDist.length = 0;
    const [cx, cy, cz] = grid.cellOf(i);
    for (let ax = cx - 1; ax <= cx + 1; ax += 1) {
      if (ax < 0 || ax >= grid.nx) continue;
      for (let ay = cy - 1; ay <= cy + 1; ay += 1) {
        if (ay < 0 || ay >= grid.ny) continue;
        for (let az = cz - 1; az <= cz + 1; az += 1) {
          if (az < 0 || az >= grid.nz) continue;
          const bucket = grid.buckets.get(grid.keyOf(ax, ay, az));
          if (bucket === undefined) continue;
          for (const j of bucket) {
            if (j === i) continue;
            const dx = xyz[j * 3] - xi;
            const dy = xyz[j * 3 + 1] - yi;
            const dz = xyz[j * 3 + 2] - zi;
            const d2 = dx * dx + dy * dy + dz * dz;
            const reach = ri + ext[j];
            if (d2 < reach * reach) { candIdx.push(j); candDist.push(d2); }
          }
        }
      }
    }

    if (candIdx.length === 0) {
      out[i] = areaPerPoint * ri * ri * nPoints;
      continue;
    }

    // Nearest first: the closest spheres bury the most points, so the
    // all-buried early exit below fires sooner.
    const order = candIdx.map((_, k) => k).sort((a, b) => candDist[a] - candDist[b]);

    for (let k = 0; k < nPoints; k += 1) {
      alive[k] = 1;
      test[k * 3] = xi + sphere[k * 3] * ri;
      test[k * 3 + 1] = yi + sphere[k * 3 + 1] * ri;
      test[k * 3 + 2] = zi + sphere[k * 3 + 2] * ri;
    }

    let nAlive = nPoints;
    for (const k of order) {
      const j = candIdx[k];
      const rj = ext[j];
      const rj2 = rj * rj;
      const xj = xyz[j * 3];
      const yj = xyz[j * 3 + 1];
      const zj = xyz[j * 3 + 2];
      for (let p = 0; p < nPoints; p += 1) {
        if (!alive[p]) continue;
        const dx = test[p * 3] - xj;
        const dy = test[p * 3 + 1] - yj;
        const dz = test[p * 3 + 2] - zj;
        if (dx * dx + dy * dy + dz * dz < rj2) { alive[p] = 0; nAlive -= 1; }
      }
      if (nAlive === 0) break;
    }

    out[i] = areaPerPoint * ri * ri * nAlive;
  }

  return out;
}

/**
 * SASA over a selected structure, rolled up per residue.
 *
 * With `reference: true` each residue is measured again on its own, with the
 * rest of the structure removed; the ratio is a relative accessibility that
 * needs no table of per-residue maxima.
 */
export function computeSasa(structure, options = {}) {
  const {
    probe = PROBE_RADIUS,
    nPoints = DEFAULT_N_POINTS,
    reference = true,
  } = options;
  const atoms = structure.atoms;
  const nAtoms = atoms.length;
  const coords = new Float64Array(nAtoms * 3);
  const vdw = new Float64Array(nAtoms);
  for (let i = 0; i < nAtoms; i += 1) {
    coords[i * 3] = atoms[i].x;
    coords[i * 3 + 1] = atoms[i].y;
    coords[i * 3 + 2] = atoms[i].z;
    vdw[i] = radiusFor(atoms[i].element);
  }

  const atomSasa = sasaFromArrays(coords, vdw, { probe, nPoints });

  const order = [];
  const members = new Map();
  for (let i = 0; i < nAtoms; i += 1) {
    const a = atoms[i];
    const key = `${a.chain}|${a.resSeq}${a.insCode || ''}|${a.resName}`;
    if (!members.has(key)) { members.set(key, []); order.push(key); }
    members.get(key).push(i);
  }

  const residues = order.map((key) => {
    const idx = members.get(key);
    const [chain, seq, name] = key.split('|');
    let sasa = 0;
    for (const i of idx) sasa += atomSasa[i];
    let ref = null;
    if (reference) {
      const rc = new Float64Array(idx.length * 3);
      const rr = new Float64Array(idx.length);
      idx.forEach((i, k) => {
        rc[k * 3] = coords[i * 3];
        rc[k * 3 + 1] = coords[i * 3 + 1];
        rc[k * 3 + 2] = coords[i * 3 + 2];
        rr[k] = vdw[i];
      });
      const parts = sasaFromArrays(rc, rr, { probe, nPoints });
      ref = 0;
      for (const v of parts) ref += v;
    }
    return {
      key,
      chain,
      seq,
      name,
      sasa,
      reference: ref,
      relative: ref && ref > 0 ? sasa / ref : null,
      nAtoms: idx.length,
    };
  });

  let total = 0;
  for (const v of atomSasa) total += v;

  return {
    atomSasa,
    radii: vdw,
    probe,
    nPoints,
    total,
    residues,
    unknownElements: unknownElements(atoms.map((a) => a.element)),
  };
}

export function classifyBurial(residues, threshold = 0.25) {
  return residues.map((r) => {
    if (r.relative === null || r.relative === undefined) return null;
    return r.relative < threshold ? 'buried' : 'exposed';
  });
}

// Exact SASA of one isolated atom.
export function analyticSingleSphere(radius, probe = PROBE_RADIUS) {
  return 4.0 * Math.PI * (radius + probe) ** 2;
}

// Exact SASA of two atoms d apart, from the spherical cap each removes.
export function analyticTwoSpheres(r1, r2, d, probe = PROBE_RADIUS) {
  const R1 = r1 + probe;
  const R2 = r2 + probe;
  if (d >= R1 + R2) return 4.0 * Math.PI * (R1 * R1 + R2 * R2);
  if (d <= Math.abs(R1 - R2)) {
    const big = Math.max(R1, R2);
    return 4.0 * Math.PI * big * big;
  }
  const h1 = R1 - (d * d + R1 * R1 - R2 * R2) / (2.0 * d);
  const h2 = R2 - (d * d + R2 * R2 - R1 * R1) / (2.0 * d);
  return (
    4.0 * Math.PI * R1 * R1 - 2.0 * Math.PI * R1 * h1
    + 4.0 * Math.PI * R2 * R2 - 2.0 * Math.PI * R2 * h2
  );
}
