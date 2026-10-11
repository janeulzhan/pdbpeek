// Wiring for the pdbpeek demo page.
//
// All of the analysis lives in ./pdbpeek/, which is the code the Python
// package is tested against. This file only fetches, calls into those
// modules, and draws the result. Nothing is computed here that is not also
// computed in pdbpeek/*.py and checked by tests/test_parity.py.
//
// Two RCSB endpoints are used directly from the browser because both send
// access-control-allow-origin: *, which is why this page needs no backend,
// no proxy and no API key:
//
//   https://files.rcsb.org/download/<ID>.cif     coordinates
//   https://data.rcsb.org/rest/v1/core/entry/<ID> metadata for the summary

import { parseMmcif, select, chains, MmcifError } from './pdbpeek/mmcif.js';
import { computeSasa, classifyBurial } from './pdbpeek/sasa.js';
import { contactMap, contactOrder, DEFAULT_CUTOFF } from './pdbpeek/contacts.js';
import { summarizeEntry } from './pdbpeek/summary.js';

const CIF_URL = 'https://files.rcsb.org/download/{id}.cif';
const ENTRY_URL = 'https://data.rcsb.org/rest/v1/core/entry/{id}';

const el = (id) => document.getElementById(id);

const ui = {
  form: el('lookup'),
  input: el('pdb-id'),
  go: el('go'),
  status: el('status'),
  summary: el('summary'),
  viewerHolder: el('viewer-holder'),
  nPoints: el('n-points'),
  nPointsValue: el('n-points-value'),
  accessibility: el('accessibility'),
  residueReadout: el('residue-readout'),
  canvas: el('contact-map'),
  contactReadout: el('contact-readout'),
  cutoffLabel: el('cutoff-label'),
  stats: {
    atoms: el('stat-atoms'),
    residues: el('stat-residues'),
    chains: el('stat-chains'),
    sasa: el('stat-sasa'),
    buried: el('stat-buried'),
    contacts: el('stat-contacts'),
    order: el('stat-order'),
    time: el('stat-time'),
  },
};

// Current state: the parsed structure, the entry record, and the last
// analysis. Kept so the point-count slider can recompute without refetching.
const state = {
  pdbId: null,
  structure: null,
  entry: null,
  analysis: null,
  cutoff: DEFAULT_CUTOFF,
  busy: false,
};

const fmt = {
  int: (v) => v.toLocaleString('en-US'),
  area: (v) => `${Math.round(v).toLocaleString('en-US')}`,
  one: (v) => v.toFixed(1),
  ms: (v) => {
    if (v >= 1000) return `${(v / 1000).toFixed(2)} s`;
    // Small structures finish in well under a millisecond, and rounding
    // those to "0 ms" reads like the measurement failed.
    return v < 10 ? `${v.toFixed(1)} ms` : `${Math.round(v)} ms`;
  },
};

function setStatus(message, stateName = '') {
  ui.status.textContent = message;
  if (stateName) ui.status.dataset.state = stateName;
  else delete ui.status.dataset.state;
}

function pointsFromSlider() {
  // The slider is an exponent so the useful range is reachable in a few
  // steps: 16 through 1024 test points per atom.
  return 2 ** Number(ui.nPoints.value);
}

// --------------------------------------------------------------------------
// fetching
// --------------------------------------------------------------------------

function validPdbId(value) {
  return /^[0-9][A-Z0-9]{3}$/.test(value);
}

async function fetchText(url, what) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(
      response.status === 404
        ? `No ${what} found for that ID.`
        : `Could not load ${what} (HTTP ${response.status}).`,
    );
  }
  return response.text();
}

// --------------------------------------------------------------------------
// analysis
// --------------------------------------------------------------------------

function analyse(structure, nPoints) {
  const started = performance.now();
  const selected = select(structure, { dropHetatm: true });
  const sasa = computeSasa(selected, { nPoints });
  const burial = classifyBurial(sasa.residues);
  const map = contactMap(selected, { cutoff: state.cutoff });
  const elapsed = performance.now() - started;

  const buried = burial.filter((b) => b === 'buried').length;

  return {
    selected,
    sasa,
    burial,
    map,
    elapsed,
    nPoints,
    buried,
    buriedFraction: burial.length ? buried / burial.length : 0,
    order: contactOrder(map),
  };
}

function renderStats(analysis) {
  const { selected, sasa, map, analysisOrder } = analysis;
  ui.stats.atoms.textContent = fmt.int(selected.atoms.length);
  ui.stats.residues.textContent = fmt.int(sasa.residues.length);
  ui.stats.chains.textContent = chains(selected).join(', ') || '0';
  ui.stats.sasa.textContent = `${fmt.area(sasa.total)} A²`;
  ui.stats.buried.textContent = `${analysis.buried} / ${sasa.residues.length}`;
  ui.stats.contacts.textContent = fmt.int(map.nContacts);
  ui.stats.order.textContent = fmt.one(analysis.order);
  ui.stats.time.textContent = fmt.ms(analysis.elapsed);
  void analysisOrder;
}

function renderSummary(entry, analysis) {
  if (!entry) {
    ui.summary.textContent =
      'Entry metadata was unavailable, so there is nothing to template. '
      + 'The measurements below were still computed from the coordinate file.';
    return;
  }
  const measured = {
    nAtoms: analysis.selected.atoms.length,
    nResidues: analysis.sasa.residues.length,
    nChains: chains(analysis.selected).length,
    totalSasa: analysis.sasa.total,
    buriedFraction: analysis.buriedFraction,
  };
  ui.summary.textContent = summarizeEntry(entry, measured);
}

// --------------------------------------------------------------------------
// accessibility bars
// --------------------------------------------------------------------------

function renderAccessibility(analysis) {
  const { sasa, burial } = analysis;
  const fragment = document.createDocumentFragment();

  sasa.residues.forEach((residue, i) => {
    const relative = residue.relative === null ? 0 : residue.relative;
    const bar = document.createElement('div');
    bar.className = 'bar';
    // Scale against 1.0, clamped, so the axis means something fixed rather
    // than rescaling to whatever the maximum happens to be.
    bar.style.height = `${Math.max(2, Math.min(1, relative) * 100)}%`;
    bar.dataset.burial = burial[i] || 'unknown';
    bar.dataset.index = String(i);
    bar.title = `${residue.name} ${residue.chain}${residue.seq}`;
    fragment.appendChild(bar);
  });

  ui.accessibility.replaceChildren(fragment);
  ui.residueReadout.textContent =
    `${sasa.residues.length} residues, ${analysis.buried} buried under 25% relative accessibility.`;
}

function describeResidue(analysis, i) {
  const residue = analysis.sasa.residues[i];
  if (!residue) return '';
  const relative = residue.relative === null
    ? 'n/a'
    : `${(residue.relative * 100).toFixed(0)}%`;
  return (
    `${residue.name} ${residue.chain}${residue.seq}   `
    + `${residue.sasa.toFixed(1)} A²   relative ${relative}   `
    + `${analysis.burial[i] || 'unknown'}`
  );
}

ui.accessibility.addEventListener('pointerover', (event) => {
  const bar = event.target.closest('.bar');
  if (!bar || !state.analysis) return;
  ui.accessibility.querySelectorAll('.bar.active')
    .forEach((n) => n.classList.remove('active'));
  bar.classList.add('active');
  ui.residueReadout.textContent = describeResidue(
    state.analysis, Number(bar.dataset.index),
  );
});

// --------------------------------------------------------------------------
// contact map
// --------------------------------------------------------------------------

function cssVar(name) {
  return getComputedStyle(document.body).getPropertyValue(name).trim();
}

function renderContactMap(analysis) {
  const { map } = analysis;
  const canvas = ui.canvas;
  const context = canvas.getContext('2d');
  const n = map.n;

  if (n === 0) {
    context.clearRect(0, 0, canvas.width, canvas.height);
    ui.contactReadout.textContent = 'No residues with a beta or alpha carbon.';
    return;
  }

  // One device pixel per residue pair, then scaled up by CSS. Keeps the draw
  // cheap for a 574 residue structure and keeps every pair visible.
  canvas.width = n;
  canvas.height = n;

  const image = context.createImageData(n, n);
  const background = hexToRgb(cssVar('--surface')) || [255, 255, 255];
  const near = hexToRgb(cssVar('--accent')) || [47, 125, 111];

  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      const offset = (i * n + j) * 4;
      const d = map.distances[i * n + j];
      let colour = background;
      let alpha = 255;
      if (i === j) {
        colour = near;
      } else if (map.contacts[i * n + j]) {
        // Closer pairs render darker, so the map shows distance and not just
        // a binary threshold.
        const t = 1 - d / map.cutoff;
        colour = mix(background, near, 0.25 + 0.75 * t);
      } else {
        alpha = 255;
      }
      image.data[offset] = colour[0];
      image.data[offset + 1] = colour[1];
      image.data[offset + 2] = colour[2];
      image.data[offset + 3] = alpha;
    }
  }

  context.putImageData(image, 0, 0);
  ui.contactReadout.textContent =
    `${n} residues, ${map.nContacts} contacting pairs under ${map.cutoff} A.`;
}

function hexToRgb(value) {
  const match = /^#?([0-9a-f]{6})$/i.exec(value);
  if (!match) return null;
  const int = Number.parseInt(match[1], 16);
  return [(int >> 16) & 255, (int >> 8) & 255, int & 255];
}

function mix(a, b, t) {
  return [
    Math.round(a[0] + (b[0] - a[0]) * t),
    Math.round(a[1] + (b[1] - a[1]) * t),
    Math.round(a[2] + (b[2] - a[2]) * t),
  ];
}

ui.canvas.addEventListener('pointermove', (event) => {
  if (!state.analysis) return;
  const { map } = state.analysis;
  if (map.n === 0) return;
  const rect = ui.canvas.getBoundingClientRect();
  const i = Math.floor(((event.clientY - rect.top) / rect.height) * map.n);
  const j = Math.floor(((event.clientX - rect.left) / rect.width) * map.n);
  if (i < 0 || j < 0 || i >= map.n || j >= map.n) return;
  const a = map.labels[i];
  const b = map.labels[j];
  const d = map.distances[i * map.n + j];
  ui.contactReadout.textContent =
    `${a.name} ${a.chain}${a.seq} to ${b.name} ${b.chain}${b.seq}   `
    + `${d.toFixed(2)} A   ${map.contacts[i * map.n + j] ? 'in contact' : 'apart'}`;
});

// --------------------------------------------------------------------------
// the 3D view
// --------------------------------------------------------------------------

function renderViewer(pdbId) {
  // pdbe-molstar is a custom element, so swapping the node is the simplest
  // way to load a different entry and avoids reaching into its internals.
  const viewer = document.createElement('pdbe-molstar');
  viewer.setAttribute('molecule-id', pdbId.toLowerCase());
  viewer.setAttribute('hide-controls', 'true');
  viewer.setAttribute('bg-color-r', '255');
  viewer.setAttribute('bg-color-g', '255');
  viewer.setAttribute('bg-color-b', '255');
  if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
    viewer.setAttribute('bg-color-r', '29');
    viewer.setAttribute('bg-color-g', '32');
    viewer.setAttribute('bg-color-b', '36');
  }
  ui.viewerHolder.replaceChildren(viewer);
}

// --------------------------------------------------------------------------
// the run loop
// --------------------------------------------------------------------------

function paint(analysis) {
  renderStats(analysis);
  renderAccessibility(analysis);
  renderContactMap(analysis);
  renderSummary(state.entry, analysis);
}

async function load(rawId) {
  const pdbId = String(rawId || '').trim().toUpperCase();
  if (!validPdbId(pdbId)) {
    setStatus('A PDB ID is four characters, starting with a digit, for example 1CRN.', 'error');
    return;
  }
  if (state.busy) return;

  state.busy = true;
  ui.go.disabled = true;
  setStatus(`Downloading ${pdbId} from RCSB.`, 'busy');

  try {
    const cif = await fetchText(CIF_URL.replace('{id}', pdbId), 'structure file');

    setStatus(`Parsing ${pdbId}.`, 'busy');
    const structure = parseMmcif(cif);

    state.pdbId = pdbId;
    state.structure = structure;
    ui.input.value = pdbId;
    renderViewer(pdbId);

    // Metadata is for the summary only, so a failure there must not lose the
    // analysis. Fetch it alongside and tolerate it going missing.
    state.entry = null;
    const entryPromise = fetchText(ENTRY_URL.replace('{id}', pdbId), 'entry record')
      .then((text) => JSON.parse(text))
      .catch(() => null);

    const nPoints = pointsFromSlider();
    setStatus(`Computing solvent accessibility at ${nPoints} points per atom.`, 'busy');

    // Yield once so the status text actually paints before the compute
    // blocks the main thread.
    await new Promise((resolve) => setTimeout(resolve, 0));

    const analysis = analyse(structure, nPoints);
    state.analysis = analysis;

    state.entry = await entryPromise;
    paint(analysis);

    const models = structure.models.length;
    setStatus(
      `${pdbId}: ${fmt.int(analysis.selected.atoms.length)} atoms analysed in `
      + `${fmt.ms(analysis.elapsed)}`
      + (models > 1 ? `, first of ${models} models.` : '.'),
    );
  } catch (error) {
    const message = error instanceof MmcifError
      ? `Could not read that structure file: ${error.message}`
      : error.message || String(error);
    setStatus(message, 'error');
  } finally {
    state.busy = false;
    ui.go.disabled = false;
  }
}

function recompute() {
  if (!state.structure || state.busy) return;
  const nPoints = pointsFromSlider();
  state.busy = true;
  setStatus(`Recomputing at ${nPoints} points per atom.`, 'busy');
  // Same yield as above, so the slider stays responsive.
  setTimeout(() => {
    try {
      const analysis = analyse(state.structure, nPoints);
      state.analysis = analysis;
      paint(analysis);
      setStatus(
        `${state.pdbId}: recomputed at ${nPoints} points per atom in `
        + `${fmt.ms(analysis.elapsed)}.`,
      );
    } catch (error) {
      setStatus(error.message || String(error), 'error');
    } finally {
      state.busy = false;
    }
  }, 0);
}

// --------------------------------------------------------------------------
// events
// --------------------------------------------------------------------------

ui.form.addEventListener('submit', (event) => {
  event.preventDefault();
  load(ui.input.value);
});

document.querySelectorAll('.examples .link').forEach((button) => {
  button.addEventListener('click', () => load(button.dataset.id));
});

ui.nPoints.addEventListener('input', () => {
  ui.nPointsValue.textContent = String(pointsFromSlider());
});

ui.nPoints.addEventListener('change', recompute);

ui.cutoffLabel.textContent = String(DEFAULT_CUTOFF);
ui.nPointsValue.textContent = String(pointsFromSlider());

// Expose the analysis for the screenshot script and for anyone poking at the
// page in a console. Read-only as far as the page is concerned.
window.pdbpeek = {
  state,
  analyse,
  load,
  get ready() {
    return state.analysis !== null && !state.busy;
  },
};

load(ui.input.value);
