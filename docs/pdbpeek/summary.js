// Plain-English entry summary built by template, with no language model.
// The twin of pdbpeek/summary.py, and the reason the demo page needs no API
// key and no backend.
//
// Everything below is string formatting over fields from the RCSB entry
// endpoint. There is no model and no generation step, so the output cannot
// hallucinate: a sentence appears only when the field it reads is present and
// well formed, and every number in the text is copied from the entry record
// rather than predicted. The cost is that the prose is plainer than a
// language model's would be. That is the trade.

export const METHOD_PHRASES = {
  'X-RAY DIFFRACTION': 'X-ray crystallography',
  'SOLUTION NMR': 'solution NMR',
  'SOLID-STATE NMR': 'solid-state NMR',
  'ELECTRON MICROSCOPY': 'cryo-electron microscopy',
  'ELECTRON CRYSTALLOGRAPHY': 'electron crystallography',
  'NEUTRON DIFFRACTION': 'neutron diffraction',
  'FIBER DIFFRACTION': 'fiber diffraction',
  'SOLUTION SCATTERING': 'solution scattering',
  'POWDER DIFFRACTION': 'powder diffraction',
  'THEORETICAL MODEL': 'computational modelling',
};

// Methods for which "resolution" is not a meaningful quantity.
const RESOLUTIONLESS = new Set(['SOLUTION NMR', 'SOLID-STATE NMR', 'THEORETICAL MODEL']);

function get(obj, ...path) {
  let cur = obj;
  for (const key of path) {
    if (cur === null || typeof cur !== 'object') return null;
    cur = cur[key];
    if (cur === null || cur === undefined) return null;
  }
  return cur;
}

function cleanText(value) {
  if (typeof value !== 'string') return null;
  const text = value.split(/\s+/).filter(Boolean).join(' ');
  if (!text || text === '.' || text === '?') return null;
  return text;
}

// Soften the ALL CAPS titles older PDB entries use. Only applied when the
// text is entirely uppercase, so modern mixed-case titles and embedded
// acronyms are left alone.
function sentenceCase(text) {
  let out = text;
  const letters = [...text].filter((c) => /[a-z]/i.test(c));
  if (letters.length && letters.every((c) => c === c.toUpperCase())) {
    out = text.toLowerCase();
  }
  if (!out) return out;
  const chars = [...out];
  let capitalise = true;
  for (let i = 0; i < chars.length; i += 1) {
    if (capitalise && /[a-z]/i.test(chars[i])) {
      chars[i] = chars[i].toUpperCase();
      capitalise = false;
    } else if ('.!?'.includes(chars[i])) {
      capitalise = true;
    }
  }
  return chars.join('');
}

function joinList(parts) {
  if (parts.length === 0) return '';
  if (parts.length === 1) return parts[0];
  if (parts.length === 2) return `${parts[0]} and ${parts[1]}`;
  return `${parts.slice(0, -1).join(', ')}, and ${parts[parts.length - 1]}`;
}

function methodsOf(entry) {
  const raw = entry.exptl;
  if (!Array.isArray(raw)) return [];
  const out = [];
  for (const item of raw) {
    const method = cleanText(get(item, 'method'));
    if (method && !out.some((m) => m.toUpperCase() === method.toUpperCase())) {
      out.push(method);
    }
  }
  return out;
}

function keywordsOf(entry) {
  const seen = [];
  const lowered = new Set();
  for (const field of ['pdbx_keywords', 'text']) {
    const text = cleanText(get(entry, 'struct_keywords', field));
    if (!text) continue;
    for (const piece of text.split(',')) {
      const kw = piece.trim().toLowerCase();
      if (kw && !lowered.has(kw)) { lowered.add(kw); seen.push(kw); }
    }
  }
  return seen;
}

function citationOf(entry) {
  const cit = entry.rcsb_primary_citation;
  if (cit === null || typeof cit !== 'object') return null;

  const authors = cit.rcsb_authors;
  let first = Array.isArray(authors) && authors.length ? cleanText(authors[0]) : null;
  if (first && first.includes(',')) first = first.split(',')[0].trim();

  const journal = cleanText(cit.rcsb_journal_abbrev) || cleanText(cit.journal_abbrev);
  const year = typeof cit.year === 'number' ? String(cit.year) : cleanText(cit.year);

  const bits = [];
  if (first) {
    bits.push(Array.isArray(authors) && authors.length > 1 ? `${first} et al.` : first);
  }
  if (journal) bits.push(journal);
  if (year) bits.push(year);
  if (bits.length === 0) return null;

  let text = first ? `Published by ${bits.join(', ')}` : `Published in ${bits.join(', ')}`;
  const doi = cleanText(cit.pdbx_database_id_DOI);
  if (doi) text += ` (doi:${doi})`;
  return `${text}.`;
}

// One sentence for values this repo computed, kept clearly separate from the
// deposited metadata so a reader can tell them apart.
function measuredSentence(measured) {
  const bits = [];
  if (Number.isInteger(measured.nAtoms)) bits.push(`${measured.nAtoms} atoms`);
  if (Number.isInteger(measured.nResidues)) bits.push(`${measured.nResidues} residues`);
  if (Number.isInteger(measured.nChains)) {
    bits.push(`${measured.nChains} ${measured.nChains === 1 ? 'chain' : 'chains'}`);
  }

  const tail = [];
  if (typeof measured.totalSasa === 'number' && Number.isFinite(measured.totalSasa)) {
    const rounded = Math.round(measured.totalSasa).toLocaleString('en-US');
    tail.push(`a total solvent-accessible surface of ${rounded} square angstroms`);
  }
  if (typeof measured.buriedFraction === 'number'
      && Number.isFinite(measured.buriedFraction)) {
    tail.push(`${Math.round(measured.buriedFraction * 100)}% of residues buried`);
  }

  const parts = [];
  if (bits.length) parts.push(joinList(bits));
  parts.push(...tail);
  if (parts.length === 0) return '';
  return `Computed from the coordinate file: ${joinList(parts)}.`;
}

/**
 * Render a paragraph describing an RCSB entry record.
 *
 * @param {object} entry body of data.rcsb.org/rest/v1/core/entry/<id>
 * @param {object} [measured] values computed from the coordinate file
 */
export function summarizeEntry(entry, measured = null) {
  if (entry === null || typeof entry !== 'object' || Array.isArray(entry)) {
    throw new TypeError('entry must be the decoded entry JSON object');
  }

  const pdbId = (cleanText(entry.rcsb_id) || cleanText(get(entry, 'entry', 'id')) || '')
    .toUpperCase();
  const sentences = [];

  const title = cleanText(get(entry, 'struct', 'title'));
  const label = pdbId || 'This entry';
  sentences.push(title
    ? `${label}: ${sentenceCase(title)}.`
    : `${label} is a Protein Data Bank entry.`);

  const methods = methodsOf(entry);
  const resList = get(entry, 'rcsb_entry_info', 'resolution_combined');
  let resolution = null;
  if (Array.isArray(resList) && resList.length) {
    const parsed = Number(resList[0]);
    if (Number.isFinite(parsed)) resolution = parsed;
  }

  if (methods.length) {
    const phrases = methods.map((m) => METHOD_PHRASES[m.toUpperCase()] || m.toLowerCase());
    let clause = `Determined by ${joinList(phrases)}`;
    const resolutionless = methods.every((m) => RESOLUTIONLESS.has(m.toUpperCase()));
    if (resolution !== null && !resolutionless) {
      clause += ` at ${resolution} A resolution`;
    }
    sentences.push(`${clause}.`);
  } else if (resolution !== null) {
    sentences.push(`Refined to ${resolution} A resolution.`);
  }

  const models = get(entry, 'rcsb_entry_info', 'deposited_model_count');
  if (Number.isInteger(models) && models > 1) {
    sentences.push(
      `The deposition contains ${models} models; analysis here uses the first.`,
    );
  }

  const nPoly = get(entry, 'rcsb_entry_info', 'polymer_entity_count');
  const nNonPoly = get(entry, 'rcsb_entry_info', 'nonpolymer_entity_count');
  const counts = [];
  if (Number.isInteger(nPoly) && nPoly > 0) {
    counts.push(`${nPoly} polymer ${nPoly === 1 ? 'entity' : 'entities'}`);
  }
  if (Number.isInteger(nNonPoly) && nNonPoly > 0) {
    counts.push(`${nNonPoly} non-polymer ${nNonPoly === 1 ? 'entity' : 'entities'}`);
  }
  if (counts.length) sentences.push(`It holds ${joinList(counts)}.`);

  const keywords = keywordsOf(entry);
  if (keywords.length) sentences.push(`Classified as ${joinList(keywords.slice(0, 3))}.`);

  if (measured) {
    const sentence = measuredSentence(measured);
    if (sentence) sentences.push(sentence);
  }

  const citation = citationOf(entry);
  if (citation) sentences.push(citation);

  return sentences.filter(Boolean).join(' ');
}
