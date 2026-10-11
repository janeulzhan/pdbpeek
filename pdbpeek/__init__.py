"""pdbpeek: read a PDB structure file and measure it yourself.

The point of this package is that nothing interesting here is fetched. The
mmCIF reader, the Shrake-Rupley solvent-accessible surface area and the
residue contact map are all computed from coordinates in this repository's
own code. The only thing taken from a web service is entry metadata for the
summary, and that summary is a fixed template rather than generated text.
"""

from .contacts import DEFAULT_CUTOFF, ContactMap, contact_map, contact_pairs
from .mmcif import Atom, MmcifError, Structure, parse_file, parse_mmcif, select
from .radii import DEFAULT_RADIUS, PROBE_RADIUS, VDW_RADII, radius_for
from .sasa import (
    DEFAULT_N_POINTS,
    SasaResult,
    analytic_single_sphere,
    analytic_two_spheres,
    compute_sasa,
    golden_spiral_points,
    sasa_from_arrays,
)
from .summary import summarize_entry, summarize_pdbe

__version__ = "0.1.0"

__all__ = [
    "Atom",
    "ContactMap",
    "DEFAULT_CUTOFF",
    "DEFAULT_N_POINTS",
    "DEFAULT_RADIUS",
    "MmcifError",
    "PROBE_RADIUS",
    "SasaResult",
    "Structure",
    "VDW_RADII",
    "__version__",
    "analytic_single_sphere",
    "analytic_two_spheres",
    "compute_sasa",
    "contact_map",
    "contact_pairs",
    "golden_spiral_points",
    "parse_file",
    "parse_mmcif",
    "radius_for",
    "sasa_from_arrays",
    "select",
    "summarize_entry",
    "summarize_pdbe",
]
