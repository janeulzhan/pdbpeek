"""Van der Waals radii and the solvent probe radius.

Radii are Bondi's values (A. Bondi, "van der Waals Volumes and Radii",
J. Phys. Chem. 68:441-451, 1964), which is the set most Shrake-Rupley
implementations use. The 1.4 A probe approximates the radius of a water
oxygen and is the value used in Shrake & Rupley (1973).

Anything not in the table falls back to DEFAULT_RADIUS. That is a real
approximation, not a silent one: `unknown_elements()` reports which
elements in a structure were given the fallback so a caller can decide
whether the number is trustworthy.
"""

from __future__ import annotations

PROBE_RADIUS = 1.4

#: Element symbol (uppercase) -> van der Waals radius in angstroms.
VDW_RADII: dict[str, float] = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "CL": 1.75,
    "SE": 1.90,
    "BR": 1.85,
    "I": 1.98,
    # Common metals and ions in deposited structures. Bondi does not cover
    # metals, so these are ionic/metallic radii in wide use.
    "NA": 2.27,
    "MG": 1.73,
    "K": 2.75,
    "CA": 2.31,
    "MN": 2.05,
    "FE": 2.04,
    "CO": 2.00,
    "NI": 1.97,
    "CU": 1.96,
    "ZN": 2.01,
}

#: Used when an element is absent from VDW_RADII.
DEFAULT_RADIUS = 1.80


def radius_for(element: str) -> float:
    """Return the van der Waals radius for an element symbol."""
    return VDW_RADII.get(element.strip().upper(), DEFAULT_RADIUS)


def unknown_elements(elements) -> list[str]:
    """Return the sorted distinct elements that would use DEFAULT_RADIUS."""
    seen = {e.strip().upper() for e in elements}
    return sorted(e for e in seen if e and e not in VDW_RADII)
