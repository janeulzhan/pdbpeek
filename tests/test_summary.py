"""Tests for the deterministic summary template.

The property that matters here is not prose quality, it is that the output is
a pure function of the input record. Two things follow and are both pinned
down below: the same record always produces the same bytes (the golden file),
and a missing or null field drops its clause rather than leaking "None",
"null", "?" or "undefined" into the text.
"""

from __future__ import annotations

import json
import os

import pytest

from pdbpeek.summary import summarize_entry, summarize_pdbe

from .conftest import FIXTURES

GOLDEN = os.path.join(FIXTURES, "summary_1CRN.golden.txt")

#: Strings that mean the templating leaked an internal value into the prose.
LEAKS = ("None", "null", "nan", "NaN", "undefined", "?", "{", "}", "[", "]")


def _assert_no_leaks(text: str) -> None:
    for leak in LEAKS:
        assert leak not in text, f"{leak!r} leaked into: {text!r}"


# --------------------------------------------------------------------------
# golden file
# --------------------------------------------------------------------------


def test_matches_the_golden_file(entry_1crn):
    # Regenerate with:
    #   python3 -c "import json;from pdbpeek.summary import summarize_entry;\
    #     print(summarize_entry(json.load(open(
    #       'tests/fixtures/entry_1CRN.json'))), end='')" \
    #     > tests/fixtures/summary_1CRN.golden.txt
    with open(GOLDEN, "r", encoding="utf-8") as fh:
        expected = fh.read()
    assert summarize_entry(entry_1crn) == expected


def test_output_is_byte_stable_across_calls(entry_1crn):
    assert summarize_entry(entry_1crn) == summarize_entry(entry_1crn)


def test_golden_text_reads_as_prose(entry_1crn):
    text = summarize_entry(entry_1crn)
    _assert_no_leaks(text)
    assert text.startswith("1CRN: ")
    assert text.endswith(".")
    assert "X-ray crystallography" in text
    assert "1.5 A resolution" in text


def test_measured_values_are_labelled_as_computed(entry_1crn):
    measured = {
        "n_atoms": 327,
        "n_residues": 46,
        "n_chains": 1,
        "total_sasa": 2962.72,
        "buried_fraction": 0.4782,
    }
    text = summarize_entry(entry_1crn, measured=measured)
    assert "Computed from the coordinate file:" in text
    assert "327 atoms" in text
    assert "46 residues" in text
    assert "1 chain" in text and "1 chains" not in text
    assert "2,963 square angstroms" in text
    assert "48% of residues buried" in text
    _assert_no_leaks(text)


def test_measured_values_are_optional(entry_1crn):
    assert "Computed from" not in summarize_entry(entry_1crn)
    assert "Computed from" not in summarize_entry(entry_1crn, measured={})


def test_partial_measured_values_render_what_is_there(entry_1crn):
    text = summarize_entry(entry_1crn, measured={"n_atoms": 10})
    assert "10 atoms" in text
    assert "square angstroms" not in text
    _assert_no_leaks(text)


# --------------------------------------------------------------------------
# degrading on missing fields
# --------------------------------------------------------------------------


def test_empty_record_still_produces_a_sentence():
    text = summarize_entry({})
    assert text == "This entry is a Protein Data Bank entry."
    _assert_no_leaks(text)


def test_every_field_null_degrades_without_leaking():
    text = summarize_entry(
        {
            "rcsb_id": "9XYZ",
            "struct": {"title": None},
            "exptl": None,
            "rcsb_entry_info": {
                "resolution_combined": None,
                "polymer_entity_count": None,
                "nonpolymer_entity_count": None,
                "deposited_model_count": None,
            },
            "struct_keywords": {"pdbx_keywords": None, "text": None},
            "rcsb_primary_citation": None,
        }
    )
    assert text == "9XYZ is a Protein Data Bank entry."
    _assert_no_leaks(text)


@pytest.mark.parametrize("dropped", [
    "struct",
    "exptl",
    "rcsb_entry_info",
    "struct_keywords",
    "rcsb_primary_citation",
])
def test_dropping_any_single_field_is_safe(entry_1crn, dropped):
    record = dict(entry_1crn)
    record.pop(dropped, None)
    text = summarize_entry(record)
    assert text
    assert text.endswith(".")
    _assert_no_leaks(text)


def test_cif_null_markers_are_treated_as_absent():
    text = summarize_entry(
        {
            "rcsb_id": "1ABC",
            "struct": {"title": "?"},
            "struct_keywords": {"pdbx_keywords": ".", "text": "   "},
        }
    )
    assert text == "1ABC is a Protein Data Bank entry."
    _assert_no_leaks(text)


def test_wrongly_typed_fields_do_not_crash():
    text = summarize_entry(
        {
            "rcsb_id": "1ABC",
            "struct": "not a dict",
            "exptl": {"method": "X-RAY DIFFRACTION"},
            "rcsb_entry_info": [1, 2, 3],
            "struct_keywords": 42,
            "rcsb_primary_citation": "nope",
        }
    )
    assert text == "1ABC is a Protein Data Bank entry."
    _assert_no_leaks(text)


def test_unparseable_resolution_is_dropped():
    text = summarize_entry(
        {
            "rcsb_id": "1ABC",
            "exptl": [{"method": "X-RAY DIFFRACTION"}],
            "rcsb_entry_info": {"resolution_combined": ["not a number"]},
        }
    )
    assert "resolution" not in text
    assert "Determined by X-ray crystallography." in text
    _assert_no_leaks(text)


def test_non_dict_entry_raises():
    for bad in ([], "1CRN", 7, None):
        with pytest.raises(TypeError, match="decoded entry JSON object"):
            summarize_entry(bad)


# --------------------------------------------------------------------------
# field handling details
# --------------------------------------------------------------------------


def test_all_caps_title_is_softened_but_acronyms_survive():
    shouting = summarize_entry(
        {"rcsb_id": "1ABC", "struct": {"title": "STRUCTURE OF LYSOZYME"}}
    )
    assert "Structure of lysozyme" in shouting

    mixed = summarize_entry(
        {"rcsb_id": "1ABC", "struct": {"title": "Structure of HIV-1 protease"}}
    )
    assert "Structure of HIV-1 protease" in mixed, "mixed case must be left alone"


def test_sentence_boundaries_inside_a_title_are_recapitalised():
    text = summarize_entry(
        {"rcsb_id": "1ABC", "struct": {"title": "FIRST PART. SECOND PART"}}
    )
    assert "First part. Second part." in text


def test_resolution_is_omitted_for_nmr():
    text = summarize_entry(
        {
            "rcsb_id": "1L2Y",
            "exptl": [{"method": "SOLUTION NMR"}],
            "rcsb_entry_info": {"resolution_combined": [2.5]},
        }
    )
    assert text == "1L2Y: Determined by solution NMR." or "resolution" not in text
    assert "resolution" not in text, "NMR has no diffraction resolution"


def test_resolution_is_kept_for_crystallography():
    text = summarize_entry(
        {
            "rcsb_id": "1ABC",
            "exptl": [{"method": "X-RAY DIFFRACTION"}],
            "rcsb_entry_info": {"resolution_combined": [1.74]},
        }
    )
    assert "at 1.74 A resolution" in text


def test_resolution_without_a_method_still_reported():
    text = summarize_entry(
        {"rcsb_id": "1ABC", "rcsb_entry_info": {"resolution_combined": [2.0]}}
    )
    assert "Refined to 2 A resolution." in text


def test_multiple_methods_are_joined():
    text = summarize_entry(
        {
            "rcsb_id": "1ABC",
            "exptl": [
                {"method": "X-RAY DIFFRACTION"},
                {"method": "SOLUTION NMR"},
                {"method": "NEUTRON DIFFRACTION"},
            ],
        }
    )
    assert (
        "X-ray crystallography, solution NMR, and neutron diffraction" in text
    )


def test_duplicate_methods_are_collapsed():
    text = summarize_entry(
        {
            "rcsb_id": "1ABC",
            "exptl": [{"method": "X-RAY DIFFRACTION"}, {"method": "X-RAY DIFFRACTION"}],
        }
    )
    assert text.count("X-ray crystallography") == 1


def test_unmapped_method_is_passed_through_lowercased():
    text = summarize_entry(
        {"rcsb_id": "1ABC", "exptl": [{"method": "SOME NEW TECHNIQUE"}]}
    )
    assert "Determined by some new technique." in text


def test_multi_model_deposition_is_called_out():
    text = summarize_entry(
        {"rcsb_id": "1L2Y", "rcsb_entry_info": {"deposited_model_count": 38}}
    )
    assert "contains 38 models; analysis here uses the first" in text


def test_single_model_deposition_is_not_mentioned():
    text = summarize_entry(
        {"rcsb_id": "1CRN", "rcsb_entry_info": {"deposited_model_count": 1}}
    )
    assert "model" not in text


def test_entity_counts_are_pluralised():
    one = summarize_entry(
        {"rcsb_id": "1ABC", "rcsb_entry_info": {"polymer_entity_count": 1}}
    )
    assert "1 polymer entity." in one

    many = summarize_entry(
        {
            "rcsb_id": "1ABC",
            "rcsb_entry_info": {
                "polymer_entity_count": 4,
                "nonpolymer_entity_count": 2,
            },
        }
    )
    assert "4 polymer entities and 2 non-polymer entities" in many


def test_zero_entity_counts_are_omitted():
    text = summarize_entry(
        {
            "rcsb_id": "1ABC",
            "rcsb_entry_info": {
                "polymer_entity_count": 0,
                "nonpolymer_entity_count": 0,
            },
        }
    )
    assert "It holds" not in text


def test_keywords_are_deduplicated_and_capped():
    text = summarize_entry(
        {
            "rcsb_id": "1ABC",
            "struct_keywords": {
                "pdbx_keywords": "HYDROLASE",
                "text": "hydrolase, enzyme, serine protease, digestion, extra",
            },
        }
    )
    assert "Classified as hydrolase, enzyme, and serine protease." in text
    assert "digestion" not in text, "keyword list is capped at three"


def test_citation_with_one_author_omits_et_al():
    text = summarize_entry(
        {
            "rcsb_id": "1ABC",
            "rcsb_primary_citation": {
                "rcsb_authors": ["Teeter, M.M."],
                "rcsb_journal_abbrev": "Proc Natl Acad Sci U S A",
                "year": 1984,
            },
        }
    )
    assert "Published by Teeter, Proc Natl Acad Sci U S A, 1984." in text
    assert "et al" not in text


def test_citation_with_several_authors_uses_et_al():
    text = summarize_entry(
        {
            "rcsb_id": "1ABC",
            "rcsb_primary_citation": {
                "rcsb_authors": ["Neidigh, J.W.", "Fesinmeyer, R.M."],
                "rcsb_journal_abbrev": "Nat Struct Biol",
                "year": 2002,
                "pdbx_database_id_DOI": "10.1038/nsb798",
            },
        }
    )
    assert "Published by Neidigh et al., Nat Struct Biol, 2002 (doi:10.1038/nsb798)." in text


def test_citation_without_authors_says_published_in():
    text = summarize_entry(
        {
            "rcsb_id": "1ABC",
            "rcsb_primary_citation": {"rcsb_journal_abbrev": "Nature", "year": 2020},
        }
    )
    assert "Published in Nature, 2020." in text


def test_empty_citation_is_dropped():
    text = summarize_entry({"rcsb_id": "1ABC", "rcsb_primary_citation": {}})
    assert "Published" not in text
    _assert_no_leaks(text)


def test_id_falls_back_to_the_entry_subrecord():
    text = summarize_entry({"entry": {"id": "2xyz"}})
    assert text.startswith("2XYZ ")


# --------------------------------------------------------------------------
# the PDBe schema
# --------------------------------------------------------------------------


def test_pdbe_schema_renders():
    record = {
        "1crn": [
            {
                "title": "WATER STRUCTURE OF A HYDROPHOBIC PROTEIN",
                "experimental_method": ["X-ray diffraction"],
                "number_of_entities": {"polypeptide": 1},
                "release_date": "19810416",
            }
        ]
    }
    text = summarize_pdbe(record, "1CRN")
    assert text.startswith("1CRN: Water structure of a hydrophobic protein.")
    assert "X-ray crystallography" in text
    assert "1 polypeptide entity" in text
    assert "Released 1981." in text
    _assert_no_leaks(text)


def test_pdbe_missing_record_is_reported_not_crashed():
    assert summarize_pdbe({}, "9ZZZ") == "No PDBe summary record for 9ZZZ."
    assert summarize_pdbe({"9zzz": []}, "9ZZZ") == "No PDBe summary record for 9ZZZ."


def test_pdbe_partial_record_degrades():
    text = summarize_pdbe({"1abc": [{"title": "Something"}]}, "1ABC")
    assert text == "1ABC: Something."
    _assert_no_leaks(text)


# --------------------------------------------------------------------------
# cross-language agreement
# --------------------------------------------------------------------------


def test_golden_file_is_committed_and_non_trivial():
    assert os.path.exists(GOLDEN), "the golden file must be committed"
    with open(GOLDEN, "r", encoding="utf-8") as fh:
        text = fh.read()
    assert len(text) > 100
    assert "\n" not in text, "the summary is a single paragraph"


def test_entry_fixture_has_the_fields_the_template_reads(entry_1crn):
    # If RCSB ever renames one of these, the golden test would still pass on
    # the committed fixture while the live demo quietly lost a sentence. This
    # asserts the contract the template depends on.
    assert isinstance(entry_1crn["struct"]["title"], str)
    assert isinstance(entry_1crn["exptl"], list)
    assert isinstance(entry_1crn["exptl"][0]["method"], str)
    assert isinstance(entry_1crn["rcsb_entry_info"]["resolution_combined"], list)
    assert isinstance(entry_1crn["rcsb_entry_info"]["polymer_entity_count"], int)
    assert isinstance(entry_1crn["struct_keywords"], dict)
    assert isinstance(entry_1crn["rcsb_primary_citation"], dict)
