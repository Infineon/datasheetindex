"""Tests for multi-variant datasheet detection.

A datasheet is multi-variant when it covers more than one part number whose
FEATURES differ, so a per-part question cannot be answered from body text
alone. Package- or temperature-grade-only variation is NOT multi-variant --
those parts share one die and one feature set.

The title strings below are the real page-1 title blocks of the measurement
corpus, so a rule change that regresses one of them regresses a real document.
"""

import json
from pathlib import Path
from typing import cast

import pymupdf

from datasheetindex import DatasheetIndex
from datasheetindex.core.structure import build_tree
from datasheetindex.core.variants import detect_variants, title_text
from datasheetindex.models import TocNode


class TestFires:
    """Titles that state a family, and the rule each one exercises."""

    def test_wildcard_lowercase_x(self):
        signal = detect_variants("PSC3P5xD, PSC3M5xD 32-bit Arm Cortex-M33")
        assert signal is not None
        assert signal.rule == "wildcard"

    def test_wildcard_in_isolation(self):
        assert detect_variants("ADS111x Ultra-Small ADC") is not None

    def test_slash_list_of_bare_numbers(self):
        signal = detect_variants("PIC16F882/883/884/886/887 Data Sheet")
        assert signal is not None
        assert signal.rule == "slash-list"

    def test_explicit_list_sharing_a_prefix(self):
        signal = detect_variants("1N4001, 1N4002, 1N4003 Rectifier")
        assert signal is not None
        assert signal.rule == "list"

    def test_near_identical_differing_at_the_first_digit(self):
        # LM111/LM211/LM311 differ in position 2, which a shared-prefix
        # test cannot see.
        signal = detect_variants("LM111, LM211, LM311 Voltage Comparator")
        assert signal is not None
        assert signal.rule == "near-identical"

    def test_series_keyword_with_a_part_token(self):
        signal = detect_variants("ESP32 Series Datasheet")
        assert signal is not None
        assert signal.rule == "series"


class TestDoesNotFire:
    """Single-part titles. These carry the precision the flag depends on."""

    def test_single_part_number(self):
        assert detect_variants("BME280 Combined humidity and pressure sensor") is None

    def test_single_part_with_a_long_suffix(self):
        assert detect_variants("CC2640R2F SimpleLink Bluetooth Wireless MCU") is None

    def test_series_keyword_without_a_part_token(self):
        assert detect_variants("Series Resistance Measurement") is None

    def test_empty_title(self):
        assert detect_variants("") is None

    def test_prose_with_no_part_numbers(self):
        assert detect_variants("Ultra-low-power operational amplifier") is None


class TestSignalContent:
    """What the signal carries, since the agent is shown it verbatim."""

    def test_family_names_the_matched_text(self):
        signal = detect_variants("PSC3P5xD, PSC3M5xD 32-bit Arm Cortex-M33")
        assert signal is not None
        assert "PSC3P5xD" in signal.family

    def test_family_is_bounded(self):
        signal = detect_variants("ADS111x " + "filler " * 200)
        assert signal is not None
        assert len(signal.family) <= 120


class TestTitleText:
    """What the detector is fed.

    Measured across the corpus: the page-1 largest-font block reaches the
    stated family in 85% of multi-variant documents, PDF metadata alone in
    62%. Both are read, because they miss different documents.
    """

    def test_reads_the_largest_font_block_on_page_one(self):
        doc = pymupdf.open()
        page = doc.new_page(width=612, height=792)
        page.insert_textbox(
            pymupdf.Rect(50, 60, 560, 120), "ADS111x Precision ADC", fontsize=22
        )
        page.insert_textbox(
            pymupdf.Rect(50, 140, 560, 400),
            "The device is a low-power converter used in many designs.",
            fontsize=9,
        )
        text = title_text(doc)
        doc.close()
        assert "ADS111x" in text
        assert "low-power converter" not in text

    def test_includes_pdf_metadata_title(self):
        doc = pymupdf.open()
        doc.new_page(width=612, height=792)
        doc.set_metadata({"title": "LM111, LM211, LM311 Comparator"})
        text = title_text(doc)
        doc.close()
        assert "LM211" in text

    def test_empty_document_yields_empty_text(self):
        doc = pymupdf.open()
        doc.new_page(width=612, height=792)
        text = title_text(doc)
        doc.close()
        assert text.strip() == ""

    def test_body_text_far_below_the_title_is_not_included(self):
        """A page whose body is set only slightly smaller must still be cut.

        Guards the precision of the whole flag: if body text leaked in, every
        part number mentioned anywhere on page 1 would become evidence, which
        is the measured-and-rejected page-1-body detector.
        """
        doc = pymupdf.open()
        page = doc.new_page(width=612, height=792)
        page.insert_textbox(
            pymupdf.Rect(50, 60, 560, 120), "BME280 Humidity Sensor", fontsize=20
        )
        page.insert_textbox(
            pymupdf.Rect(50, 140, 560, 400),
            "Compatible with BME281 and BME282 evaluation boards.",
            fontsize=10,
        )
        text = title_text(doc)
        doc.close()
        assert detect_variants(text) is None


class TestBuildTreeCarriesTheFlag:
    """`build_tree` owns boilerplate flagging, so it must accept the signal."""

    def test_ordering_suppressed_when_multi_variant(self):
        raw = [[1, "Features", 1], [1, "Ordering Information", 3]]
        nodes = build_tree(raw, total_pages=5, multi_variant=True)
        ordering = next(n for n in nodes if n.title == "Ordering Information")
        assert ordering.boilerplate_category == ""

    def test_ordering_flagged_by_default(self):
        raw = [[1, "Features", 1], [1, "Ordering Information", 3]]
        nodes = build_tree(raw, total_pages=5)
        ordering = next(n for n in nodes if n.title == "Ordering Information")
        assert ordering.boilerplate_category == "ordering"


def _family_pdf(tmp_path, title, name="ds.pdf"):
    """A small datasheet whose page-1 title block is ``title``.

    Hermetic by construction. The bundled corpus PDFs are gitignored, so a
    test resting on one silently skips in the CI clone that gates releases.
    """
    doc = pymupdf.open()
    for index in range(4):
        page = doc.new_page(width=612, height=792)
        if index == 0:
            page.insert_textbox(pymupdf.Rect(50, 60, 560, 130), title, fontsize=22)
        page.insert_textbox(
            pymupdf.Rect(50, 160, 560, 700),
            f"Body text for page {index + 1} of this device datasheet.",
            fontsize=10,
        )
    doc.set_toc([[1, "Features", 1], [1, "Ordering Information", 3]])
    path = tmp_path / name
    doc.save(str(path))
    doc.close()
    return path


class TestBuildPublishesTheFlag:
    """End-to-end: the artifact carries the signal the agent is shown."""

    def test_multi_variant_datasheet_publishes_the_signal(self, tmp_path):
        pdf = _family_pdf(tmp_path, "ADS111x Precision Analog-to-Digital Converter")
        with DatasheetIndex(str(pdf)) as index:
            artifacts = index.build(output_dir=str(tmp_path / "out"))
        signal = artifacts.json_data["multi_variant"]
        assert signal["family"] == "ADS111x"
        assert signal["rule"] == "wildcard"

    def test_single_part_datasheet_omits_the_key(self, tmp_path):
        pdf = _family_pdf(tmp_path, "BME280 Humidity Sensor", name="single.pdf")
        with DatasheetIndex(str(pdf)) as index:
            artifacts = index.build(output_dir=str(tmp_path / "out"))
        assert "multi_variant" not in artifacts.json_data

    def test_ordering_section_is_unflagged_on_a_family_datasheet(self, tmp_path):
        """The mis-steer this whole change exists to remove."""
        pdf = _family_pdf(tmp_path, "ADS111x Precision Analog-to-Digital Converter")
        with DatasheetIndex(str(pdf)) as index:
            artifacts = index.build(output_dir=str(tmp_path / "out"))
        ordering = next(
            n
            for n in artifacts.json_data["toc"]
            if n["title"] == "Ordering Information"
        )
        assert "boilerplate_category" not in ordering


class TestTitleTextIsAdvisoryOnly:
    """A hint must never take a build down.

    ``title_text`` runs on every build over an arbitrary PDF, and page 1 is
    exactly where a malformed or partially-encrypted document fails first.
    The signal it feeds is advisory -- losing it costs a caution, while
    raising costs the whole artifact.
    """

    def test_unreadable_first_page_yields_empty_text(self):
        class _RaisingPage:
            def get_text(self, _kind):
                raise RuntimeError("cannot read page")

        class _RaisingDoc:
            metadata = {"title": ""}

            def __len__(self):
                return 1

            def __getitem__(self, _index):
                return _RaisingPage()

        assert title_text(cast("pymupdf.Document", _RaisingDoc())) == ""

    def test_metadata_still_used_when_the_page_is_unreadable(self):
        """Losing one source must not lose the other."""

        class _RaisingPage:
            def get_text(self, _kind):
                raise RuntimeError("cannot read page")

        class _RaisingDoc:
            metadata = {"title": "ADS111x Precision ADC"}

            def __len__(self):
                return 1

            def __getitem__(self, _index):
                return _RaisingPage()

        doc = cast("pymupdf.Document", _RaisingDoc())
        assert detect_variants(title_text(doc)) is not None


class TestManifestSurfacesTheSignal:
    """The manifest is the whole tool surface an MCP agent is handed.

    A signal computed so the agent can be cautious, that the agent never
    sees, is doing no work -- the lesson 0.25.0's figures digest was added
    to answer.
    """

    def _tools(self, json_data):
        from datasheetindex.models import DatasheetArtifacts
        from datasheetindex.tools.bound import DatasheetTools

        tools = DatasheetTools.__new__(DatasheetTools)
        tools._artifacts = DatasheetArtifacts(
            json_path=Path("d.json"),
            text_path=Path("d.txt"),
            json_data=json_data,
            toc_source="pdf_outline",
        )
        return tools

    def test_signal_is_published_when_present(self):
        from datasheetindex.tools.bound import DatasheetTools

        tools = self._tools(
            {
                "total_pages": 3,
                "toc": [{"title": "1 Overview", "start_page": 1}],
                "figures": [],
                "multi_variant": {"family": "PSC3P5xD, PSC3M5xD", "rule": "wildcard"},
            }
        )
        manifest = DatasheetTools.get_artifact_manifest(tools)
        signal = cast("dict[str, str]", manifest["multi_variant"])
        assert signal["family"] == "PSC3P5xD, PSC3M5xD"

    def test_key_is_absent_on_a_single_part_datasheet(self):
        """Absent, not null: the key costs tokens and says nothing when unset."""
        from datasheetindex.tools.bound import DatasheetTools

        tools = self._tools(
            {
                "total_pages": 3,
                "toc": [{"title": "1 Overview", "start_page": 1}],
                "figures": [],
            }
        )
        assert "multi_variant" not in DatasheetTools.get_artifact_manifest(tools)


class TestReadTimeNote:
    """The note lands where the agent actually goes wrong.

    In the observed failure the agent had the ToC, the ordering section and
    page 1 in context, then read a features section and answered from it. The
    build-time flag fires many turns earlier; this fires on the text itself.
    """

    def _tools(self, *, multi_variant=True, nodes=None, pages=90):
        from datasheetindex.models import DatasheetArtifacts
        from datasheetindex.tools.bound import DatasheetTools

        text = "\n".join(
            f"--- PAGE {n} ---\nBody of page {n}." for n in range(1, pages + 1)
        )
        json_data: dict = {"total_pages": pages, "toc": [], "figures": []}
        if multi_variant:
            json_data["multi_variant"] = {
                "family": "PSC3P5xD, PSC3M5xD",
                "rule": "wildcard",
            }
        tools = DatasheetTools.__new__(DatasheetTools)
        tools._artifacts = DatasheetArtifacts(
            json_data=json_data, text_content=text, nodes=nodes or []
        )
        return tools

    def test_note_names_the_family(self):
        from datasheetindex.tools.bound import DatasheetTools

        tools = self._tools()
        out = DatasheetTools.get_section_text(tools, 40, 41)
        assert "=== NOTE:" in out
        assert "PSC3P5xD, PSC3M5xD" in out

    def test_no_note_on_a_single_part_datasheet(self):
        from datasheetindex.tools.bound import DatasheetTools

        tools = self._tools(multi_variant=False)
        assert "product family" not in DatasheetTools.get_section_text(tools, 40, 41)

    def test_note_points_at_the_ordering_section(self):
        from datasheetindex.tools.bound import DatasheetTools

        nodes = [
            TocNode(title="4 Features", level=1, start_page=10, end_page=68),
            TocNode(
                title="8 Ordering information", level=1, start_page=69, end_page=83
            ),
        ]
        tools = self._tools(nodes=nodes)
        out = DatasheetTools.get_section_text(tools, 40, 41)
        assert "8 Ordering information" in out
        assert "69" in out

    def test_note_omits_the_pointer_when_no_ordering_section_exists(self):
        from datasheetindex.tools.bound import DatasheetTools

        nodes = [TocNode(title="4 Features", level=1, start_page=10, end_page=68)]
        tools = self._tools(nodes=nodes)
        out = DatasheetTools.get_section_text(tools, 40, 41)
        assert "product family" in out
        assert "Before answering, read" not in out

    def test_no_note_when_already_reading_the_ordering_section(self):
        """Inside the per-part table, the note would point at the current page."""
        from datasheetindex.tools.bound import DatasheetTools

        nodes = [
            TocNode(
                title="8 Ordering information", level=1, start_page=69, end_page=83
            ),
        ]
        tools = self._tools(nodes=nodes)
        assert "product family" not in DatasheetTools.get_section_text(tools, 70, 71)

    def test_note_precedes_the_section_text(self):
        """Framing before content, as the documented result order requires."""
        from datasheetindex.tools.bound import DatasheetTools

        tools = self._tools()
        out = DatasheetTools.get_section_text(tools, 40, 41)
        assert out.index("=== NOTE:") < out.index("--- PAGE 40 ---")


class TestAlwaysOnCaution:
    """The floor for the 15% of families the title rule does not detect.

    The flag is precision-1.00 but recall-0.85, so a miss must degrade to a
    standing instruction rather than to silence. These assert the instruction
    is present on both tools an agent uses to answer a per-part question.
    """

    def _defs(self):
        from datasheetindex.tools.defs import create_datasheet_tool_defs

        return {d.name: d for d in create_datasheet_tool_defs()}

    def test_build_datasheet_explains_the_multi_variant_key(self):
        description = self._defs()["build_datasheet"].description
        assert "multi_variant" in description

    def test_build_datasheet_states_the_rule_unconditionally(self):
        """Phrased for every datasheet, not only flagged ones."""
        description = self._defs()["build_datasheet"].description.lower()
        assert "family" in description
        assert "ordering" in description

    def test_get_section_text_warns_about_family_level_text(self):
        description = self._defs()["get_section_text"].description.lower()
        assert "family" in description


class TestDetectorPrecisionGuards:
    """Non-part tokens that look like part lists to a naive rule.

    Precision is the property this flag lives on -- a false positive prints
    an agent-visible NOTE naming something that is not a product family. Each
    case below was found by review, and each fires a different rule.
    """

    def test_a_revision_token_is_not_a_part_list(self):
        assert detect_variants("SGP30 Rev1/2020 Gas Sensor") is None

    def test_a_version_token_is_not_a_part_list(self):
        assert detect_variants("BME280 Ver2/2023 Humidity Sensor") is None

    def test_a_document_number_is_not_a_part_list(self):
        assert detect_variants("BMP388 Doc2/2019 Pressure Sensor") is None

    def test_bus_names_are_not_a_part_family(self):
        assert detect_variants("DDR3/DDR4 Memory Interface Controller") is None

    def test_interface_names_are_not_a_part_family(self):
        assert detect_variants("USB2.0 to USB3.0 bridge") is None

    def test_a_part_and_its_own_order_code_are_not_a_family(self):
        """title_text concatenates metadata with the page-1 block, so a base
        part and its order code routinely co-occur. Package suffixes are the
        one thing this detector must never read as a feature family."""
        assert detect_variants("TPS7A4901DGNR datasheet TPS7A4901 Rev C") is None

    def test_real_families_still_fire(self):
        """The guards above must not cost the corpus true positives."""
        for title in (
            "1N4001, 1N4002, 1N4003 Rectifier",
            "LM111, LM211, LM311 Voltage Comparator",
            "PIC16F882/883/884/886/887 Data Sheet",
            "ADS111x Ultra-Small ADC",
        ):
            assert detect_variants(title) is not None, title


class TestOrderingSectionSelection:
    """Which section the read-time note points the agent at."""

    def test_a_top_level_ordering_chapter_beats_a_nested_subsection(self):
        """`Part Numbering` under an early chapter classifies as `ordering`
        too, and pointing at a naming-convention subsection sends the agent
        to the wrong page."""
        from datasheetindex.tools.bound import _ordering_section

        nodes = [
            TocNode(
                title="2 Overview",
                level=1,
                start_page=3,
                end_page=20,
                nodes=[TocNode(title="2.4 Part Numbering", level=2, start_page=8)],
            ),
            TocNode(title="8 Ordering information", level=1, start_page=69),
        ]
        found = _ordering_section(nodes)
        assert found is not None
        assert found.start_page == 69

    def test_returns_none_when_no_section_classifies(self):
        from datasheetindex.tools.bound import _ordering_section

        assert (
            _ordering_section([TocNode(title="1 Features", level=1, start_page=1)])
            is None
        )


class TestLlmFallbackKeepsTheSuppression:
    """The fallback rebuilds the tree, so it must carry the flag too.

    This is the path taken on exactly the documents the fallback exists for --
    a weak or absent outline -- and on an explicit regenerate_toc=true.
    Without it the artifact publishes `multi_variant` and a
    `boilerplate_category: "ordering"` node at the same time.
    """

    def test_generate_toc_from_text_accepts_and_applies_the_flag(self):
        from datasheetindex.llm.toc_fallback import generate_toc_from_text

        canned = json.dumps(
            [
                {"level": 1, "title": "Features", "start_page": 1},
                {"level": 1, "title": "Ordering Information", "start_page": 3},
            ]
        )

        def fake_llm(system: str, user: str) -> str:
            """Parameter names match ``LlmCallable``, which is a Protocol."""
            return canned

        nodes = generate_toc_from_text(
            "--- PAGE 1 ---\nFeatures\n--- PAGE 3 ---\nOrdering\n",
            total_pages=4,
            llm_callable=fake_llm,
            multi_variant=True,
        )
        ordering = next(n for n in nodes if "Ordering" in n.title)
        assert ordering.boilerplate_category == ""


class TestWildcardIsAPartNumberWildcard:
    """`x` marks a varying character in a part number, not any letter x.

    The rule is that a wildcard token is written in vendor part-number casing
    -- uppercase and digits -- with a lowercase `x` standing in for what
    varies. Ordinary words fail that test, which is what keeps a core name or
    a lowercase filename out.
    """

    def test_an_arm_core_name_is_not_a_family(self):
        assert detect_variants("STM32L476 Arm Cortex-M4 32-bit MCU") is None

    def test_a_wide_core_name_is_not_a_family(self):
        assert detect_variants("XMC7100 Arm Cortex-M33 microcontroller") is None

    def test_a_lowercase_filename_is_not_a_family(self):
        assert detect_variants("max31855.pdf") is None

    def test_every_corpus_wildcard_still_fires(self):
        for title in (
            "ADS111x Ultra-Small ADC",
            "MSP430F552x, MSP430F551x Mixed-Signal Microcontrollers",
            "OPAx340 Rail-to-Rail Operational Amplifier",
            "SNx4HC595 8-Bit Shift Registers",
            "xx555 Precision Timers",
            "TLV906xS Operational Amplifiers",
            "PSC3P5xD, PSC3M5xD 32-bit Arm Cortex-M33",
        ):
            assert detect_variants(title) is not None, title


class TestPairRulesIgnoreCase:
    """An ALL-CAPS cover plus a mixed-case metadata title is one document.

    `title_text` concatenates both sources deliberately, so the same token
    routinely appears twice in different casing. Read case-sensitively, the
    two copies look like two parts.
    """

    def test_the_same_token_in_two_casings_is_not_a_family(self):
        title = "SN74HC595 8-Bit Shift Register SN74HC595 8-BIT SHIFT REGISTER"
        assert detect_variants(title) is None

    def test_a_bit_width_in_two_casings_is_not_a_family(self):
        assert detect_variants("ADC121S 12-Bit ADC 12-BIT, 50 kSPS") is None


class TestSeriesRuleNeedsAdjacency:
    """ "Series" must qualify the leading part token, not appear anywhere."""

    def test_a_package_code_before_series_is_not_a_family(self):
        assert detect_variants("MAX4173 Low-Cost SOT23 Series Current Monitor") is None

    def test_the_leading_part_token_still_fires(self):
        signal = detect_variants("ESP32 Series Datasheet")
        assert signal is not None
        assert signal.family == "ESP32"

    def test_a_hyphenated_leading_part_token_still_fires(self):
        assert detect_variants("ESP32-C3 Series Datasheet") is not None


class TestFamilyNamesEveryMatchedPart:
    """The note is the agent's only view of who the family covers."""

    def test_a_long_explicit_list_is_not_truncated_to_two(self):
        signal = detect_variants(
            "1N4001, 1N4002, 1N4003, 1N4004, 1N4005, 1N4006, 1N4007 Rectifier"
        )
        assert signal is not None
        # An agent asked about 1N4007 must not read a caution that names only
        # 1N4001 and 1N4002 and conclude its part is out of scope.
        assert "1N4007" in signal.family


class TestNoteSuppressionIsNarrow:
    """Suppression exists so the note does not point at the page being read.

    A range that merely *touches* the ordering section is not that case, and
    stripping the caution from wide reads removes it from exactly the reads
    where family-level body text is most likely to be misread.
    """

    def _tools(self, pages=120):
        from datasheetindex.models import DatasheetArtifacts
        from datasheetindex.tools.bound import DatasheetTools

        text = "\n".join(f"--- PAGE {n} ---\nBody {n}." for n in range(1, pages + 1))
        tools = DatasheetTools.__new__(DatasheetTools)
        tools._artifacts = DatasheetArtifacts(
            json_data={
                "total_pages": pages,
                "toc": [],
                "figures": [],
                "multi_variant": {"family": "ADS111x", "rule": "wildcard"},
            },
            text_content=text,
            nodes=[
                TocNode(title="4 Features", level=1, start_page=10, end_page=68),
                TocNode(
                    title="8 Ordering information", level=1, start_page=69, end_page=71
                ),
            ],
        )
        return tools

    def test_a_wide_read_spanning_the_section_keeps_the_note(self):
        from datasheetindex.tools.bound import DatasheetTools

        out = DatasheetTools.get_section_text(self._tools(), 60, 90)
        assert "product family" in out

    def test_a_read_inside_the_section_is_still_suppressed(self):
        from datasheetindex.tools.bound import DatasheetTools

        out = DatasheetTools.get_section_text(self._tools(), 69, 71)
        assert "product family" not in out

    def test_a_single_page_inside_the_section_is_suppressed(self):
        from datasheetindex.tools.bound import DatasheetTools

        out = DatasheetTools.get_section_text(self._tools(), 70, 70)
        assert "product family" not in out


class TestFamilyCoversEveryImplicatedPart:
    """A real TI title can implicate parts through more than one rule.

    "LM393B, LM2903B, LM193, LM293, LM393 and LM2903" pairs some tokens by
    shared prefix and others only by near-identity. Reporting one group's
    members alone names a family that omits parts the document covers, and
    an agent asked about an omitted part can read the caution as not
    applying to it.
    """

    def test_a_family_spanning_two_rules_names_both_halves(self):
        signal = detect_variants(
            "LM393B, LM2903B, LM193, LM293, LM393 and LM2903 Dual Comparators"
        )
        assert signal is not None
        for part in ("LM193", "LM293", "LM393", "LM2903"):
            assert part in signal.family, f"{part} missing from {signal.family!r}"

    def test_a_single_rule_family_is_unchanged(self):
        signal = detect_variants("TL431, TL432 Precision Programmable Reference")
        assert signal is not None
        assert signal.family == "TL431, TL432"


class TestNoteIsDirective:
    """The note names a prohibited action and a required next step.

    Measured, not stylistic. Against a live Sonnet agent asked a per-part
    question on a real family datasheet, n=10 per variant: a descriptive
    note ("may describe the family ... per-part differences are tabulated
    in X") answered correctly 1/10; this phrasing answered 10/10, Fisher
    exact p < 0.001. Every descriptive run stopped at 5 turns without
    opening the ordering table.

    These assertions are deliberately about the *shape* of the instruction
    rather than exact prose, so wording can be improved but not softened
    back into a description.
    """

    def _note(self, with_ordering=True):
        from datasheetindex.models import DatasheetArtifacts
        from datasheetindex.tools.bound import _variant_note

        nodes = (
            [
                TocNode(
                    title="8 Ordering information", level=1, start_page=69, end_page=71
                )
            ]
            if with_ordering
            else []
        )
        artifacts = DatasheetArtifacts(
            json_data={"multi_variant": {"family": "PSC3P5xD", "rule": "wildcard"}},
            nodes=nodes,
        )
        return _variant_note(artifacts, 22, 22)[0]

    def test_it_forbids_answering_from_this_text(self):
        assert "Do NOT report a per-part answer" in self._note()

    def test_it_requires_reading_the_per_part_table_first(self):
        note = self._note()
        assert "Before answering, read" in note
        assert "8 Ordering information" in note
        assert "69" in note

    def test_the_prohibition_survives_without_an_ordering_section(self):
        """The instruction not to answer from this text does not depend on
        our being able to name where the real answer lives."""
        assert "Do NOT report a per-part answer" in self._note(with_ordering=False)


class TestSeriesSurvivesTitleConcatenation:
    """`title_text` joins the metadata title to the page-1 block.

    So the family token routinely appears twice, and the copy carrying
    "Series" is rarely the first part-shaped token in the joined string.
    Anchoring the rule to the leading token alone silently loses the
    detection on the input shape this module actually receives.
    """

    def test_a_repeated_token_still_fires(self):
        assert detect_variants("ESP32 Datasheet ESP32 Series Datasheet") is not None

    def test_a_filename_style_metadata_prefix_still_fires(self):
        signal = detect_variants(
            "Infineon-XMC1400-DataSheet-v01_02-EN XMC1400 Series Datasheet"
        )
        assert signal is not None
        assert signal.family == "XMC1400"

    def test_an_order_code_prefix_still_fires(self):
        assert detect_variants("ADS1115IDGSR ADS1115 Series") is not None

    def test_a_package_code_before_series_is_still_rejected(self):
        """The precision this anchoring was added for must survive."""
        assert detect_variants("MAX4173 Low-Cost SOT23 Series Current Monitor") is None


class TestCoreNamesAreNotFamiliesInPairRules:
    """The wildcard guard closed only half of the Cortex problem.

    Two cores named in one title share a 3-character prefix and pass every
    structural test the pair rule applies, so a single-part dual-core MCU
    datasheet -- which is exactly this project's corpus; the motivating
    document is Cortex-M33 -- was published as a family called "Cortex-M33".
    """

    def test_two_arm_cores_are_not_a_family(self):
        assert (
            detect_variants("XMC7200 Arm Cortex-M7 and Cortex-M0 dual-core MCU") is None
        )

    def test_two_cores_in_another_vendor_style(self):
        assert detect_variants("RA6M4 Arm Cortex-M33 and Cortex-M23 MCU") is None

    def test_real_families_still_fire(self):
        for title in (
            "1N4001, 1N4002, 1N4003 Rectifier",
            "LM111, LM211, LM311 Voltage Comparator",
            "TL431, TL432 Precision Programmable Reference",
            "CD4017B, CD4022B TYPES",
        ):
            assert detect_variants(title) is not None, title


class TestFamilyTruncationNeverInventsAPart:
    """`family` is shown to the agent verbatim inside an instruction.

    Now that every matching part is named, the 120-character cap is
    load-bearing: cutting mid-token presents a part number that does not
    exist, in a note telling the agent to trust it.
    """

    def test_truncation_falls_on_a_separator(self):
        signal = detect_variants(
            ", ".join(f"TPS621{n}" for n in range(30, 50)) + " Step-Down Converters"
        )
        assert signal is not None
        assert signal.family.endswith("...")
        parts = [p for p in signal.family[:-3].split(", ") if p]
        for part in parts:
            assert len(part) == len("TPS62130"), f"truncated part {part!r}"


class TestMixedCaseVendorPartsAreParts:
    """Several vendors write part numbers in mixed case as house style.

    A rule that treats single-casing as the mark of a part number kills
    Microchip/Atmel and Nordic families outright. That is the costly
    direction: a false negative is a confident wrong answer, where a false
    positive is noise.
    """

    def test_atmel_megaavr_family(self):
        signal = detect_variants("ATmega48A, ATmega88A, ATmega168A, ATmega328P megaAVR")
        assert signal is not None
        assert "ATmega328P" in signal.family

    def test_atmel_tiny_family(self):
        assert detect_variants("ATtiny25, ATtiny45, ATtiny85 Datasheet") is not None

    def test_nordic_family(self):
        assert detect_variants("nRF52832 nRF52840 Multiprotocol SoC") is not None

    def test_title_cased_metadata_rendering(self):
        """`title_text` puts the metadata title first, and its casing wins the
        case-insensitive dedup, so a title-cased rendering is what survives."""
        assert detect_variants("Ads1113, Ads1114, Ads1115 ADS1113 ADS1115") is not None

    def test_arm_cores_are_still_rejected(self):
        """The precision case the single-case rule was reaching for."""
        assert (
            detect_variants("XMC7200 Arm Cortex-M7 and Cortex-M0 dual-core MCU") is None
        )
        assert detect_variants("RA6M4 Arm Cortex-M33 and Cortex-M23 MCU") is None


class TestSeriesOccurrenceScanIgnoresCase:
    """The dedup keeps the metadata spelling; the cover carries "Series".

    Scanning for occurrences case-sensitively can only find the spelling the
    metadata used, so the copy that actually carries the keyword is never
    examined -- the exact concatenation shape the rule exists for.
    """

    def test_lowercase_metadata_with_uppercase_cover(self):
        assert detect_variants("esp32 ESP32 Series") is not None

    def test_lowercase_metadata_with_words_between(self):
        assert detect_variants("esp32 datasheet ESP32 Series Datasheet") is not None


class TestCoreNamesAreExcludedPerToken:
    """A CPU core named in a title is not a part, and saying so directly
    is what two indirect proxies failed to do.

    A casing rule ("part numbers are single-case") lost ATmega and nRF
    families. A principal-part gate ("the family must include the leading
    token") lost any title whose leading part-shaped token is a filename,
    a descriptor or a document number -- and, being all-or-nothing over the
    combined group, let core names ride along whenever a real family
    co-occurred. Excluding the cores themselves has neither failure mode.
    """

    def test_cores_alone_do_not_make_a_family(self):
        assert (
            detect_variants("XMC7200 Arm Cortex-M7 and Cortex-M0 dual-core MCU") is None
        )

    def test_cores_do_not_ride_along_with_a_real_family(self):
        signal = detect_variants("RA6M4, RA6M5 Arm Cortex-M33 and Cortex-M23 MCU")
        assert signal is not None
        assert signal.family == "RA6M4, RA6M5"

    def test_cores_do_not_ride_along_when_named_first(self):
        signal = detect_variants(
            "Dual-Core Cortex-M7 and Cortex-M4 STM32H745 STM32H755"
        )
        assert signal is not None
        assert "Cortex" not in signal.family

    def test_a_leading_filename_no_longer_kills_the_family(self):
        signal = detect_variants(
            "Infineon-BSC0902NSI-DataSheet-v02_01-EN BSC0902NSI BSC0901NSI"
        )
        assert signal is not None
        assert "BSC0901NSI" in signal.family

    def test_a_leading_descriptor_no_longer_kills_the_family(self):
        assert detect_variants("Automotive 16-Bit ADS1113 ADS1114") is not None

    def test_a_leading_document_number_no_longer_kills_the_family(self):
        assert detect_variants("SBAS444H ADS1113 ADS1114 ADS1115") is not None
        assert detect_variants("Rev2 datasheet LM158, LM258, LM358") is not None
        assert detect_variants("Doc12345 TL431, TL432 Precision Reference") is not None


class TestNonPartVocabularyIsExcluded:
    """Tokens shaped like part numbers that name something else.

    Cores, instruction sets, interface standards, package codes and
    document furniture all take the form "letters, separator, digits", so
    two of them in one title satisfy every structural test the pair rules
    apply. Naming the vocabulary is the only approach that has not
    over-reached: two structural proxies were tried and each lost real
    families (see `_NON_PART_PREFIXES`).
    """

    def test_interface_standards(self):
        assert (
            detect_variants("SN65HVD75 Half-Duplex RS-485/RS-422 Transceiver") is None
        )
        assert detect_variants("MAX13487E RS-485 and RS-422 Transceivers") is None

    def test_package_codes(self):
        assert detect_variants("STM32F407 LQFP100 LQFP144 Package Options") is None

    def test_instruction_set_names(self):
        assert detect_variants("ESP32-C6 RISC-V RV32IMAC RV32IMC dual core") is None
        assert detect_variants("SiFive RISC-V64 and RISC-V32 cores") is None

    def test_document_furniture(self):
        for title in (
            "Figure1 and Figure2 show the block diagram of INA219",
            "Table12 and Table13 list the BME280 registers",
            "Grade1 Grade2 automotive qualification for TCAN1044A",
            "STM32F4 Discovery Kit UM1472 and UM1570 boards",
        ):
            assert detect_variants(title) is None, title

    def test_the_series_rule_honours_it_too(self):
        """The principle was enforced in two of three rule paths."""
        assert (
            detect_variants("Arm Cortex-M4 Series Technical Reference Manual") is None
        )
        assert detect_variants("Cortex-M55 Family Reference") is None

    def test_real_families_are_untouched(self):
        for title in (
            "ATmega48A, ATmega88A, ATmega328P megaAVR",
            "nRF52832 nRF52840 SoC",
            "1N4001, 1N4002, 1N4003 Rectifier",
            "LM111, LM211, LM311 Comparator",
            "TL431, TL432 Precision Reference",
            "SBAS444H ADS1113 ADS1114 ADS1115",
            "ESP32 Series Datasheet",
        ):
            assert detect_variants(title) is not None, title


class TestBoundedDegradesRatherThanFabricate:
    """When no whole part fits, name none rather than half of one.

    `_variant_note` already omits the parenthetical when `family` is empty,
    so the note degrades to "this datasheet covers a product family" -- true,
    and better than a part-number prefix that names nothing.
    """

    def test_a_single_over_budget_entry_yields_no_name(self):
        from datasheetindex.core.variants import _bounded

        assert _bounded(["A" + "B" * 200 + "x1"]) == ""

    def test_the_note_stays_well_formed_without_a_family(self):
        from datasheetindex.models import DatasheetArtifacts
        from datasheetindex.tools.bound import _variant_note

        artifacts = DatasheetArtifacts(
            json_data={"multi_variant": {"family": "", "rule": "wildcard"}}, nodes=[]
        )
        note = _variant_note(artifacts, 5, 5)[0]
        assert "product family." in note
        assert "()" not in note
