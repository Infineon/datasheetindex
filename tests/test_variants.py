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
        assert "tabulated in" not in out

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
