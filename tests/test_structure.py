"""Tests for ToC tree building and enrichment."""

import concurrent.futures
from pathlib import Path

import pymupdf
import pytest

from datasheetindex.core import structure
from datasheetindex.core.structure import (
    _available_cpus,
    _build_table_count_cache_parallel,
    _mp_context,
    _read_cgroup_cpu_quota,
    assign_breadcrumbs,
    build_tree,
    enrich_with_table_counts,
    extract_toc,
    find_breadcrumb_for_page,
)
from datasheetindex.models import TocNode

DATA2PAGE_DIR = Path(__file__).resolve().parent.parent.parent / "data2page"
TLE9350_PATH = DATA2PAGE_DIR / "Infineon-TLE9350BSJ-DataSheet-v01_00-EN.pdf"


# --- Unit tests with synthetic ToC data ---


def test_empty_toc():
    nodes = build_tree([], total_pages=10)
    assert nodes == []


def test_single_entry():
    raw = [[1, "Overview", 1]]
    nodes = build_tree(raw, total_pages=5)
    assert len(nodes) == 1
    assert nodes[0].title == "Overview"
    assert nodes[0].start_page == 1
    assert nodes[0].end_page == 5
    assert nodes[0].node_id == "0001"


def test_invalid_toc_entry_shape_raises():
    raw = [[1, "Overview"]]
    with pytest.raises(ValueError, match="must include"):
        build_tree(raw, total_pages=5)


def test_invalid_toc_level_raises():
    raw = [[0, "Overview", 1]]
    with pytest.raises(ValueError, match="Invalid ToC level"):
        build_tree(raw, total_pages=5)


def test_flat_entries():
    raw = [
        [1, "A", 1],
        [1, "B", 3],
        [1, "C", 6],
    ]
    nodes = build_tree(raw, total_pages=10)
    assert len(nodes) == 3
    assert nodes[0].end_page == 2  # next sibling starts at 3
    assert nodes[1].end_page == 5  # next sibling starts at 6
    assert nodes[2].end_page == 10  # last node gets parent_end


def test_nested_two_levels():
    raw = [
        [1, "Section 1", 1],
        [2, "Sub 1.1", 1],
        [2, "Sub 1.2", 3],
        [1, "Section 2", 5],
    ]
    nodes = build_tree(raw, total_pages=10)
    assert len(nodes) == 2
    assert nodes[0].title == "Section 1"
    assert len(nodes[0].nodes) == 2
    assert nodes[0].nodes[0].title == "Sub 1.1"
    assert nodes[0].nodes[0].end_page == 2
    assert nodes[0].nodes[1].title == "Sub 1.2"
    assert nodes[0].nodes[1].end_page == 4  # parent end is 4

    assert nodes[1].title == "Section 2"
    assert nodes[1].end_page == 10


def test_deep_three_levels():
    raw = [
        [1, "Ch1", 1],
        [2, "Sec1.1", 1],
        [3, "Sub1.1.1", 1],
        [3, "Sub1.1.2", 3],
        [2, "Sec1.2", 5],
        [1, "Ch2", 8],
    ]
    nodes = build_tree(raw, total_pages=12)
    assert len(nodes) == 2
    ch1 = nodes[0]
    assert ch1.end_page == 7
    assert len(ch1.nodes) == 2
    sec11 = ch1.nodes[0]
    assert sec11.end_page == 4
    assert len(sec11.nodes) == 2
    assert sec11.nodes[0].end_page == 2
    assert sec11.nodes[1].end_page == 4


def test_breadcrumb_root_node():
    raw = [[1, "Overview", 1]]
    nodes = build_tree(raw, total_pages=5)
    assert nodes[0].breadcrumb == "Overview"


def test_breadcrumb_nested():
    raw = [
        [1, "5 Electrical Characteristics", 10],
        [2, "5.1 Absolute Maximum Ratings", 10],
        [3, "5.1.1 Junction Temperature", 11],
        [1, "6 Pin Configuration", 15],
    ]
    nodes = build_tree(raw, total_pages=20)
    assert nodes[0].breadcrumb == "5 Electrical Characteristics"
    assert (
        nodes[0].nodes[0].breadcrumb
        == "5 Electrical Characteristics > 5.1 Absolute Maximum Ratings"
    )
    assert nodes[0].nodes[0].nodes[0].breadcrumb == (
        "5 Electrical Characteristics > 5.1 Absolute Maximum Ratings"
        " > 5.1.1 Junction Temperature"
    )
    assert nodes[1].breadcrumb == "6 Pin Configuration"


def test_find_breadcrumb_for_page_returns_deepest_section():
    raw = [
        [1, "5 Electrical Characteristics", 10],
        [2, "5.1 Absolute Maximum Ratings", 10],
        [3, "5.1.1 Junction Temperature", 11],
        [1, "6 Pin Configuration", 15],
    ]
    nodes = build_tree(raw, total_pages=20)

    # Page 11 is covered by the parent, child, and grandchild; deepest wins.
    assert find_breadcrumb_for_page(nodes, 11) == (
        "5 Electrical Characteristics > 5.1 Absolute Maximum Ratings"
        " > 5.1.1 Junction Temperature"
    )
    # Page 10 is covered by the parent and 5.1 (not the grandchild on page 11).
    assert find_breadcrumb_for_page(nodes, 10) == (
        "5 Electrical Characteristics > 5.1 Absolute Maximum Ratings"
    )
    assert find_breadcrumb_for_page(nodes, 15) == "6 Pin Configuration"


def test_find_breadcrumb_for_page_overlapping_siblings_first_wins():
    # Two same-level siblings share a start page; compute_end_pages gives the
    # first [1, 1] and the second [1, 20], so both cover page 1 at the same
    # depth. The section that actually starts there (the first) must win.
    raw = [
        [1, "A Intro", 1],
        [1, "B Body", 1],
    ]
    nodes = build_tree(raw, total_pages=20)

    assert find_breadcrumb_for_page(nodes, 1) == "A Intro"


def test_find_breadcrumb_for_page_returns_none_when_uncovered():
    nodes = build_tree([[1, "Overview", 5]], total_pages=10)

    # Page 1 precedes the only section (which starts on page 5).
    assert find_breadcrumb_for_page(nodes, 1) is None
    assert find_breadcrumb_for_page([], 3) is None


def test_build_tree_populates_boilerplate_category():
    """Catches regressions where `flag_boilerplate` gets removed from
    `build_tree`. Without it, the field would be empty for a clear match."""
    raw = [
        [1, "Electrical Characteristics", 1],
        [1, "Revision History", 10],
    ]
    nodes = build_tree(raw, total_pages=15)
    assert nodes[0].boilerplate_category == ""
    assert nodes[1].boilerplate_category == "revision"


def test_assign_breadcrumbs_strips_title_whitespace():
    nodes = [
        TocNode(
            title="  Outer  ",
            level=1,
            start_page=1,
            nodes=[TocNode(title="\tInner\n", level=2, start_page=1)],
        )
    ]
    assign_breadcrumbs(nodes)
    assert nodes[0].breadcrumb == "Outer"
    assert nodes[0].nodes[0].breadcrumb == "Outer > Inner"


def test_node_ids_depth_first():
    raw = [
        [1, "A", 1],
        [2, "A.1", 1],
        [2, "A.2", 3],
        [1, "B", 5],
    ]
    nodes = build_tree(raw, total_pages=10)
    # Depth-first: A=0001, A.1=0002, A.2=0003, B=0004
    assert nodes[0].node_id == "0001"
    assert nodes[0].nodes[0].node_id == "0002"
    assert nodes[0].nodes[1].node_id == "0003"
    assert nodes[1].node_id == "0004"


def test_architecture_doc_example():
    """Verify end_page computation matches the architecture doc example.

    Architecture doc specifies:
    - Block diagram start=5, Pin Config start=6 -> Block diagram end=5
    - Last section gets total_pages as end
    """
    raw = [
        [1, "Overview", 1],
        [1, "Block Diagram", 5],
        [1, "Pin Configuration", 6],
        [1, "Electrical Characteristics", 10],
    ]
    nodes = build_tree(raw, total_pages=20)
    assert nodes[0].end_page == 4  # Overview: 1 to 4
    assert nodes[1].end_page == 5  # Block Diagram: 5 to 5
    assert nodes[2].end_page == 9  # Pin Configuration: 6 to 9
    assert nodes[3].end_page == 20  # Last section: 10 to 20


def test_malformed_child_start_after_parent_end_is_clamped():
    """Last child should never get end_page lower than its start_page."""
    raw = [
        [1, "16 Communication", 50],
        [2, "16.1 Functional description", 51],
        [3, "16.1.1 Register write modes", 53],
        [3, "16.1.2 Communication frames", 53],
        [3, "16.1.3 Register read modes", 55],
        [2, "16.2 Electrical characteristics communication", 55],
    ]
    nodes = build_tree(raw, total_pages=73)

    section_161 = nodes[0].nodes[0]
    last_child = section_161.nodes[2]
    assert last_child.start_page == 55
    assert last_child.end_page == 55


# --- Integration tests with real PDF ---


@pytest.mark.real_pdf
def test_real_pdf_extract_toc():
    if not TLE9350_PATH.exists():
        pytest.skip("Test PDF not found")
    doc = pymupdf.open(str(TLE9350_PATH))
    raw_toc = extract_toc(doc)
    doc.close()
    assert len(raw_toc) > 0
    # Each entry should be [level, title, page]
    for entry in raw_toc:
        assert len(entry) >= 3
        assert isinstance(entry[0], int)
        assert isinstance(entry[1], str)
        assert isinstance(entry[2], int)


@pytest.mark.real_pdf
def test_real_pdf_build_tree():
    if not TLE9350_PATH.exists():
        pytest.skip("Test PDF not found")
    doc = pymupdf.open(str(TLE9350_PATH))
    raw_toc = extract_toc(doc)
    total_pages = len(doc)
    doc.close()

    nodes = build_tree(raw_toc, total_pages)
    assert len(nodes) > 0

    # All node_ids should be unique
    all_ids: list[str] = []
    _collect_ids(nodes, all_ids)
    assert len(all_ids) == len(set(all_ids))

    # All end_pages should be >= start_pages
    _assert_valid_ranges(nodes)


@pytest.mark.real_pdf
def test_real_pdf_table_enrichment():
    if not TLE9350_PATH.exists():
        pytest.skip("Test PDF not found")
    doc = pymupdf.open(str(TLE9350_PATH))
    raw_toc = extract_toc(doc)
    total_pages = len(doc)
    nodes = build_tree(raw_toc, total_pages)
    enrich_with_table_counts(nodes, doc)
    doc.close()

    # At least some sections should have tables (it's a datasheet)
    all_nodes: list = []
    _collect_all(nodes, all_nodes)
    has_any_tables = any(n.has_tables for n in all_nodes)
    assert has_any_tables


# --- Helpers ---


def _collect_ids(nodes, ids):
    for node in nodes:
        ids.append(node.node_id)
        _collect_ids(node.nodes, ids)


def _assert_valid_ranges(nodes):
    for node in nodes:
        assert node.end_page >= node.start_page, (
            f"{node.title}: end_page {node.end_page} < start_page {node.start_page}"
        )
        _assert_valid_ranges(node.nodes)


def _collect_all(nodes, result):
    for node in nodes:
        result.append(node)
        _collect_all(node.nodes, result)


# --- Worker pool sizing and start method ---


def _write_cgroup_v2(root: Path, contents: str) -> Path:
    (root / "cpu.max").write_text(contents, encoding="utf-8")
    return root


def _write_cgroup_v1(root: Path, quota: str, period: str) -> Path:
    cpu_dir = root / "cpu"
    cpu_dir.mkdir(parents=True, exist_ok=True)
    (cpu_dir / "cpu.cfs_quota_us").write_text(quota, encoding="utf-8")
    (cpu_dir / "cpu.cfs_period_us").write_text(period, encoding="utf-8")
    return root


@pytest.mark.parametrize(
    ("cpu_max", "expected"),
    [
        ("100000 100000", 1),  # 1.0 CPU
        ("50000 100000", 1),  # 0.5 CPU rounds up to one worker
        ("250000 100000", 2),  # 2.5 CPU truncates to two workers
        ("800000 100000", 8),
    ],
)
def test_cgroup_v2_quota_is_read(tmp_path, cpu_max, expected):
    root = _write_cgroup_v2(tmp_path, cpu_max)
    assert _read_cgroup_cpu_quota(root) == expected


def test_cgroup_v2_unlimited_reports_no_quota(tmp_path):
    root = _write_cgroup_v2(tmp_path, "max 100000")
    assert _read_cgroup_cpu_quota(root) is None


def test_cgroup_v1_quota_is_read(tmp_path):
    root = _write_cgroup_v1(tmp_path, "200000", "100000")
    assert _read_cgroup_cpu_quota(root) == 2


def test_cgroup_v1_unlimited_reports_no_quota(tmp_path):
    root = _write_cgroup_v1(tmp_path, "-1", "100000")
    assert _read_cgroup_cpu_quota(root) is None


def test_cgroup_absent_reports_no_quota(tmp_path):
    assert _read_cgroup_cpu_quota(tmp_path) is None


def test_cgroup_garbage_reports_no_quota(tmp_path):
    root = _write_cgroup_v2(tmp_path, "not-a-quota")
    assert _read_cgroup_cpu_quota(root) is None


@pytest.mark.parametrize("cpu_max", ["100000 0", "100000 -1"])
def test_cgroup_v2_zero_period_does_not_raise(tmp_path, cpu_max):
    """A malformed period must degrade to None, not ZeroDivisionError."""
    root = _write_cgroup_v2(tmp_path, cpu_max)
    assert _read_cgroup_cpu_quota(root) is None


def test_cgroup_v1_zero_period_does_not_raise(tmp_path):
    root = _write_cgroup_v1(tmp_path, "100000", "0")
    assert _read_cgroup_cpu_quota(root) is None


def _write_proc_cgroup(tmp_path: Path, contents: str) -> Path:
    proc = tmp_path / "proc_self_cgroup"
    proc.write_text(contents, encoding="utf-8")
    return proc


def test_cgroup_v2_quota_on_nested_cgroup_is_found(tmp_path):
    """systemd applies CPUQuota to a nested slice; the root reports 'max'."""
    root = tmp_path / "sys"
    service = root / "system.slice" / "app.service"
    service.mkdir(parents=True)
    _write_cgroup_v2(root, "max 100000")
    _write_cgroup_v2(service, "200000 100000")
    proc = _write_proc_cgroup(tmp_path, "0::/system.slice/app.service\n")

    assert _read_cgroup_cpu_quota(root, proc) == 2


def test_cgroup_v2_tightest_ancestor_quota_wins(tmp_path):
    """A quota on a parent slice constrains us even if our own is looser."""
    root = tmp_path / "sys"
    slice_dir = root / "system.slice"
    service = slice_dir / "app.service"
    service.mkdir(parents=True)
    _write_cgroup_v2(root, "max 100000")
    _write_cgroup_v2(slice_dir, "100000 100000")  # parent: 1 CPU
    _write_cgroup_v2(service, "800000 100000")  # own: 8 CPUs
    proc = _write_proc_cgroup(tmp_path, "0::/system.slice/app.service\n")

    assert _read_cgroup_cpu_quota(root, proc) == 1


def test_cgroup_v2_container_namespace_reads_root(tmp_path):
    """In its own cgroup namespace a container reports '0::/'."""
    root = tmp_path / "sys"
    root.mkdir()
    _write_cgroup_v2(root, "100000 100000")
    proc = _write_proc_cgroup(tmp_path, "0::/\n")

    assert _read_cgroup_cpu_quota(root, proc) == 1


def test_cgroup_v1_quota_on_nested_cgroup_is_found(tmp_path):
    root = tmp_path / "sys"
    service = root / "cpu" / "system.slice" / "app.service"
    service.mkdir(parents=True)
    _write_cgroup_v1(root, "-1", "100000")  # root: unlimited
    (service / "cpu.cfs_quota_us").write_text("400000", encoding="utf-8")
    (service / "cpu.cfs_period_us").write_text("100000", encoding="utf-8")
    proc = _write_proc_cgroup(tmp_path, "4:cpu,cpuacct:/system.slice/app.service\n")

    assert _read_cgroup_cpu_quota(root, proc) == 4


def test_cgroup_unreadable_proc_falls_back_to_root(tmp_path):
    root = _write_cgroup_v2(tmp_path, "300000 100000")
    assert _read_cgroup_cpu_quota(root, tmp_path / "missing") == 3


def test_available_cpus_clamps_to_cgroup_quota(monkeypatch):
    """A 1-CPU container on a 128-core host must report 1, not 128."""
    monkeypatch.setattr(structure.os, "process_cpu_count", lambda: 128)
    monkeypatch.setattr(structure, "_read_cgroup_cpu_quota", lambda: 1)
    assert _available_cpus() == 1


def test_available_cpus_falls_back_to_process_cpu_count(monkeypatch):
    monkeypatch.setattr(structure.os, "process_cpu_count", lambda: 4)
    monkeypatch.setattr(structure, "_read_cgroup_cpu_quota", lambda: None)
    assert _available_cpus() == 4


def test_available_cpus_never_returns_zero(monkeypatch):
    monkeypatch.setattr(structure.os, "process_cpu_count", lambda: None)
    monkeypatch.setattr(structure, "_read_cgroup_cpu_quota", lambda: None)
    assert _available_cpus() == 1


def test_mp_context_never_forks():
    """fork() in a threaded host copies held mutexes and every open fd."""
    assert _mp_context().get_start_method() != "fork"


class _FakePool:
    """Stand-in for ProcessPoolExecutor that records how it was constructed."""

    kwargs: dict = {}

    def __init__(self, **kwargs):
        _FakePool.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def map(self, fn, args):
        return [(page_idx, 0) for _, page_idx in args]


@pytest.fixture
def fake_pool(monkeypatch):
    _FakePool.kwargs = {}
    monkeypatch.setattr(
        concurrent.futures, "ProcessPoolExecutor", _FakePool, raising=True
    )
    return _FakePool


@pytest.mark.parametrize(
    ("cpus", "total_pages", "expected"),
    [
        (128, 42, 8),  # hard cap bounds a big host
        (1, 42, 1),  # a 1-CPU cgroup quota bounds the fan-out
        (128, 3, 3),  # never more workers than pages
        (4, 42, 4),
    ],
)
def test_parallel_pool_is_bounded(monkeypatch, fake_pool, cpus, total_pages, expected):
    monkeypatch.setattr(structure, "_available_cpus", lambda: cpus)
    _build_table_count_cache_parallel("doc.pdf", total_pages)
    assert fake_pool.kwargs["max_workers"] == expected


def test_parallel_pool_does_not_fork(monkeypatch, fake_pool):
    monkeypatch.setattr(structure, "_available_cpus", lambda: 4)
    _build_table_count_cache_parallel("doc.pdf", 20)
    assert fake_pool.kwargs["mp_context"].get_start_method() != "fork"


def test_parallel_cache_covers_every_page(monkeypatch, fake_pool):
    monkeypatch.setattr(structure, "_available_cpus", lambda: 4)
    cache = _build_table_count_cache_parallel("doc.pdf", 20)
    assert cache == dict.fromkeys(range(20), 0)
