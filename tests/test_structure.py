"""Tests for ToC tree building and enrichment."""

import concurrent.futures
import json
import subprocess
from pathlib import Path
from typing import Any

import pymupdf
import pytest

from datasheetindex.core import structure
from datasheetindex.core.structure import (
    _available_cpus,
    _build_table_count_cache_parallel,
    _build_table_count_cache_pool,
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


def test_cgroup_v1_decoy_controllers_do_not_match_cpu(tmp_path):
    """'cpuset' and 'cpuacct' contain 'cpu' as a substring; only the exact
    controller token may select the cgroup path."""
    root = tmp_path / "sys"
    correct = root / "cpu" / "right"
    correct.mkdir(parents=True)
    (correct / "cpu.cfs_quota_us").write_text("200000", encoding="utf-8")
    (correct / "cpu.cfs_period_us").write_text("100000", encoding="utf-8")
    # A quota on the path a substring match would wrongly select.
    wrong = root / "cpu" / "wrong"
    wrong.mkdir(parents=True)
    (wrong / "cpu.cfs_quota_us").write_text("700000", encoding="utf-8")
    (wrong / "cpu.cfs_period_us").write_text("100000", encoding="utf-8")

    proc = _write_proc_cgroup(
        tmp_path,
        "12:cpuset:/wrong\n11:cpuacct:/wrong\n5:cpu,cpuacct:/right\n",
    )
    assert _read_cgroup_cpu_quota(root, proc) == 2


def test_cgroup_v1_co_mounted_cpu_cpuacct_directory(tmp_path):
    """Some runtimes bind-mount the real 'cpu,cpuacct' name with no 'cpu'
    compat symlink; the quota must still be found."""
    root = tmp_path / "sys"
    base = root / "cpu,cpuacct"
    base.mkdir(parents=True)
    (base / "cpu.cfs_quota_us").write_text("100000", encoding="utf-8")
    (base / "cpu.cfs_period_us").write_text("100000", encoding="utf-8")
    proc = _write_proc_cgroup(tmp_path, "5:cpu,cpuacct:/docker/abc\n")

    assert _read_cgroup_cpu_quota(root, proc) == 1


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
    """Stand-in for ProcessPoolExecutor that records how it was driven."""

    kwargs: dict = {}
    map_timeout: float | None = None
    shutdowns: list[dict] = []
    map_raises: BaseException | None = None
    events: list[str] = []

    def __init__(self, **kwargs):
        _FakePool.kwargs = kwargs

    def map(self, fn, args, timeout=None):
        """Returns a LAZY iterator, like the real executor.

        Returning a materialised list would make the fake unable to reproduce
        the bug it exists to guard: `pool.map` is lazy, and consuming it after
        shutdown meant the pool was torn down before a single result was read.
        """
        _FakePool.map_timeout = timeout
        if _FakePool.map_raises is not None:
            raise _FakePool.map_raises

        def _lazy():
            _FakePool.events.append("consumed")
            for _, page_idx in args:
                yield (page_idx, 0)

        return _lazy()

    def shutdown(self, wait=True, cancel_futures=False):
        _FakePool.events.append("shutdown")
        _FakePool.shutdowns.append({"wait": wait, "cancel_futures": cancel_futures})


@pytest.fixture
def fake_pool(monkeypatch):
    _FakePool.kwargs = {}
    _FakePool.map_timeout = None
    _FakePool.shutdowns = []
    _FakePool.map_raises = None
    _FakePool.events = []
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
    _build_table_count_cache_pool("doc.pdf", total_pages)
    assert fake_pool.kwargs["max_workers"] == expected


def test_parallel_pool_does_not_fork(monkeypatch, fake_pool):
    monkeypatch.setattr(structure, "_available_cpus", lambda: 4)
    _build_table_count_cache_pool("doc.pdf", 20)
    assert fake_pool.kwargs["mp_context"].get_start_method() != "fork"


def test_parallel_pool_redirects_worker_stdio(monkeypatch, fake_pool):
    """Workers still inherit fds 0-2 across exec; without _subprocess_init an
    MCP stdio parent's JSON-RPC channel collides with worker output."""
    monkeypatch.setattr(structure, "_available_cpus", lambda: 4)
    _build_table_count_cache_pool("doc.pdf", 20)
    assert fake_pool.kwargs["initializer"] is structure._subprocess_init


def test_parallel_cache_covers_every_page(monkeypatch, fake_pool):
    """Every page index is present. Values come from the fake, so they prove
    nothing -- real per-page counts are covered by the unmocked pool test."""
    monkeypatch.setattr(structure, "_available_cpus", lambda: 4)
    cache = _build_table_count_cache_pool("doc.pdf", 20)
    assert sorted(cache) == list(range(20))


def test_mp_context_falls_back_to_spawn_without_forkserver(monkeypatch):
    """Windows has no forkserver; the fallback must be spawn, never fork."""
    monkeypatch.setattr(
        structure.multiprocessing, "get_all_start_methods", lambda: ["fork", "spawn"]
    )
    assert _mp_context().get_start_method() == "spawn"


# --- The real pool, with nothing mocked ---


def _tables_on(page_index: int) -> int:
    """Table count `_write_table_pdf` draws on a page. Alternates 1 and 2 so a
    worker that returns a constant without reading the page cannot pass."""
    return 1 + (page_index % 2)


def _write_table_pdf(path: Path, pages: int) -> None:
    doc = pymupdf.open()
    for p in range(pages):
        page = doc.new_page()
        for grid in range(_tables_on(p)):
            top = 60 + grid * 330  # separated enough to read as distinct tables
            for row in range(4):
                for col in range(3):
                    rect = pymupdf.Rect(
                        50 + col * 90,
                        top + row * 25,
                        50 + (col + 1) * 90,
                        top + (row + 1) * 25,
                    )
                    page.draw_rect(rect, color=(0, 0, 0), width=0.7)
                    page.insert_text(
                        (rect.x0 + 3, rect.y0 + 15), f"{p}{grid}{row}{col}", fontsize=7
                    )
    doc.save(str(path))
    doc.close()


def _expected_counts(pages: int) -> dict[int, int]:
    return {i: _tables_on(i) for i in range(pages)}


def test_real_pool_counts_tables(tmp_path):
    """Spawns actual workers under the real start method.

    The mocked pool tests assert construction kwargs but never start a process,
    so they cannot catch an unpicklable argument, an initializer that raises, or
    a forkserver that fails to bootstrap -- exactly the bug class this module
    exists to avoid. enrich_with_table_counts swallows such failures, so without
    this test a permanently broken pool would pass CI.

    Counts are asserted against the fixture rather than against an in-process
    sequential scan. Both paths are pinned to the classic detector now, so a
    direct comparison is also valid -- see
    test_parallel_and_sequential_agree_under_a_layout_hook, which makes it.
    """
    pages = 13
    pdf = tmp_path / "tables.pdf"
    _write_table_pdf(pdf, pages=pages)

    parallel = _build_table_count_cache_parallel(str(pdf), pages)

    assert parallel == _expected_counts(pages)


def test_helper_counts_tables(tmp_path):
    """The Windows path, exercised on every platform.

    Production only takes this branch on win32, so CI's Linux lane would never
    run it if the test gated on sys.platform -- and a helper broken by a typo,
    a bad module path or an unwritable temp file would ship green, falling back
    to a silent 3x-slower sequential scan (or, before the timeout existed, an
    unbounded hang). The helper is a plain subprocess, so it runs anywhere.
    """
    pages = 13
    pdf = tmp_path / "tables.pdf"
    _write_table_pdf(pdf, pages=pages)

    assert structure._build_table_count_cache_helper(str(pdf), pages) == (
        _expected_counts(pages)
    )


def test_helper_and_pool_agree(tmp_path):
    """The two parallel paths must be interchangeable, not merely both green.

    A platform split is only safe if the branches return the same answer; this
    is what lets the dispatcher pick either one without changing results.
    """
    pages = 13
    pdf = tmp_path / "tables.pdf"
    _write_table_pdf(pdf, pages=pages)

    assert structure._build_table_count_cache_helper(str(pdf), pages) == (
        structure._build_table_count_cache_pool(str(pdf), pages)
    )


def test_parallel_dispatches_by_platform(monkeypatch):
    """win32 must not build a pool in-process: that is the deadlock."""
    calls = []
    monkeypatch.setattr(
        structure,
        "_build_table_count_cache_helper",
        lambda path, pages: calls.append("helper") or {},
    )
    monkeypatch.setattr(
        structure,
        "_build_table_count_cache_pool",
        lambda path, pages: calls.append("pool") or {},
    )

    monkeypatch.setattr(structure, "_is_windows", lambda: True)
    _build_table_count_cache_parallel("doc.pdf", 20)
    monkeypatch.setattr(structure, "_is_windows", lambda: False)
    _build_table_count_cache_parallel("doc.pdf", 20)

    assert calls == ["helper", "pool"]


def test_pool_map_is_bounded_by_a_timeout(monkeypatch, fake_pool):
    """Without a timeout there is no way back from a wedged pool.

    enrich_with_table_counts recovers by catching an exception, so the scan has
    to raise rather than block -- the Windows hang was exactly a pool that
    never raised.
    """
    monkeypatch.setattr(structure, "_available_cpus", lambda: 4)
    _build_table_count_cache_pool("doc.pdf", 20)
    assert fake_pool.map_timeout == structure._scan_timeout(20)


def test_pool_failure_does_not_wait_on_shutdown(monkeypatch, fake_pool):
    """shutdown(wait=True) on a wedged pool re-creates the hang it escaped."""
    monkeypatch.setattr(structure, "_available_cpus", lambda: 4)
    _FakePool.map_raises = TimeoutError("wedged")

    with pytest.raises(TimeoutError):
        _build_table_count_cache_pool("doc.pdf", 20)

    assert fake_pool.shutdowns == [{"wait": False, "cancel_futures": True}]


def test_pool_consumes_the_map_before_shutting_down(monkeypatch, fake_pool):
    """The original bug, pinned by ordering rather than by outcome.

    pool.map is lazy: materialising it after the `with` block ran
    shutdown(wait=True) before a single result had been read, so a stalled
    worker blocked in shutdown instead of raising. Asserting only the returned
    counts cannot catch a regression to that -- the order is the invariant.
    """
    monkeypatch.setattr(structure, "_available_cpus", lambda: 4)
    _build_table_count_cache_pool("doc.pdf", 20)
    assert fake_pool.events == ["consumed", "shutdown"]


def _install_fake_popen(
    monkeypatch,
    *,
    returncode=0,
    pages_written=None,
    timeout_first_wait=False,
    wait_raises=None,
    stderr_text=None,
):
    """Drive the helper's failure branches without a real worker.

    These paths (kill-on-timeout, non-zero exit, short result) only run when
    something has already gone wrong, and they are where a hang fix is easiest
    to get wrong -- so they need coverage that does not depend on being able to
    wedge a real subprocess.
    """
    # Annotated: the values are heterogeneous, and an inferred union makes
    # every use of them a type error.
    state: dict[str, Any] = {"kills": 0, "waits": []}

    class _Proc:
        def __init__(self, cmd, **kwargs):
            state["cmd"] = cmd
            state["kwargs"] = kwargs
            self.returncode = returncode
            if stderr_text is not None:
                kwargs["stderr"].write(stderr_text.encode("utf-8"))
                kwargs["stderr"].flush()
            if pages_written is not None:
                with open(cmd[-1], "w", encoding="utf-8") as handle:
                    json.dump({str(i): 0 for i in range(pages_written)}, handle)

        def wait(self, timeout=None):
            state["waits"].append(timeout)
            if wait_raises is not None and len(state["waits"]) == 1:
                raise wait_raises
            if timeout_first_wait and len(state["waits"]) == 1:
                raise subprocess.TimeoutExpired(cmd="worker", timeout=timeout or 0)
            return self.returncode

        def kill(self):
            state["kills"] += 1

    monkeypatch.setattr(structure.subprocess, "Popen", _Proc)
    return state


def test_helper_never_pipes_worker_stderr(monkeypatch):
    """stderr must be a file, never a PIPE.

    The worker's own pool children inherit fd 2, so they hold the write end.
    communicate() on a PIPE would then wait for an EOF that kill() cannot
    deliver -- kill() reaches only the direct child -- which is precisely the
    hang this module exists to remove. Verified reproducible before this guard.
    """
    state = _install_fake_popen(monkeypatch, pages_written=20)
    structure._build_table_count_cache_helper("doc.pdf", 20)

    assert state["kwargs"]["stderr"] is not subprocess.PIPE
    assert hasattr(state["kwargs"]["stderr"], "write")
    assert state["kwargs"]["stdin"] is subprocess.DEVNULL
    assert state["kwargs"]["stdout"] is subprocess.DEVNULL
    assert state["kwargs"]["creationflags"] == getattr(
        subprocess, "CREATE_NO_WINDOW", 0
    )
    assert state["cmd"][:3] == [
        structure.sys.executable,
        "-m",
        "datasheetindex.core._scan_worker",
    ]


def test_helper_bounds_every_wait_on_the_timeout_path(monkeypatch):
    """A bare wait() after kill() would reintroduce the unbounded block."""
    state = _install_fake_popen(monkeypatch, timeout_first_wait=True)

    with pytest.raises(RuntimeError, match="timed out"):
        structure._build_table_count_cache_helper("doc.pdf", 20)

    assert state["kills"] == 1
    assert len(state["waits"]) == 2
    assert all(timeout is not None for timeout in state["waits"])


def test_helper_kills_the_child_on_a_non_timeout_interruption(monkeypatch):
    """A narrow `except TimeoutExpired` leaves the child alive on Ctrl-C.

    KeyboardInterrupt out of wait() is the realistic case. The child would keep
    its whole 8-worker pool running and hold a handle into the temp directory
    being torn down -- the orphan this module exists to prevent, merely
    relocated, and invisible to a test that only ever raises TimeoutExpired.
    """
    state = _install_fake_popen(monkeypatch, wait_raises=KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        structure._build_table_count_cache_helper("doc.pdf", 20)

    assert state["kills"] == 1


@pytest.mark.parametrize("pages", [20, 100_000])
def test_helper_deadline_exceeds_the_workers_own(monkeypatch, pages):
    """The child must get to hit its own deadline first.

    The parent's clock starts earlier (it spawns the child, which then imports
    pymupdf), so without a margin the parent always wins the race and kills a
    worker that was about to report a real traceback.
    """
    state = _install_fake_popen(monkeypatch, pages_written=pages)
    structure._build_table_count_cache_helper("doc.pdf", pages)

    # The 100_000 case is the point: it proves the ceiling is applied INSIDE
    # _scan_timeout, so it bounds both sides equally and the margin survives.
    # Hoisting the cap outside the addition collapses parent and child onto the
    # same deadline and revives the race, while the 20-page case stays green.
    assert state["waits"][0] > structure._scan_timeout(pages)


def test_timeout_carries_the_workers_traceback(monkeypatch):
    """The grace margin exists so the child fails first with a real traceback.
    Discarding it is what made every stall look identical, so the message
    carrying it is the claim worth pinning."""
    _install_fake_popen(
        monkeypatch, timeout_first_wait=True, stderr_text="MuPDF error: page 41"
    )

    with pytest.raises(RuntimeError, match="page 41"):
        structure._build_table_count_cache_helper("doc.pdf", 20)


def test_timeout_message_omits_an_empty_stderr(monkeypatch):
    """On a true parent-side timeout the child had not yet failed, so there is
    usually nothing to report -- say nothing rather than trail a colon."""
    _install_fake_popen(monkeypatch, timeout_first_wait=True)

    with pytest.raises(RuntimeError) as excinfo:
        structure._build_table_count_cache_helper("doc.pdf", 20)

    assert not str(excinfo.value).rstrip().endswith(":")


def test_worker_failure_carries_its_stderr(monkeypatch):
    _install_fake_popen(monkeypatch, returncode=3, stderr_text="boom on page 7")

    with pytest.raises(RuntimeError, match="boom on page 7"):
        structure._build_table_count_cache_helper("doc.pdf", 20)


def test_read_worker_stderr_survives_a_missing_file(tmp_path):
    assert structure._read_worker_stderr(str(tmp_path / "nope.err")) == (
        "(worker stderr unavailable)"
    )


def test_read_worker_stderr_keeps_only_the_tail(tmp_path):
    """Its pool workers all inherit fd 2 and MuPDF is voluble, so this file can
    reach megabytes; reading it whole risks a MemoryError on the failure path."""
    err = tmp_path / "big.err"
    err.write_bytes(b"x" * 100_000 + b"TAIL")

    result = structure._read_worker_stderr(str(err))

    assert result.endswith("TAIL")
    assert len(result) <= 2000


def test_helper_raises_when_the_worker_fails(monkeypatch):
    state = _install_fake_popen(monkeypatch, returncode=3)

    with pytest.raises(RuntimeError, match="exited 3"):
        structure._build_table_count_cache_helper("doc.pdf", 20)

    assert state["kills"] == 0


def test_helper_rejects_a_short_result(monkeypatch):
    """A missing page is indistinguishable from a page with no tables.

    _apply_table_counts defaults an absent page to 0, so a truncated result
    would ship as a confident wrong answer. Raising lets the sequential
    fallback produce a right one.
    """
    _install_fake_popen(monkeypatch, pages_written=17)

    with pytest.raises(RuntimeError, match="17 of 20"):
        structure._build_table_count_cache_helper("doc.pdf", 20)


def test_helper_refuses_a_frozen_interpreter(monkeypatch):
    """`-m` against a frozen host re-runs the app itself, which for a frozen
    MCP server means starting a second server."""
    monkeypatch.setattr(structure.sys, "frozen", True, raising=False)

    with pytest.raises(RuntimeError, match="frozen"):
        structure._build_table_count_cache_helper("doc.pdf", 20)


def test_abandon_pool_prefers_kill_workers_where_available():
    """kill_workers() (3.14+) is the only API that stops a wedged worker.

    shutdown(wait=False) returns promptly but leaves the manager thread joining
    those workers, and concurrent.futures joins it from an atexit hook -- so the
    process still hangs, just at exit. Our floor is 3.13, where the attribute is
    absent, so without this test the branch that matters ships unverified.
    """

    class _Killable:
        def __init__(self):
            self.killed = 0
            self.shutdowns = 0

        def kill_workers(self):
            self.killed += 1

        def shutdown(self, wait=True, cancel_futures=False):
            self.shutdowns += 1

    pool = _Killable()
    structure._abandon_pool(pool)

    assert (pool.killed, pool.shutdowns) == (1, 0)


def test_abandon_pool_falls_back_to_shutdown_without_kill_workers():
    class _Plain:
        def __init__(self):
            self.calls = []

        def shutdown(self, wait=True, cancel_futures=False):
            self.calls.append((wait, cancel_futures))

    pool = _Plain()
    structure._abandon_pool(pool)

    assert pool.calls == [(False, True)]


def test_scan_timeout_is_capped():
    """An uncapped deadline outlives the caller's own request timeout, so the
    sequential fallback it exists to reach never runs."""
    assert structure._scan_timeout(100_000) == structure._SCAN_TIMEOUT_CEILING_SECONDS


def test_scan_timeout_has_a_floor():
    """A small document must not get a timeout so tight that a slow host trips
    it; the deadline only exists to catch a stall."""
    assert structure._scan_timeout(1) == 120.0
    assert structure._scan_timeout(500) == 500.0


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off", " 0 "])
def test_parallel_can_be_disabled_by_env(tmp_path, monkeypatch, value):
    """The escape hatch for a host where process creation misbehaves.

    Recorded, never raised: enrich_with_table_counts catches Exception to fall
    back, so an AssertionError from a stub is swallowed and the test passes
    whatever the env check does. (Verified: neutering the check to `and True`
    left the raising version green.) The assertion has to be on the record.
    """
    monkeypatch.setenv("DATASHEETINDEX_PARALLEL", value)
    attempts: list[str] = []
    monkeypatch.setattr(
        structure,
        "_build_table_count_cache_parallel",
        lambda path, pages: attempts.append(path) or {},
    )

    pages = 13
    pdf = tmp_path / "tables.pdf"
    _write_table_pdf(pdf, pages=pages)
    doc = pymupdf.open(str(pdf))
    try:
        nodes = build_tree([[1, "A", 1]], total_pages=pages)
        enrich_with_table_counts(nodes, doc, pdf_path=str(pdf))
    finally:
        doc.close()

    assert attempts == []


def test_parallel_runs_when_the_env_var_does_not_disable_it(tmp_path, monkeypatch):
    """The other half of the guard: without it, the test above would pass on
    code that simply never parallelises anything."""
    monkeypatch.delenv("DATASHEETINDEX_PARALLEL", raising=False)
    attempts: list[str] = []
    monkeypatch.setattr(
        structure,
        "_build_table_count_cache_parallel",
        lambda path, pages: attempts.append(path) or {},
    )

    pages = 13
    pdf = tmp_path / "tables.pdf"
    _write_table_pdf(pdf, pages=pages)
    doc = pymupdf.open(str(pdf))
    try:
        nodes = build_tree([[1, "A", 1]], total_pages=pages)
        enrich_with_table_counts(nodes, doc, pdf_path=str(pdf))
    finally:
        doc.close()

    assert attempts == [str(pdf)]


def test_sequential_counts_tables(tmp_path):
    """The fallback path must count real tables, not just avoid raising.

    Asserted against the same fixture as the parallel test. Both paths report
    the classic detector's answer regardless of process state, so this pins the
    sequential scan to a known number rather than to whatever engine happens to
    be active.
    """
    pages = 7
    pdf = tmp_path / "tables.pdf"
    _write_table_pdf(pdf, pages=pages)

    doc = pymupdf.open(str(pdf))
    try:
        sequential = structure._build_table_count_cache_sequential(doc)
    finally:
        doc.close()

    assert sequential == _expected_counts(pages)


def _classic_tables_on(page_index: int) -> int:
    """Ruled grids `_write_mixed_table_pdf` draws on a page, which is also the
    count the classic detector reports. Alternates 1 and 2 so a worker that
    returns a constant without reading the page cannot pass."""
    return 1 + (page_index % 2)


def _write_mixed_table_pdf(path: Path, pages: int) -> None:
    """Each page gets N fully-ruled grids plus exactly one table drawn with
    horizontal rules only.

    The classic detector finds only the ruled grids; the ML layout engine finds
    those plus the unruled one. So the two engines return provably different
    numbers on this fixture, which is what makes it able to catch the bug. The
    existing `_write_table_pdf` fixture is a pure ruled grid -- the one style
    both engines agree on -- and can never catch it.
    """
    doc = pymupdf.open()
    for p in range(pages):
        page = doc.new_page()
        for grid in range(_classic_tables_on(p)):
            top = 60 + grid * 330
            for row in range(4):
                for col in range(3):
                    rect = pymupdf.Rect(
                        50 + col * 90,
                        top + row * 25,
                        50 + (col + 1) * 90,
                        top + (row + 1) * 25,
                    )
                    page.draw_rect(rect, color=(0, 0, 0), width=0.7)
                    page.insert_text(
                        (rect.x0 + 3, rect.y0 + 15), f"{p}{grid}{row}{col}", fontsize=7
                    )
        unruled_top = 560
        for row in range(4):
            y = unruled_top + row * 24
            page.draw_line((50, y), (320, y), width=0.7)
        for row in range(3):
            for col in range(3):
                page.insert_text(
                    (55 + col * 90, unruled_top + row * 24 + 16),
                    f"u{p}{row}{col}",
                    fontsize=7,
                )
    # One bookmark per page. Without a ToC, build_datasheet produces zero nodes
    # and no node ever carries a table_count, so tests/_fresh_layout_process.py
    # would have nothing to assert against.
    doc.set_toc([[1, f"Section {i + 1}", i + 1] for i in range(pages)])
    doc.save(str(path))
    doc.close()


def _expected_classic_counts(pages: int) -> dict[int, int]:
    return {i: _classic_tables_on(i) for i in range(pages)}


def _fake_layout_hook(page, **kwargs):
    """Stand in for the ONNX layout analyzer without installing it.

    Returns one more "table" box than the classic detector finds, mimicking the
    real engine on `_write_mixed_table_pdf` (measured: classic [1,2,1,2,1,2],
    ML [2,3,2,3,2,3]). find_tables() reports exactly one table per box.
    """
    boxes = _classic_tables_on(page.number) + 1
    return [(50.0, 60.0 + i * 40, 320.0, 90.0 + i * 40, "table") for i in range(boxes)]


def test_sequential_ignores_an_active_layout_hook(tmp_path, monkeypatch):
    """The sequential path must report classic counts even in a process that
    has imported pymupdf4llm. Without the guard it reports the hook's answer."""
    pages = 6
    pdf = tmp_path / "mixed.pdf"
    _write_mixed_table_pdf(pdf, pages=pages)
    monkeypatch.setattr(pymupdf, "_get_layout", _fake_layout_hook)

    doc = pymupdf.open(str(pdf))
    try:
        sequential = structure._build_table_count_cache_sequential(doc)
    finally:
        doc.close()

    assert sequential == _expected_classic_counts(pages)


def test_parallel_and_sequential_agree_under_a_layout_hook(tmp_path, monkeypatch):
    """The two paths must return the same numbers for the same document.

    The hook is installed in the parent *before* the pool starts, so this also
    pins worker hook non-inheritance: workers spawn under forkserver/spawn and
    never import pymupdf4llm, and a monkeypatched parent global cannot cross
    that boundary. Installing it afterwards would spawn the workers into a
    pristine parent and assert nothing about inheritance.

    Before the guard, this fixture returned {0: 2, 1: 3, ...} sequentially and
    {0: 1, 1: 2, ...} in parallel.
    """
    pages = 6
    pdf = tmp_path / "mixed.pdf"
    _write_mixed_table_pdf(pdf, pages=pages)

    monkeypatch.setattr(pymupdf, "_get_layout", _fake_layout_hook)

    parallel = structure._build_table_count_cache_parallel(str(pdf), pages)

    doc = pymupdf.open(str(pdf))
    try:
        sequential = structure._build_table_count_cache_sequential(doc)
    finally:
        doc.close()

    assert parallel == sequential == _expected_classic_counts(pages)


# --- enrich_with_table_counts dispatch and degradation ---


def _blank_doc(pages: int):
    doc = pymupdf.open()
    for _ in range(pages):
        doc.new_page()
    return doc


@pytest.mark.parametrize(
    ("pages", "expect_parallel"),
    [(11, False), (12, True)],
)
def test_parallel_dispatch_respects_page_threshold(monkeypatch, pages, expect_parallel):
    calls = []
    monkeypatch.setattr(
        structure,
        "_build_table_count_cache_parallel",
        lambda path, total: calls.append(total) or dict.fromkeys(range(total), 0),
    )
    doc = _blank_doc(pages)
    try:
        nodes = build_tree([[1, "S", 1]], total_pages=pages)
        enrich_with_table_counts(nodes, doc, pdf_path="doc.pdf")
    finally:
        doc.close()
    assert bool(calls) is expect_parallel


def test_parallel_failure_degrades_to_sequential_and_warns(monkeypatch, caplog):
    """A pool that cannot start must not fail the build -- but must be audible."""

    def _boom(path, total):
        raise RuntimeError("pool did not start")

    monkeypatch.setattr(structure, "_build_table_count_cache_parallel", _boom)
    doc = _blank_doc(15)
    try:
        nodes = build_tree([[1, "S", 1]], total_pages=15)
        with caplog.at_level("WARNING", logger=structure.logger.name):
            result = enrich_with_table_counts(nodes, doc, pdf_path="doc.pdf")
    finally:
        doc.close()

    assert result[0].table_count == 0  # sequential scan still ran
    assert "falling back to sequential" in caplog.text
    assert any(r.levelname == "WARNING" for r in caplog.records)
