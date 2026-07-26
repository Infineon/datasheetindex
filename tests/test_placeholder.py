"""Placeholder test to verify the test setup works."""


def test_import():
    """Verify the package can be imported."""
    import datasheetindex

    assert hasattr(datasheetindex, "DatasheetIndex")
    assert hasattr(datasheetindex, "create_datasheet_tools_server")


def test_tools_package_exports():
    """Tool-oriented entry points should be importable from datasheetindex.tools."""
    from datasheetindex import tools

    assert hasattr(tools, "DatasheetTools")
    assert hasattr(tools, "create_datasheet_tools_server")
    assert hasattr(tools, "inspect_page")


def test_tools_package_dir_lists_the_lazy_reexports():
    """Lazy attributes are absent from globals() until touched.

    Without ``__dir__``, ``dir(datasheetindex.tools)`` reports a package with no
    public surface -- the one behaviour the PEP 562 conversion did not preserve.
    """
    from datasheetindex import tools

    assert dir(tools) == sorted(tools.__all__)
    assert "DatasheetTools" in dir(tools)
