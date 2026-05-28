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
