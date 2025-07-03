import pytest


def test_package_import():
    """
    Tests that the main 'neupi' package can be imported.
    """
    try:
        import neupi
    except ImportError as e:
        pytest.fail(f"Failed to import the 'neupi' package. Error: {e}")


def test_cython_module_import():
    """
    Tests that the compiled Cython module can be imported successfully.
    This is a crucial check for the build process.
    """
    try:
        from neupi.pm_ssl.io import uai_reader_cython
    except ImportError as e:
        pytest.fail(f"Failed to import the Cython module 'uai_reader_cython'. Error: {e}")
