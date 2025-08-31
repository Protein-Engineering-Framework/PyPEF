import sys
import tempfile
import pytest
from unittest.mock import patch

from pypef import __version__


@pytest.fixture(scope="session")
def app():
    """Create and return a QApplication for all tests in the session."""
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app  # No explicit teardown needed for QApplication


@pytest.fixture
def main_window(app):
    from pypef.gui.qt_window import MainWidget
    window = MainWidget()
    window.show()
    yield window

    window.close()
    app.processEvents()


#@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Skipping GUI test in CI")
@pytest.mark.pip_specific
def test_button_click_changes_label(main_window, app):
    window = main_window

    assert window.version_text.text() == f"PyPEF v. {__version__}"

    tmp_dir = tempfile.mkdtemp()
    with patch("PySide6.QtWidgets.QFileDialog.getExistingDirectory", return_value=str(tmp_dir)):
        window.button_work_dir.click()
        app.processEvents()

    assert hasattr(window, "working_directory")
    assert window.working_directory == str(tmp_dir)
