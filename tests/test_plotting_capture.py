"""
Exercise figure capture used by staged plotting workflows.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from kanapy.core.plotting import capture_figures


def test_batch_capture_saves_each_show_and_closes_figures(tmp_path):
    """
    Preserve successive plots and the figure left open at stage exit.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary output directory.
    """
    paths = []
    original_show = plt.show
    with capture_figures(tmp_path, "maps", paths):
        plt.plot([0, 1], [0, 1])
        plt.show(block=True)
        plt.plot([0, 1], [1, 0])
        plt.show()
        plt.plot([0, 1], [2, 3])
    assert [p.name for p in paths] == ["maps_01.png", "maps_02.png", "maps_03.png"]
    assert all(p.stat().st_size > 1000 for p in paths)
    assert plt.show is original_show
    assert plt.get_fignums() == []


def test_preview_receives_already_saved_figures(tmp_path, monkeypatch):
    """
    Make figure files available before a preview can block execution.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary output directory.
    monkeypatch : pytest.MonkeyPatch
        Replacement for the native display operation.
    """
    viewed = []

    def preview(*args, **kwargs):
        """
        Inspect files when the real display would receive control.

        Parameters
        ----------
        *args, **kwargs : object
            Arguments accepted by pyplot display.
        """
        viewed.append((tmp_path / "preview_01.png").stat().st_size)

    monkeypatch.setattr(plt, "show", preview)
    with capture_figures(tmp_path, "preview", [], show=True):
        plt.plot([0, 1])
        plt.show()
    assert viewed[0] > 1000
    assert plt.show is preview
    assert plt.get_fignums() == []


def test_plot_failure_restores_show_and_preserves_existing_figure(tmp_path):
    """
    Restore pyplot state and close only figures made by a failed stage.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary output directory.
    """
    existing = plt.figure()
    original_show = plt.show
    try:
        with pytest.raises(RuntimeError, match="failed stage"):
            with capture_figures(tmp_path, "broken", []):
                plt.figure()
                raise RuntimeError("failed stage")
        assert plt.show is original_show
        assert plt.get_fignums() == [existing.number]
    finally:
        plt.close(existing)
