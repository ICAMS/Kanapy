"""Headless unit tests for Kanapy GUI helpers."""

from types import SimpleNamespace

import pytest

from kanapy.core import gui


class RecordingWidget:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.grid_calls = []
        self.pack_calls = []
        self.configure_calls = []
        self.current_calls = []

    def grid(self, **kwargs):
        self.grid_calls.append(kwargs)
        return self

    def pack(self, **kwargs):
        self.pack_calls.append(kwargs)
        return self

    def configure(self, **kwargs):
        self.configure_calls.append(kwargs)

    def current(self, index):
        self.current_calls.append(index)


class RecordingPopup:
    def __init__(self):
        self.title_calls = []
        self.geometry_calls = []
        self.after_calls = []
        self.destroy = object()
        self.updated = False

    def title(self, value):
        self.title_calls.append(value)

    def geometry(self, value):
        self.geometry_calls.append(value)

    def winfo_screenwidth(self):
        return 1000

    def winfo_screenheight(self):
        return 800

    def after(self, duration, callback):
        self.after_calls.append((duration, callback))

    def update_idletasks(self):
        self.updated = True



def test_parse_entry_returns_integer_list():
    assert gui.parse_entry(" 1, 20, -3 ") == [1, 20, -3]


def test_parse_entry_rejects_non_integer_values():
    with pytest.raises(ValueError):
        gui.parse_entry("1, two, 3")


@pytest.mark.parametrize(
    "entry_type, expected_widget",
    [("entry", "Entry"), ("checkbox", "Checkbutton"), ("combobox", "Combobox")],
)
def test_add_label_and_entry_supports_widget_types(monkeypatch, entry_type, expected_widget):
    created = []

    def factory(name):
        def create(*args, **kwargs):
            widget = RecordingWidget(*args, **kwargs)
            created.append((name, widget))
            return widget
        return create

    monkeypatch.setattr(gui.ttk, "Label", factory("Label"))
    monkeypatch.setattr(gui.ttk, "Entry", factory("Entry"))
    monkeypatch.setattr(gui.ttk, "Checkbutton", factory("Checkbutton"))
    monkeypatch.setattr(gui.ttk, "Combobox", factory("Combobox"))

    gui.add_label_and_entry(
        SimpleNamespace(),
        row=2,
        label_text="Value",
        entry_var=object(),
        entry_type=entry_type,
        bold=True,
        options=["one", "two"],
        col=3,
    )

    names = [name for name, _ in created]
    assert names[0] == "Label"
    assert expected_widget in names
    selected = next(widget for name, widget in created if name == expected_widget)
    assert selected.grid_calls == [{"row": 2, "column": 4, "sticky": "e"}]
    if entry_type == "combobox":
        assert selected.current_calls == [0]
        assert selected.configure_calls == [{"font": ("Helvetica", 12)}]


def test_add_label_and_entry_without_label_uses_requested_column(monkeypatch):
    entry = RecordingWidget()
    monkeypatch.setattr(gui.ttk, "Entry", lambda *args, **kwargs: entry)

    gui.add_label_and_entry(
        SimpleNamespace(), row=1, label_text=None, entry_var=object(), col=5
    )

    assert entry.grid_calls == [{"row": 1, "column": 5, "sticky": "e"}]


def test_self_closing_message_configures_popup(monkeypatch):
    popup = RecordingPopup()
    label = RecordingWidget()
    monkeypatch.setattr(gui, "Toplevel", lambda: popup)
    monkeypatch.setattr(gui.ttk, "Label", lambda *args, **kwargs: label)

    gui.self_closing_message("Saved", duration=1234)

    assert popup.title_calls == ["Information"]
    assert popup.geometry_calls[0] == "300x100"
    assert popup.geometry_calls[1] == "300x100+350+350"
    assert popup.after_calls == [(1234, popup.destroy)]
    assert popup.updated is True
    assert label.grid_calls == []
    assert label.pack_calls == [{"expand": True}]
