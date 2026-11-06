"""Build Qt parameter controls from parameters.json."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QSlider, QDoubleSpinBox, QSpinBox, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox
)


CONFIG_PATH = Path(__file__).resolve().with_name("parameters.json")


class CustomSlider(QSlider):
    """Integer-backed slider that can represent decimal parameter values."""

    def __init__(self, range_min, range_max, default, step=1, decimal=0, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.scale = 10 ** int(decimal)
        self.setRange(round(float(range_min) * self.scale),
                      round(float(range_max) * self.scale))
        self.setSingleStep(max(1, round(float(step) * self.scale)))
        self.setPageStep(max(1, self.singleStep() * 5))
        self.setValue(round(float(default) * self.scale))
        self.setTracking(True)

    def get_value(self):
        return self.value() / self.scale


class CustomSpinBox(QSpinBox):
    def __init__(self, range_min, range_max, default, step=1, parent=None):
        super().__init__(parent)
        self.setRange(int(range_min), int(range_max))
        self.setSingleStep(max(1, int(step)))
        self.setValue(int(default))


class CustomDoubleSpinBox(QDoubleSpinBox):
    def __init__(self, range_min, range_max, default, step=0.01, decimal=2, parent=None):
        super().__init__(parent)
        self.setDecimals(int(decimal))
        self.setRange(float(range_min), float(range_max))
        self.setSingleStep(float(step))
        self.setKeyboardTracking(False)
        self.setValue(float(default))


class CustomComboBox(QComboBox):
    def __init__(self, values, default, parent=None):
        super().__init__(parent)
        if not values:
            raise ValueError("comboBox controls require a non-empty 'values' list.")
        self.addItems([str(value) for value in values])
        index = self.findText(str(default))
        if index < 0:
            raise ValueError(f"Default value {default!r} is not in {values!r}.")
        self.setCurrentIndex(index)


class ParserUI(QWidget):
    value_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.root_layout = QVBoxLayout(self)
        self.root_layout.setContentsMargins(8, 8, 8, 8)
        self.root_layout.setSpacing(10)
        self.controls = {}
        self.filter_name = None

    def _clear(self):
        while self.root_layout.count():
            item = self.root_layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                self._clear_layout(child_layout)
        self.controls.clear()

    @classmethod
    def _clear_layout(cls, layout):
        while layout.count():
            item = layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
            elif item.layout() is not None:
                cls._clear_layout(item.layout())

    def initialize(self, filter_name):
        self._clear()
        try:
            with CONFIG_PATH.open("r", encoding="utf-8") as stream:
                data = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read parameter configuration: {exc}") from exc

        filters = data.get("filters", {})
        if filter_name not in filters or not isinstance(filters[filter_name], dict):
            raise KeyError(f"No parameter configuration found for filter {filter_name!r}.")

        common = filters.get("common_conversions", {})
        default_decimals = int(common.get("decimal", 2))
        self.filter_name = filter_name
        self.root_layout.addWidget(QLabel(filter_name.replace("/", " / ").title()))

        for raw_name, config in filters[filter_name].items():
            if not isinstance(config, dict) or "type" not in config:
                continue
            name = raw_name.strip().lower().replace(" ", "_")
            self._add_parameter(name, raw_name, config, default_decimals)

        self.root_layout.addStretch()

    def _add_parameter(self, name, display_name, config, default_decimals):
        required = ("default", "range_min", "range_max")
        types = config.get("type", [])
        if not isinstance(types, list) or not types:
            raise ValueError(f"Parameter {display_name!r} needs a non-empty type list.")
        if "default" not in config:
            raise ValueError(f"Parameter {display_name!r} has no default value.")

        default = config["default"]
        minimum = config.get("range_min")
        maximum = config.get("range_max")
        step = config.get("step", 1)
        values = config.get("values")

        if "comboBox" in types:
            if not isinstance(values, list):
                raise ValueError(f"Parameter {display_name!r} needs a values list.")
        elif any(key not in config for key in required):
            raise ValueError(f"Parameter {display_name!r} needs default/range_min/range_max.")
        elif minimum > maximum or not minimum <= default <= maximum:
            raise ValueError(f"Invalid range/default for parameter {display_name!r}.")

        is_float = (
            isinstance(default, float)
            or isinstance(step, float)
            or "doubleSpinBox" in types
        )
        decimals = int(config.get("decimals", default_decimals)) if is_float else 0

        section = QWidget(self)
        section_layout = QVBoxLayout(section)
        section_layout.setContentsMargins(0, 0, 0, 0)
        section_layout.setSpacing(3)
        label = QLabel(display_name.replace("_", " ").capitalize())
        section_layout.addWidget(label)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        widgets = {}
        for widget_type in types:
            if widget_type == "slider":
                widget = CustomSlider(minimum, maximum, default, step, decimals)
            elif widget_type == "spinBox":
                widget = CustomSpinBox(minimum, maximum, default, step)
            elif widget_type == "doubleSpinBox":
                widget = CustomDoubleSpinBox(
                    minimum, maximum, default, step, decimals
                )
            elif widget_type == "comboBox":
                widget = CustomComboBox(values, default)
            else:
                raise ValueError(f"Unknown widget type {widget_type!r} for {display_name!r}.")
            widget.setToolTip(display_name.replace("_", " ").capitalize())
            row.addWidget(widget)
            widgets[widget_type] = widget

        self._synchronize(widgets)
        self.controls[name] = {"widgets": widgets}
        self._connect_parameter_signal(widgets)
        section_layout.addLayout(row)
        self.root_layout.addWidget(section)

    @staticmethod
    def _synchronize(widgets):
        slider = widgets.get("slider")
        spin = widgets.get("doubleSpinBox") or widgets.get("spinBox")
        if slider is None or spin is None:
            return
        scale = slider.scale
        slider.valueChanged.connect(lambda value: spin.setValue(value / scale))
        spin.valueChanged.connect(lambda value: slider.setValue(round(value * scale)))

    def _connect_parameter_signal(self, widgets):
        # Emit once per parameter; the slider and spin box are synchronized.
        for key in ("doubleSpinBox", "spinBox", "comboBox", "slider"):
            control = widgets.get(key)
            if control is None:
                continue
            signal = control.currentTextChanged if key == "comboBox" else control.valueChanged
            signal.connect(lambda *_: self.value_changed.emit())
            return

    def get_parameter_value(self, param_name):
        if param_name not in self.controls:
            raise KeyError(f"Unknown parameter: {param_name!r}.")
        widgets = self.controls[param_name]["widgets"]
        if "doubleSpinBox" in widgets:
            return widgets["doubleSpinBox"].value()
        if "spinBox" in widgets:
            return widgets["spinBox"].value()
        if "comboBox" in widgets:
            return widgets["comboBox"].currentText()
        if "slider" in widgets:
            return widgets["slider"].get_value()
        raise ValueError(f"No readable widget configured for {param_name!r}.")

    def get_values(self):
        return {name: self.get_parameter_value(name) for name in self.controls}
