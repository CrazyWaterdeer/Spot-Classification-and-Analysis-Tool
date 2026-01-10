"""
Main GUI application for SCAT.
Integrates labeling, training, analysis, and results viewing.
"""

import sys
import os
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np
import pandas as pd
import cv2

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTabWidget, QGroupBox,
    QSpinBox, QDoubleSpinBox, QFormLayout, QComboBox, QCheckBox,
    QProgressBar, QTextEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QSplitter, QLineEdit, QMessageBox, QScrollArea,
    QDialog, QKeySequenceEdit, QDialogButtonBox,
    QGraphicsView, QGraphicsScene, QGraphicsPathItem,
    QListWidget, QMenu
)
from PySide6.QtCore import Qt, QThread, Signal, QSize, QRectF
from PySide6.QtGui import (
    QFont, QPixmap, QImage, QIcon, QKeySequence, QPainter, 
    QPen, QColor, QBrush, QShortcut, QPalette, QWheelEvent,
    QFontDatabase
)

# Import SCAT modules
from .detector import DepositDetector
from .classifier import ClassifierConfig, get_classifier
from .analyzer import Analyzer, ReportGenerator
from .config import config, get_timestamped_output_dir, DEFAULT_CONFIG

# Note: trainer is imported lazily when needed to avoid loading sklearn at startup


# =============================================================================
# Custom Widgets - SpinBox/ComboBox without scroll wheel
# =============================================================================
class NoScrollSpinBox(QSpinBox):
    """SpinBox that ignores mouse wheel events to prevent accidental changes."""
    def wheelEvent(self, event: QWheelEvent):
        event.ignore()  # Pass to parent for scrolling


class NoScrollDoubleSpinBox(QDoubleSpinBox):
    """DoubleSpinBox that ignores mouse wheel events to prevent accidental changes."""
    def wheelEvent(self, event: QWheelEvent):
        event.ignore()  # Pass to parent for scrolling


class NoScrollComboBox(QComboBox):
    """ComboBox that ignores mouse wheel events to prevent accidental changes."""
    def wheelEvent(self, event: QWheelEvent):
        event.ignore()  # Pass to parent for scrolling


# =============================================================================
# Theme Colors - Dark Theme with Coral Accent (DIC2497 #DA4E42)
# =============================================================================
class Theme:
    """SCAT Application Color Theme - Dark with Coral Accent (minimal use)"""
    
    # Primary accent - DIC2497 (minimal use for accents only)
    PRIMARY = "#DA4E42"        # Coral red - selected tabs, primary buttons only
    PRIMARY_DARK = "#C44539"
    PRIMARY_LIGHT = "#E8695E"
    
    # Secondary - DIC540 (general UI elements)
    SECONDARY = "#636867"      # Gray-green
    SECONDARY_DARK = "#525756"
    SECONDARY_LIGHT = "#7A7F7E"
    
    # Semantic colors for deposits
    NORMAL = "#4CAF50"         # Green - normal deposits
    NORMAL_DARK = "#388E3C"
    
    ROD = "#DA4E42"            # Same as primary - ROD deposits  
    ROD_DARK = "#C44539"
    
    ARTIFACT = "#636867"       # Same as secondary - artifacts
    ARTIFACT_DARK = "#525756"
    
    # Background layers (very dark, almost black)
    BG_DARKEST = "#0A0A0A"     # Main window background
    BG_DARK = "#121212"        # Card/group background
    BG_MEDIUM = "#1A1A1A"      # Input field background
    BG_LIGHT = "#242424"       # Hover state
    BG_LIGHTER = "#2E2E2E"     # Active/pressed state
    
    # Text
    TEXT_PRIMARY = "#FFFFFF"
    TEXT_SECONDARY = "#9A9A9A"
    TEXT_MUTED = "#5A5A5A"
    
    # Borders
    BORDER = "#2A2A2A"
    BORDER_FOCUS = "#DA4E42"   # Focus uses primary
    
    @staticmethod
    def button_style(bg_color: str, text_color: str = "#FFFFFF", hover_color: str = None) -> str:
        if hover_color is None:
            hover_color = Theme.BG_LIGHTER
        return f"""
            QPushButton {{
                background-color: {bg_color};
                color: {text_color};
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
            QPushButton:pressed {{
                background-color: {Theme.BG_LIGHT};
            }}
            QPushButton:disabled {{
                background-color: #1E1E1E;
                color: #404040;
            }}
        """
    
    @staticmethod
    def get_app_stylesheet() -> str:
        """Return the complete application stylesheet."""
        return f"""
            /* Main Window */
            QMainWindow, QDialog {{
                background-color: {Theme.BG_DARKEST};
                color: {Theme.TEXT_PRIMARY};
            }}
            
            /* Widgets */
            QWidget {{
                background-color: {Theme.BG_DARKEST};
                color: {Theme.TEXT_PRIMARY};
            }}
            
            /* Tab Widget */
            QTabWidget::pane {{
                border: 1px solid {Theme.BORDER};
                border-radius: 5px;
                background-color: {Theme.BG_DARK};
                margin-top: -1px;
            }}
            QTabBar::tab {{
                background-color: {Theme.BG_MEDIUM};
                color: {Theme.TEXT_SECONDARY};
                padding: 12px 28px;
                margin-right: 3px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-weight: bold;
            }}
            QTabBar::tab:selected {{
                background-color: {Theme.PRIMARY};
                color: {Theme.TEXT_PRIMARY};
            }}
            QTabBar::tab:hover:!selected {{
                background-color: {Theme.BG_LIGHT};
                color: {Theme.TEXT_PRIMARY};
            }}
            
            /* Group Box */
            QGroupBox {{
                background-color: {Theme.BG_DARK};
                border: 1px solid {Theme.BORDER};
                border-radius: 8px;
                margin-top: 20px;
                padding: 20px 12px 12px 12px;
                font-weight: bold;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 14px;
                top: 6px;
                padding: 2px 10px;
                color: {Theme.PRIMARY};
                background-color: {Theme.BG_DARK};
                font-size: 13px;
            }}
            
            /* Buttons - Secondary (gray) by default with visible background */
            QPushButton {{
                background-color: {Theme.BG_LIGHT};
                color: {Theme.TEXT_PRIMARY};
                border: 1px solid {Theme.BORDER};
                padding: 8px 16px;
                border-radius: 5px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {Theme.SECONDARY};
                border-color: {Theme.SECONDARY};
            }}
            QPushButton:pressed {{
                background-color: {Theme.SECONDARY_DARK};
            }}
            QPushButton:disabled {{
                background-color: #1E1E1E;
                color: #404040;
            }}
            
            /* Input Fields */
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
                background-color: {Theme.BG_MEDIUM};
                border: 1px solid {Theme.BORDER};
                border-radius: 5px;
                padding: 8px 10px;
                color: {Theme.TEXT_PRIMARY};
                min-height: 20px;
                selection-background-color: {Theme.SECONDARY};
            }}
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
                border-color: {Theme.PRIMARY};
            }}
            QComboBox::drop-down {{
                border: none;
                padding-right: 12px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid {Theme.TEXT_SECONDARY};
            }}
            QComboBox QAbstractItemView {{
                background-color: {Theme.BG_MEDIUM};
                border: 1px solid {Theme.BORDER};
                selection-background-color: {Theme.SECONDARY};
                padding: 4px;
            }}
            
            /* SpinBox buttons */
            QSpinBox::up-button, QSpinBox::down-button,
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
                background-color: {Theme.BG_LIGHT};
                border: 1px solid {Theme.BORDER};
                width: 20px;
                border-radius: 2px;
                margin: 1px;
            }}
            QSpinBox::up-button:hover, QSpinBox::down-button:hover,
            QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
                background-color: {Theme.SECONDARY};
            }}
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
                width: 10px;
                height: 10px;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-bottom: 6px solid {Theme.TEXT_PRIMARY};
            }}
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
                width: 10px;
                height: 10px;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid {Theme.TEXT_PRIMARY};
            }}
            
            /* Labels - no background, bold for form labels */
            QLabel {{
                color: {Theme.TEXT_PRIMARY};
                background-color: transparent;
                padding: 0px;
                font-weight: bold;
            }}
            
            /* Tables */
            QTableWidget {{
                background-color: {Theme.BG_DARK};
                gridline-color: {Theme.BORDER};
                border: 1px solid {Theme.BORDER};
                border-radius: 5px;
            }}
            QTableWidget::item {{
                padding: 8px;
            }}
            QTableWidget::item:selected {{
                background-color: {Theme.SECONDARY};
            }}
            QHeaderView::section {{
                background-color: {Theme.BG_MEDIUM};
                color: {Theme.TEXT_PRIMARY};
                padding: 10px 8px;
                border: none;
                border-bottom: 1px solid {Theme.BORDER};
                font-weight: bold;
            }}
            
            /* List Widget */
            QListWidget {{
                background-color: {Theme.BG_DARK};
                border: 1px solid {Theme.BORDER};
                border-radius: 5px;
            }}
            QListWidget::item {{
                padding: 10px;
            }}
            QListWidget::item:selected {{
                background-color: {Theme.SECONDARY};
            }}
            QListWidget::item:hover:!selected {{
                background-color: {Theme.BG_LIGHT};
            }}
            
            /* Tree Widget */
            QTreeWidget {{
                background-color: {Theme.BG_DARK};
                border: 1px solid {Theme.BORDER};
                border-radius: 5px;
            }}
            QTreeWidget::item {{
                padding: 4px 8px;
            }}
            QTreeWidget::item:selected {{
                background-color: {Theme.SECONDARY};
            }}
            QTreeWidget::item:hover:!selected {{
                background-color: {Theme.BG_LIGHT};
            }}
            QTreeWidget::branch:has-children:!has-siblings:closed,
            QTreeWidget::branch:closed:has-children:has-siblings {{
                border-image: none;
                image: url(none);
            }}
            QTreeWidget::branch:open:has-children:!has-siblings,
            QTreeWidget::branch:open:has-children:has-siblings {{
                border-image: none;
                image: url(none);
            }}
            
            /* ScrollBar */
            QScrollBar:vertical {{
                background-color: {Theme.BG_DARK};
                width: 10px;
                border-radius: 5px;
                margin: 2px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {Theme.BG_LIGHTER};
                border-radius: 5px;
                min-height: 30px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: {Theme.SECONDARY};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar:horizontal {{
                background-color: {Theme.BG_DARK};
                height: 10px;
                border-radius: 5px;
                margin: 2px;
            }}
            QScrollBar::handle:horizontal {{
                background-color: {Theme.BG_LIGHTER};
                border-radius: 5px;
                min-width: 30px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background-color: {Theme.SECONDARY};
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0px;
            }}
            
            /* Progress Bar */
            QProgressBar {{
                background-color: {Theme.BG_MEDIUM};
                border-radius: 5px;
                text-align: center;
                color: {Theme.TEXT_PRIMARY};
                min-height: 22px;
                border: 1px solid {Theme.BORDER};
            }}
            QProgressBar::chunk {{
                background-color: {Theme.SECONDARY};
                border-radius: 4px;
            }}
            
            /* Text Edit */
            QTextEdit {{
                background-color: {Theme.BG_DARK};
                border: 1px solid {Theme.BORDER};
                border-radius: 5px;
                padding: 10px;
            }}
            
            /* CheckBox - larger with more spacing */
            QCheckBox {{
                spacing: 10px;
                color: {Theme.TEXT_PRIMARY};
                padding: 6px 0px;
                min-height: 26px;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 2px solid {Theme.BORDER};
                background-color: {Theme.BG_MEDIUM};
            }}
            QCheckBox::indicator:checked {{
                background-color: {Theme.PRIMARY};
                border-color: {Theme.PRIMARY};
            }}
            QCheckBox::indicator:hover {{
                border-color: {Theme.PRIMARY_LIGHT};
            }}
            
            /* Scroll Area */
            QScrollArea {{
                border: none;
                background-color: transparent;
            }}
            
            /* Splitter */
            QSplitter::handle {{
                background-color: {Theme.BORDER};
            }}
            QSplitter::handle:hover {{
                background-color: {Theme.SECONDARY};
            }}
            
            /* Menu */
            QMenu {{
                background-color: {Theme.BG_DARK};
                border: 1px solid {Theme.BORDER};
                border-radius: 6px;
                padding: 6px;
            }}
            QMenu::item {{
                padding: 10px 24px;
                border-radius: 4px;
            }}
            QMenu::item:selected {{
                background-color: {Theme.SECONDARY};
            }}
            
            /* ToolTip */
            QToolTip {{
                background-color: {Theme.BG_MEDIUM};
                color: {Theme.TEXT_PRIMARY};
                border: 1px solid {Theme.BORDER};
                padding: 8px;
                border-radius: 4px;
            }}
        """


def get_icon_path() -> str:
    """Get the path to the application icon."""
    # Try multiple locations
    locations = [
        Path(__file__).parent / "resources" / "icon.ico",
        Path(__file__).parent.parent / "resources" / "icon.ico",
        Path("icon.ico"),
    ]
    for loc in locations:
        if loc.exists():
            return str(loc)
    return ""


class ShortcutEditor(QWidget):
    """Widget for editing a single shortcut."""
    
    def __init__(self, action_name: str, display_name: str, parent=None):
        super().__init__(parent)
        self.action_name = action_name
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.label = QLabel(display_name)
        self.label.setMinimumWidth(150)
        
        self.key_edit = QKeySequenceEdit()
        current = config.get_shortcut(action_name)
        if current:
            self.key_edit.setKeySequence(QKeySequence(current))
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear)
        
        layout.addWidget(self.label)
        layout.addWidget(self.key_edit, 1)
        layout.addWidget(self.clear_btn)
    
    def _clear(self):
        self.key_edit.clear()
    
    def get_shortcut(self) -> str:
        return self.key_edit.keySequence().toString()


class SettingsDialog(QDialog):
    """Settings dialog for customizing shortcuts and preferences."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumSize(500, 600)
        
        layout = QVBoxLayout(self)
        
        # Tabs
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        # Shortcuts tab
        shortcuts_widget = QWidget()
        shortcuts_layout = QVBoxLayout(shortcuts_widget)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        # Shortcut editors
        self.shortcut_editors = []
        
        shortcut_definitions = [
            ("Global", [
                ("quit", "Quit Application"),
                ("run_analysis", "Run Analysis"),
            ]),
            ("Labeling", [
                ("label_normal", "Label as Normal"),
                ("label_rod", "Label as ROD"),
                ("label_artifact", "Label as Artifact"),
                ("add_mode", "Add Mode"),
                ("select_mode", "Select Mode"),
                ("delete", "Delete Selected"),
                ("merge", "Merge Selected"),
                ("next_image", "Next Image"),
                ("prev_image", "Previous Image"),
                ("next_deposit", "Next Deposit"),
                ("prev_deposit", "Previous Deposit"),
            ]),
            ("Results", [
                ("export_excel", "Export to Excel"),
                ("open_detail", "Open Detail View"),
            ]),
        ]
        
        for group_name, shortcuts in shortcut_definitions:
            group = QGroupBox(group_name)
            group_layout = QVBoxLayout()
            
            for action_name, display_name in shortcuts:
                editor = ShortcutEditor(action_name, display_name)
                self.shortcut_editors.append(editor)
                group_layout.addWidget(editor)
            
            group.setLayout(group_layout)
            scroll_layout.addWidget(group)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        shortcuts_layout.addWidget(scroll)
        
        # Reset button
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self._reset_shortcuts)
        shortcuts_layout.addWidget(reset_btn)
        
        tabs.addTab(shortcuts_widget, "Shortcuts")
        
        # Detection tab
        detection_widget = QWidget()
        detection_layout = QFormLayout(detection_widget)
        
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(1, 1000)
        self.min_area_spin.setValue(config.get("detection.min_area", 20))
        detection_layout.addRow("Min Area:", self.min_area_spin)
        
        self.max_area_spin = QSpinBox()
        self.max_area_spin.setRange(100, 50000)
        self.max_area_spin.setValue(config.get("detection.max_area", 10000))
        detection_layout.addRow("Max Area:", self.max_area_spin)
        
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.1, 1.0)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(config.get("detection.threshold", 0.6))
        detection_layout.addRow("Circularity Threshold:", self.threshold_spin)
        
        tabs.addTab(detection_widget, "Detection")
        
        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self._save_and_close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def _reset_shortcuts(self):
        config.reset_shortcuts()
        # Reload editors
        for editor in self.shortcut_editors:
            current = config.get_shortcut(editor.action_name)
            if current:
                editor.key_edit.setKeySequence(QKeySequence(current))
            else:
                editor.key_edit.clear()
        QMessageBox.information(self, "Reset", "Shortcuts reset to defaults")
    
    def _save_and_close(self):
        # Save shortcuts
        for editor in self.shortcut_editors:
            config.set_shortcut(editor.action_name, editor.get_shortcut())
        
        # Save detection settings
        config.set("detection.min_area", self.min_area_spin.value())
        config.set("detection.max_area", self.max_area_spin.value())
        config.set("detection.threshold", self.threshold_spin.value())
        
        self.accept()


class WorkerThread(QThread):
    """Background worker for long-running tasks."""
    progress = Signal(int, int)
    status = Signal(str)
    finished = Signal(object)
    error = Signal(str)
    
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
    
    def run(self):
        try:
            result = self.func(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}")


class PathSelector(QWidget):
    """Widget for selecting file/folder paths."""
    
    pathChanged = Signal(str)  # Emitted when path changes
    
    def __init__(self, label: str, is_folder: bool = False, filter: str = "", config_key: str = "", default_path: str = ""):
        super().__init__()
        self.is_folder = is_folder
        self.filter = filter
        self.config_key = config_key
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(10)
        
        # Label - bold, no colon, no separate box
        label_text = label.rstrip(':')  # Remove colon if present
        self.label = QLabel(label_text)
        self.label.setMinimumWidth(70)
        self.label.setStyleSheet(f"""
            font-weight: bold;
            color: {Theme.TEXT_PRIMARY};
            background-color: transparent;
        """)
        
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText(f"Select {label_text.lower()}...")
        self.path_edit.textChanged.connect(self.pathChanged.emit)
        
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.setMinimumWidth(80)
        self.browse_btn.clicked.connect(self._browse)
        
        layout.addWidget(self.label)
        layout.addWidget(self.path_edit, 1)
        layout.addWidget(self.browse_btn)
        
        # Load from config, fallback to default_path
        if config_key:
            saved_path = config.get(config_key, "")
            if saved_path:
                self.path_edit.setText(saved_path)
            elif default_path:
                self.path_edit.setText(default_path)
    
    def _browse(self):
        start_dir = self.path_edit.text().replace('/', '\\') or ""  # Convert back for dialog
        
        if self.is_folder:
            path = QFileDialog.getExistingDirectory(self, f"Select {self.label.text()}", start_dir)
        else:
            path, _ = QFileDialog.getOpenFileName(self, f"Select {self.label.text()}", start_dir, self.filter)
        
        if path:
            # Display with forward slashes to avoid KRW symbol on Korean Windows
            display_path = path.replace('\\', '/')
            self.path_edit.setText(display_path)
            if self.config_key:
                config.set(self.config_key, path)  # Store original path
    
    def path(self) -> str:
        # Return path with system-appropriate separators
        return self.path_edit.text().replace('/', '\\') if sys.platform == 'win32' else self.path_edit.text()
    
    def set_path(self, path: str):
        # Display with forward slashes
        display_path = path.replace('\\', '/') if path else ''
        self.path_edit.setText(display_path)
        if self.config_key:
            config.set(self.config_key, path)


class ZoomableGraphicsView(QGraphicsView):
    """QGraphicsView with Ctrl+wheel zoom support."""
    
    def __init__(self, scene=None):
        super().__init__(scene) if scene else super().__init__()
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
    
    def wheelEvent(self, event):
        """Handle Ctrl+wheel for zoom."""
        if event.modifiers() & Qt.ControlModifier:
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.scale(factor, factor)
        else:
            super().wheelEvent(event)


class ImageViewerDialog(QDialog):
    """Dialog for viewing images in full size with fit to window."""
    
    def __init__(self, image_path: str, title: str = "Image Viewer", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(900, 700)
        self.image_path = image_path
        
        layout = QVBoxLayout(self)
        
        # Use ZoomableGraphicsView for scaling with Ctrl+wheel
        self.scene = QGraphicsScene()
        self.view = ZoomableGraphicsView(self.scene)
        
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            self.pixmap_item = self.scene.addPixmap(pixmap)
            self.scene.setSceneRect(QRectF(pixmap.rect()))
        
        layout.addWidget(self.view)
        
        # Hint label
        hint_label = QLabel("Tip: Ctrl + Mouse wheel to zoom")
        hint_label.setStyleSheet(f"color: {Theme.TEXT_SECONDARY}; font-size: 11px;")
        hint_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(hint_label)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        fit_btn = QPushButton("Fit to Window")
        fit_btn.clicked.connect(self._fit_to_window)
        btn_layout.addWidget(fit_btn)
        
        actual_btn = QPushButton("Actual Size (100%)")
        actual_btn.clicked.connect(self._actual_size)
        btn_layout.addWidget(actual_btn)
        
        close_btn = QPushButton("Close (Esc)")
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(close_btn)
        
        layout.addLayout(btn_layout)
        
        # Escape to close
        QShortcut(QKeySequence(Qt.Key_Escape), self, self.close)
        
        # Fit to window on open
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
    
    def _fit_to_window(self):
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
    
    def _actual_size(self):
        self.view.resetTransform()
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Auto-fit on resize
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
    
    def showEvent(self, event):
        super().showEvent(event)
        # Fit after dialog is shown
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)


class NumericTableWidgetItem(QTableWidgetItem):
    """QTableWidgetItem that sorts numerically instead of alphabetically."""
    
    def __init__(self, value, display_format: str = None):
        if display_format:
            super().__init__(display_format.format(value))
        else:
            super().__init__(str(value))
        self._value = value
    
    def __lt__(self, other):
        if isinstance(other, NumericTableWidgetItem):
            return self._value < other._value
        return super().__lt__(other)


class TrainingTab(QWidget):
    """Training tab for model training."""
    
    def __init__(self):
        super().__init__()
        self._setup_ui()
    
    def _setup_ui(self):
        # Main layout for the tab
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setFrameShape(QScrollArea.NoFrame)
        
        # Scroll content widget
        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Data paths
        data_group = QGroupBox("Data")
        data_layout = QVBoxLayout()
        
        self.image_dir = PathSelector("Image Folder:", is_folder=True, config_key="last_image_dir")
        self.label_dir = PathSelector("Label Folder:", is_folder=True, config_key="last_label_dir")
        self.same_folder = QCheckBox("Same as image folder")
        self.same_folder.setChecked(True)
        self.same_folder.toggled.connect(self._toggle_label_dir)
        self.image_dir.pathChanged.connect(self._on_image_dir_changed)
        
        data_layout.addWidget(self.image_dir)
        data_layout.addWidget(self.label_dir)
        data_layout.addWidget(self.same_folder)
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        
        self._toggle_label_dir(True)
        
        # Model settings
        model_group = QGroupBox("Model Settings")
        model_layout = QFormLayout()
        
        self.model_type = NoScrollComboBox()
        self.model_type.addItems(["Random Forest", "CNN (PyTorch)", "U-Net Segmentation"])
        model_type_map = {"rf": 0, "cnn": 1, "unet": 2}
        self.model_type.setCurrentIndex(model_type_map.get(config.get("training.model_type", "rf"), 0))
        self.model_type.currentIndexChanged.connect(self._on_model_type_changed)
        model_layout.addRow("Model Type:", self.model_type)
        
        self.output_path = PathSelector("Output:", filter="Model (*.pkl *.pt)")
        model_layout.addRow(self.output_path)
        
        self.n_estimators = NoScrollSpinBox()
        self.n_estimators.setRange(10, 1000)
        self.n_estimators.setValue(config.get("training.n_estimators", 100))
        model_layout.addRow("RF Trees:", self.n_estimators)
        
        self.epochs = NoScrollSpinBox()
        self.epochs.setRange(1, 100)
        self.epochs.setValue(config.get("training.epochs", 20))
        model_layout.addRow("CNN/U-Net Epochs:", self.epochs)
        
        self.image_size = NoScrollSpinBox()
        self.image_size.setRange(128, 512)
        self.image_size.setValue(256)
        self.image_size.setSingleStep(64)
        model_layout.addRow("U-Net Image Size:", self.image_size)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Train button
        self.train_btn = QPushButton("Train Model")
        self.train_btn.setMinimumHeight(40)
        self.train_btn.clicked.connect(self._train)
        layout.addWidget(self.train_btn)
        
        # Progress
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        
        # Log
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(200)
        log_layout.addWidget(self.log)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        layout.addStretch()
        
        # Set scroll content and add to main layout
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)
    
    def _toggle_label_dir(self, checked):
        self.label_dir.setEnabled(not checked)
        if checked:
            # Sync label_dir path with image_dir path
            self.label_dir.set_path(self.image_dir.path())
    
    def _on_image_dir_changed(self, path):
        """When image dir changes, sync label dir if same_folder is checked."""
        if self.same_folder.isChecked():
            self.label_dir.set_path(path)
    
    def _on_model_type_changed(self, index):
        """Show/hide settings based on model type."""
        is_rf = (index == 0)
        is_unet = (index == 2)
        
        self.n_estimators.setEnabled(is_rf)
        self.image_size.setEnabled(is_unet)
    
    def _train(self):
        image_dir = self.image_dir.path()
        if not image_dir:
            QMessageBox.warning(self, "Error", "Please select image folder")
            return
        
        label_dir = image_dir if self.same_folder.isChecked() else self.label_dir.path()
        output_path = self.output_path.path()
        
        model_type_idx = self.model_type.currentIndex()
        model_type = ["rf", "cnn", "unet"][model_type_idx]
        
        if not output_path:
            ext = ".pkl" if model_type == "rf" else ".pt"
            output_path = str(Path(image_dir) / f"model_{model_type}{ext}")
            self.output_path.set_path(output_path)
        
        # Save settings
        config.set("training.model_type", model_type)
        config.set("training.n_estimators", self.n_estimators.value())
        config.set("training.epochs", self.epochs.value())
        
        self.log.clear()
        self.log.append(f"Starting {model_type.upper()} training...")
        self.train_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        
        if model_type == "unet":
            # U-Net segmentation training
            from .segmentation import train_segmentation_model
            
            def progress_callback(epoch, train_loss, val_loss, val_iou):
                self.log.append(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, IoU={val_iou:.3f}")
            
            self.worker = WorkerThread(
                train_segmentation_model,
                image_dir=image_dir,
                label_dir=label_dir,
                output_path=output_path,
                epochs=self.epochs.value(),
                image_size=self.image_size.value(),
                progress_callback=progress_callback
            )
        else:
            # RF or CNN classifier training
            kwargs = {}
            if model_type == "rf":
                kwargs['n_estimators'] = self.n_estimators.value()
            else:
                kwargs['epochs'] = self.epochs.value()
            
            # Lazy import trainer module
            from .trainer import train_from_labels
            
            self.worker = WorkerThread(
                train_from_labels,
                image_dir=image_dir,
                label_dir=label_dir,
                output_path=output_path,
                model_type=model_type,
                **kwargs
            )
        
        self.worker.finished.connect(self._on_train_finished)
        self.worker.error.connect(self._on_train_error)
        self.worker.start()
    
    def _on_train_finished(self, results):
        self.train_btn.setEnabled(True)
        self.progress.setVisible(False)
        
        self.log.append("\n" + "="*40)
        self.log.append("TRAINING COMPLETE!")
        self.log.append("="*40)
        
        if 'accuracy' in results:
            self.log.append(f"Accuracy: {results.get('accuracy', 0)*100:.1f}%")
        
        if 'cv_mean' in results:
            self.log.append(f"Cross-validation: {results['cv_mean']:.3f} (±{results['cv_std']:.3f})")
        
        if 'best_val_loss' in results:
            self.log.append(f"Best validation loss: {results['best_val_loss']:.4f}")
        
        if 'val_iou' in results and results['val_iou']:
            self.log.append(f"Final IoU: {results['val_iou'][-1]:.3f}")
        
        QMessageBox.information(self, "Success", f"Model saved to:\n{self.output_path.path()}")
    
    def _on_train_error(self, error_msg):
        self.train_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.log.append(f"\nERROR: {error_msg}")
        QMessageBox.critical(self, "Error", f"Training failed:\n{error_msg}")


class GroupEditorDialog(QDialog):
    """Dialog for grouping files into conditions with drag & drop."""
    
    def __init__(self, file_list: List[str], parent=None):
        super().__init__(parent)
        self.file_list = file_list
        self.groups: Dict[str, List[str]] = {}
        self.metadata_df = None
        
        self.setWindowTitle("Condition Editor")
        self.setMinimumSize(900, 600)
        
        self._setup_ui()
        self._populate_files()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Main content - splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left: Available files
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.addWidget(QLabel("Available Files (drag to group or right-click):"))
        
        self.file_list_widget = QListWidget()
        self.file_list_widget.setSelectionMode(QListWidget.ExtendedSelection)
        self.file_list_widget.setDragEnabled(True)
        self.file_list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.file_list_widget.customContextMenuRequested.connect(self._show_file_context_menu)
        left_layout.addWidget(self.file_list_widget)
        
        splitter.addWidget(left_widget)
        
        # Right: Groups
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.addWidget(QLabel("Groups:"))
        
        # Group buttons
        group_btn_layout = QHBoxLayout()
        add_group_btn = QPushButton("+ New Group")
        add_group_btn.clicked.connect(self._add_group)
        group_btn_layout.addWidget(add_group_btn)
        
        remove_group_btn = QPushButton("- Remove Group")
        remove_group_btn.clicked.connect(self._remove_group)
        group_btn_layout.addWidget(remove_group_btn)
        
        group_btn_layout.addStretch()
        right_layout.addLayout(group_btn_layout)
        
        # Groups container (scroll area)
        self.groups_scroll = QScrollArea()
        self.groups_scroll.setWidgetResizable(True)
        self.groups_container = QWidget()
        self.groups_layout = QVBoxLayout(self.groups_container)
        self.groups_scroll.setWidget(self.groups_container)
        right_layout.addWidget(self.groups_scroll)
        
        splitter.addWidget(right_widget)
        splitter.setSizes([400, 500])
        
        layout.addWidget(splitter)
        
        # Column name
        col_layout = QHBoxLayout()
        col_layout.addWidget(QLabel("Group Column Name:"))
        self.column_name_edit = QLineEdit("condition")
        self.column_name_edit.setMaximumWidth(200)
        col_layout.addWidget(self.column_name_edit)
        col_layout.addStretch()
        layout.addLayout(col_layout)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        save_csv_btn = QPushButton("Save as CSV")
        save_csv_btn.clicked.connect(self._save_csv)
        btn_layout.addWidget(save_csv_btn)
        
        apply_btn = QPushButton("Apply && Close")
        apply_btn.setStyleSheet(Theme.button_style(Theme.PRIMARY))
        apply_btn.clicked.connect(self._apply_and_close)
        btn_layout.addWidget(apply_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        layout.addLayout(btn_layout)
    
    def _populate_files(self):
        for f in self.file_list:
            self.file_list_widget.addItem(Path(f).name)
    
    def _show_file_context_menu(self, pos):
        menu = QMenu(self)
        
        # Add to existing groups
        if self.groups:
            for group_name in self.groups.keys():
                action = menu.addAction(f"Add to '{group_name}'")
                action.triggered.connect(lambda checked, g=group_name: self._add_selected_to_group(g))
        
        menu.addSeparator()
        
        # Create new group
        new_group_action = menu.addAction("Create new group...")
        new_group_action.triggered.connect(self._add_group_with_selection)
        
        menu.exec(self.file_list_widget.mapToGlobal(pos))
    
    def _add_group(self):
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "New Group", "Group name:")
        if ok and name and name not in self.groups:
            self.groups[name] = []
            list_widget = self._create_group_widget(name)
            # Auto-add selected files to the new group
            selected = self.file_list_widget.selectedItems()
            for item in selected:
                filename = item.text()
                if filename and filename not in self.groups[name]:
                    self.groups[name].append(filename)
                    list_widget.addItem(filename)
    
    def _add_group_with_selection(self):
        # Same as _add_group now
        self._add_group()
    
    def _create_group_widget(self, name: str):
        group_box = QGroupBox(name)
        group_box.setObjectName(f"group_{name}")
        group_layout = QVBoxLayout()
        
        list_widget = QListWidget()
        list_widget.setObjectName(f"list_{name}")
        list_widget.setSelectionMode(QListWidget.ExtendedSelection)
        # Disable drag&drop (unreliable) - use buttons instead
        list_widget.setDragEnabled(False)
        list_widget.setAcceptDrops(False)
        
        # Buttons
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("+ Add Selected")
        add_btn.clicked.connect(lambda checked, n=name, lw=list_widget: self._add_selected_to_group_widget(n, lw))
        btn_layout.addWidget(add_btn)
        
        remove_btn = QPushButton("- Remove")
        remove_btn.clicked.connect(lambda checked, n=name, lw=list_widget: self._remove_from_group(n, lw))
        btn_layout.addWidget(remove_btn)
        
        group_layout.addWidget(list_widget)
        group_layout.addLayout(btn_layout)
        group_box.setLayout(group_layout)
        
        self.groups_layout.addWidget(group_box)
        return list_widget
    
    def _add_selected_to_group_widget(self, group_name: str, list_widget: QListWidget):
        """Add selected files from file list to a specific group widget."""
        selected = self.file_list_widget.selectedItems()
        if not selected:
            return
        
        for item in selected:
            filename = item.text()
            if filename and filename not in self.groups[group_name]:
                self.groups[group_name].append(filename)
                list_widget.addItem(filename)
    
    def _add_selected_to_group(self, group_name: str):
        """Add selected files to a group (called from context menu)."""
        selected = self.file_list_widget.selectedItems()
        if not selected:
            return
        
        # Find the group's list widget
        for i in range(self.groups_layout.count()):
            widget = self.groups_layout.itemAt(i).widget()
            if widget and widget.objectName() == f"group_{group_name}":
                list_widget = widget.findChild(QListWidget, f"list_{group_name}")
                if list_widget:
                    for item in selected:
                        filename = item.text()
                        if filename and filename not in self.groups[group_name]:
                            self.groups[group_name].append(filename)
                            list_widget.addItem(filename)
                break
    
    def _on_drop(self, group_name: str, list_widget: QListWidget):
        # Not used anymore - kept for compatibility
        pass
    
    def _remove_from_group(self, group_name: str, list_widget: QListWidget):
        for item in list_widget.selectedItems():
            filename = item.text()
            if filename in self.groups[group_name]:
                self.groups[group_name].remove(filename)
            list_widget.takeItem(list_widget.row(item))
    
    def _remove_group(self):
        if not self.groups:
            return
        
        from PySide6.QtWidgets import QInputDialog
        group_name, ok = QInputDialog.getItem(
            self, "Remove Group", "Select group to remove:",
            list(self.groups.keys()), 0, False
        )
        if ok and group_name:
            del self.groups[group_name]
            # Remove widget
            for i in range(self.groups_layout.count()):
                widget = self.groups_layout.itemAt(i).widget()
                if widget and widget.objectName() == f"group_{group_name}":
                    widget.deleteLater()
                    break
    
    def _generate_metadata(self) -> pd.DataFrame:
        column_name = self.column_name_edit.text() or "condition"
        records = []
        
        for group_name, files in self.groups.items():
            for filename in files:
                records.append({
                    'filename': filename,
                    column_name: group_name
                })
        
        # Add ungrouped files
        grouped_files = set()
        for files in self.groups.values():
            grouped_files.update(files)
        
        for f in self.file_list:
            filename = Path(f).name
            if filename not in grouped_files:
                records.append({
                    'filename': filename,
                    column_name: 'ungrouped'
                })
        
        return pd.DataFrame(records)
    
    def _save_csv(self):
        df = self._generate_metadata()
        path, _ = QFileDialog.getSaveFileName(self, "Save Metadata", "", "CSV (*.csv)")
        if path:
            df.to_csv(path, index=False)
            QMessageBox.information(self, "Saved", f"Metadata saved to:\n{path}")
    
    def _apply_and_close(self):
        self.metadata_df = self._generate_metadata()
        self.accept()
    
    def get_metadata(self) -> Optional[pd.DataFrame]:
        return self.metadata_df


class AnalysisTab(QWidget):
    """Analysis tab for running analysis on images."""
    
    analysis_complete = Signal(dict)
    
    def __init__(self):
        super().__init__()
        self._metadata = None  # In-memory metadata from GroupEditor
        self._group_data = {}  # {group_name: [file1, file2, ...]}
        self._setup_ui()
    
    def _setup_ui(self):
        # Main layout for the tab
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setFrameShape(QScrollArea.NoFrame)
        
        # Container widget for scroll area content
        scroll_content = QWidget()
        scroll_content.setStyleSheet("background-color: transparent;")
        layout = QVBoxLayout(scroll_content)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        
        # Input/Output
        io_group = QGroupBox("Input / Output")
        io_layout = QVBoxLayout()
        io_layout.setSpacing(6)
        io_layout.setContentsMargins(10, 12, 10, 10)
        
        # Default paths
        default_input = str(Path.home() / "SCAT" / "data" / "images")
        default_output = str(Path.home() / "SCAT" / "data" / "results")
        
        self.input_dir = PathSelector("Input", is_folder=True, config_key="last_input_dir", default_path=default_input)
        self.output_dir = PathSelector("Output", is_folder=True, config_key="last_output_dir", default_path=default_output)
        self.model_path = PathSelector("Classifier", filter="Model (*.pkl *.pt)", config_key="last_model_path")
        self.detection_model_path = PathSelector("Detection (U-Net)", filter="Model (*.pt)", config_key="last_detection_model_path")
        self.detection_model_path.setToolTip("Optional: U-Net model for improved deposit detection")
        
        # Groups section
        groups_group = QGroupBox("Groups")
        groups_layout = QVBoxLayout()
        groups_layout.setSpacing(6)
        groups_layout.setContentsMargins(10, 12, 10, 10)
        
        # Use groups checkbox
        self.use_groups = QCheckBox("Use groups for comparison")
        self.use_groups.setChecked(config.get("analysis.use_groups", True))
        self.use_groups.toggled.connect(self._on_use_groups_toggled)
        groups_layout.addWidget(self.use_groups)
        
        # Create button
        self.create_groups_btn = QPushButton("Create Groups...")
        self.create_groups_btn.clicked.connect(self._open_group_editor)
        groups_layout.addWidget(self.create_groups_btn)
        
        # Groups tree (expandable - click to show files)
        from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem
        self.groups_tree = QTreeWidget()
        self.groups_tree.setHeaderHidden(True)
        self.groups_tree.setMinimumHeight(150)
        self.groups_tree.setMaximumHeight(200)
        self.groups_tree.setIndentation(15)
        self.groups_tree.setAnimated(True)
        # Single click to expand/collapse
        self.groups_tree.itemClicked.connect(self._on_group_tree_clicked)
        groups_layout.addWidget(self.groups_tree)
        
        groups_group.setLayout(groups_layout)
        
        io_layout.addWidget(self.input_dir)
        io_layout.addWidget(self.output_dir)
        io_layout.addWidget(self.model_path)
        io_layout.addWidget(self.detection_model_path)
        io_group.setLayout(io_layout)
        layout.addWidget(io_group)
        
        # Add groups section after I/O
        layout.addWidget(groups_group)
        
        # Update groups UI state
        self._on_use_groups_toggled(self.use_groups.isChecked())
        
        # Options
        options_group = QGroupBox("Options")
        options_layout = QFormLayout()
        options_layout.setVerticalSpacing(8)
        options_layout.setHorizontalSpacing(12)
        options_layout.setContentsMargins(10, 12, 10, 10)
        
        self.model_type = NoScrollComboBox()
        self.model_type.addItems(["Threshold", "Random Forest", "CNN"])
        model_type_map = {"threshold": 0, "rf": 1, "cnn": 2}
        self.model_type.setCurrentIndex(model_type_map.get(config.get("analysis.model_type", "rf"), 1))
        options_layout.addRow("Classifier", self.model_type)
        
        self.annotate = QCheckBox("Generate annotated images")
        self.annotate.setChecked(config.get("analysis.annotate", True))
        options_layout.addRow(self.annotate)
        
        self.visualize = QCheckBox("Generate visualizations")
        self.visualize.setChecked(config.get("analysis.visualize", True))
        options_layout.addRow(self.visualize)
        
        self.spatial = QCheckBox("Spatial analysis")
        self.spatial.setChecked(config.get("analysis.spatial", True))
        options_layout.addRow(self.spatial)
        
        self.stats = QCheckBox("Statistical analysis")
        self.stats.setChecked(config.get("analysis.stats", True))
        options_layout.addRow(self.stats)
        
        self.report = QCheckBox("Generate HTML report")
        self.report.setChecked(config.get("analysis.report", True))
        options_layout.addRow(self.report)
        
        self.save_json = QCheckBox("Save for retraining (JSON)")
        self.save_json.setChecked(config.get("analysis.save_json", True))
        self.save_json.setToolTip("Save contour data for model retraining. Disable to reduce file size.")
        options_layout.addRow(self.save_json)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # Detection settings
        detect_group = QGroupBox("Detection Settings")
        detect_layout = QFormLayout()
        detect_layout.setVerticalSpacing(8)
        detect_layout.setHorizontalSpacing(12)
        detect_layout.setContentsMargins(10, 12, 10, 10)
        
        self.min_area = NoScrollSpinBox()
        self.min_area.setRange(1, 1000)
        self.min_area.setValue(config.get("detection.min_area", 20))
        detect_layout.addRow("Min Area", self.min_area)
        
        self.max_area = NoScrollSpinBox()
        self.max_area.setRange(100, 50000)
        self.max_area.setValue(config.get("detection.max_area", 10000))
        detect_layout.addRow("Max Area", self.max_area)
        
        self.threshold = NoScrollDoubleSpinBox()
        self.threshold.setRange(0.1, 1.0)
        self.threshold.setSingleStep(0.05)
        self.threshold.setValue(config.get("detection.threshold", 0.6))
        detect_layout.addRow("Circularity", self.threshold)
        
        detect_group.setLayout(detect_layout)
        layout.addWidget(detect_group)
        
        # Run button - PRIMARY color for emphasis
        self.run_btn = QPushButton("▶  Run Analysis")
        self.run_btn.setMinimumHeight(48)
        self.run_btn.setStyleSheet(Theme.button_style(Theme.PRIMARY, "#FFFFFF", Theme.PRIMARY_LIGHT))
        self.run_btn.clicked.connect(self._run_analysis)
        layout.addWidget(self.run_btn)
        
        # Progress
        progress_layout = QVBoxLayout()
        
        progress_bar_layout = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress_label = QLabel("")
        progress_bar_layout.addWidget(self.progress)
        progress_bar_layout.addWidget(self.progress_label)
        progress_layout.addLayout(progress_bar_layout)
        
        self.eta_label = QLabel("")
        self.eta_label.setStyleSheet(f"color: {Theme.TEXT_SECONDARY};")
        progress_layout.addWidget(self.eta_label)
        
        layout.addLayout(progress_layout)
        
        layout.addStretch()
        
        # Initialize timing variables
        self._start_time = None
        
        # Set scroll content and add to main layout
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)
    
    def _save_settings(self):
        """Save current settings to config."""
        model_types = ['threshold', 'rf', 'cnn']
        config.set("analysis.model_type", model_types[self.model_type.currentIndex()])
        config.set("analysis.use_groups", self.use_groups.isChecked())
        config.set("analysis.annotate", self.annotate.isChecked())
        config.set("analysis.visualize", self.visualize.isChecked())
        config.set("analysis.spatial", self.spatial.isChecked())
        config.set("analysis.stats", self.stats.isChecked())
        config.set("analysis.report", self.report.isChecked())
        config.set("analysis.save_json", self.save_json.isChecked())
        config.set("detection.min_area", self.min_area.value())
        config.set("detection.max_area", self.max_area.value())
        config.set("detection.threshold", self.threshold.value())
    
    def _on_use_groups_toggled(self, checked):
        """Enable/disable groups UI based on checkbox."""
        self.create_groups_btn.setEnabled(checked)
        self.groups_tree.setEnabled(checked)
    
    def _on_group_tree_clicked(self, item, column):
        """Toggle expand/collapse on single click for parent items."""
        if item.childCount() > 0:  # Only for parent items (groups)
            item.setExpanded(not item.isExpanded())
    
    def _update_groups_list(self, group_data: dict):
        """Update the groups tree widget with group data."""
        from PySide6.QtWidgets import QTreeWidgetItem
        
        self._group_data = group_data
        self.groups_tree.clear()
        
        for group_name, files in sorted(group_data.items()):
            # Parent item (group name with count)
            parent = QTreeWidgetItem([f"{group_name} ({len(files)} files)"])
            parent.setExpanded(False)
            
            # Child items (file names)
            for filename in sorted(files):
                child = QTreeWidgetItem([f"  {filename}"])
                child.setForeground(0, QColor(Theme.TEXT_SECONDARY))
                parent.addChild(child)
            
            self.groups_tree.addTopLevelItem(parent)
    
    def _open_group_editor(self):
        """Open the group editor dialog to create metadata."""
        input_dir = self.input_dir.path()
        if not input_dir:
            QMessageBox.warning(self, "No Input", "Please select an input folder first.")
            return
        
        # Get list of image files (case-insensitive, no duplicates)
        input_path = Path(input_dir)
        image_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
        
        # Use lowercase extension check to avoid duplicates
        files = []
        seen = set()
        for f in input_path.iterdir():
            if f.is_file() and f.suffix.lower() in image_extensions:
                # Use lowercase name as key to prevent duplicates
                key = f.name.lower()
                if key not in seen:
                    seen.add(key)
                    files.append(f)
        
        if not files:
            QMessageBox.warning(self, "No Files", "No image files found in the input folder.")
            return
        
        file_paths = [str(f) for f in sorted(files)]
        
        dialog = GroupEditorDialog(file_paths, self)
        if dialog.exec():
            metadata = dialog.get_metadata()
            if metadata is not None and len(metadata) > 0:
                # Store metadata in memory
                self._metadata = metadata
                
                # Build group data for the list
                group_col = metadata.columns[1]  # Second column is the group
                group_data = {}
                for group_name in metadata[group_col].unique():
                    if group_name != 'ungrouped':
                        files = metadata[metadata[group_col] == group_name]['filename'].tolist()
                        group_data[group_name] = files
                
                # Update internal storage
                self._group_data = group_data
                
                # Update the groups tree
                self._update_groups_list(group_data)
    
    def _run_analysis(self):
        input_dir = self.input_dir.path()
        output_base = self.output_dir.path()
        
        if not input_dir:
            QMessageBox.warning(self, "Error", "Please select input folder")
            return
        
        if not output_base:
            output_base = str(Path(input_dir).parent)
        
        # Create timestamped output folder
        output_dir = get_timestamped_output_dir(Path(output_base))
        
        self._save_settings()
        
        self.run_btn.setEnabled(False)
        self.progress.setValue(0)
        self.eta_label.setText("")
        self._output_dir = str(output_dir)
        
        # Record start time for ETA calculation
        import time
        self._start_time = time.time()
        
        self.worker = WorkerThread(self._do_analysis, str(output_dir))
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()
    
    def _do_analysis(self, output_dir: str):
        from PIL import Image
        
        input_path = Path(self.input_dir.path())
        output_path = Path(output_dir)
        
        image_paths = list(input_path.glob('*.tif')) + list(input_path.glob('*.tiff'))
        image_paths += list(input_path.glob('*.png')) + list(input_path.glob('*.jpg'))
        
        if not image_paths:
            raise ValueError(f"No images found in {input_path}")
        
        model_types = ['threshold', 'rf', 'cnn']
        model_type = model_types[self.model_type.currentIndex()]
        
        config_obj = ClassifierConfig(
            model_type=model_type,
            circularity_threshold=self.threshold.value(),
            model_path=self.model_path.path() or None
        )
        
        # Get detection model path (U-Net)
        detection_model = self.detection_model_path.path() or None
        
        detector = DepositDetector(
            min_area=self.min_area.value(),
            max_area=self.max_area.value(),
            sensitive_mode=True,  # Always use sensitive detection (if no U-Net)
            unet_model_path=detection_model
        )
        
        analyzer = Analyzer(detector=detector, classifier_config=config_obj)
        
        # Use in-memory metadata if available and use_groups is checked
        metadata = None
        group_by = None
        if self.use_groups.isChecked():
            if self._group_data:  # In-memory groups from GroupEditor
                # Create metadata DataFrame from group data
                rows = []
                for group_name, files in self._group_data.items():
                    for filename in files:
                        rows.append({'filename': filename, 'group': group_name})
                if rows:
                    metadata = pd.DataFrame(rows)
                    group_by = ['group']
            elif hasattr(self, '_metadata') and self._metadata is not None:
                metadata = self._metadata
                # Rename second column to 'group' for consistency
                if len(metadata.columns) > 1:
                    metadata = metadata.rename(columns={metadata.columns[1]: 'group'})
                    group_by = ['group']
        
        results = []
        spatial_results = []
        
        for i, path in enumerate(image_paths):
            self.worker.progress.emit(i + 1, len(image_paths))
            result = analyzer.analyze_image(path)
            results.append(result)
            
            if self.spatial.isChecked():
                from .spatial import SpatialAnalyzer
                img = np.array(Image.open(path))
                spatial_analyzer = SpatialAnalyzer()
                spatial_result = spatial_analyzer.analyze(result.deposits, img.shape[:2])
                spatial_results.append(spatial_result)
        
        reporter = ReportGenerator(output_path)
        reports = reporter.save_all(results, metadata, group_by, save_json=self.save_json.isChecked())
        
        if self.annotate.isChecked():
            annotated_dir = output_path / 'annotated'
            annotated_dir.mkdir(exist_ok=True)
            for path, result in zip(image_paths, results):
                img = np.array(Image.open(path))
                annotated = analyzer.generate_annotated_image(img, result.deposits)
                Image.fromarray(annotated).save(annotated_dir / f"{path.stem}_annotated.png")
        
        viz_results = {}
        if self.visualize.isChecked():
            from .visualization import generate_all_visualizations, generate_spatial_visualizations
            viz_dir = output_path / 'visualizations'
            viz_results = generate_all_visualizations(
                reports['film_summary'], reports['deposit_data'], viz_dir,
                group_by=group_by[0] if group_by else None
            )
            if spatial_results:
                spatial_viz = generate_spatial_visualizations(spatial_results, viz_dir)
                viz_results.update(spatial_viz)
        
        stats_results = {}
        if self.stats.isChecked() and group_by:
            from .statistics import generate_statistics_report
            # Check if group column exists in film_summary
            if group_by[0] in reports['film_summary'].columns:
                stats_results = generate_statistics_report(reports['film_summary'], group_column=group_by[0])
            else:
                # Log warning - group column not found
                print(f"Warning: Group column '{group_by[0]}' not found in film_summary. "
                      f"Available columns: {list(reports['film_summary'].columns)}")
        
        spatial_stats = {}
        if spatial_results:
            from .spatial import aggregate_spatial_stats
            spatial_stats = aggregate_spatial_stats(spatial_results)
        
        if self.report.isChecked():
            from .report import generate_report
            generate_report(
                film_summary=reports['film_summary'],
                output_dir=output_path,
                deposit_data=reports['deposit_data'],
                spatial_stats=spatial_stats,
                statistical_results=stats_results,
                visualization_paths=viz_results,
                group_by=group_by[0] if group_by else None
            )
        
        return {
            'film_summary': reports['film_summary'],
            'deposit_data': reports['deposit_data'],
            'spatial_stats': spatial_stats,
            'stats_results': stats_results,
            'viz_results': viz_results,
            'output_dir': str(output_path),
            'image_paths': [str(p) for p in image_paths]
        }
    
    def _on_progress(self, current, total):
        import time
        
        self.progress.setMaximum(total)
        self.progress.setValue(current)
        self.progress_label.setText(f"{current}/{total}")
        
        # Calculate ETA
        if self._start_time and current > 0:
            elapsed = time.time() - self._start_time
            avg_per_item = elapsed / current
            remaining = total - current
            eta_seconds = avg_per_item * remaining
            
            if eta_seconds < 60:
                eta_text = f"ETA: {int(eta_seconds)}s remaining"
            elif eta_seconds < 3600:
                minutes = int(eta_seconds // 60)
                seconds = int(eta_seconds % 60)
                eta_text = f"ETA: {minutes}m {seconds}s remaining"
            else:
                hours = int(eta_seconds // 3600)
                minutes = int((eta_seconds % 3600) // 60)
                eta_text = f"ETA: {hours}h {minutes}m remaining"
            
            self.eta_label.setText(eta_text)
    
    def _on_finished(self, results):
        self.run_btn.setEnabled(True)
        self.progress_label.setText("Complete!")
        self.eta_label.setText("")
        self.analysis_complete.emit(results)
        QMessageBox.information(self, "Analysis Complete", f"Results saved to:\n{results['output_dir']}")
    
    def _on_error(self, error_msg):
        self.run_btn.setEnabled(True)
        self.progress_label.setText("Error!")
        QMessageBox.critical(self, "Error", f"Analysis failed:\n{error_msg}")


class ResultsTab(QWidget):
    """Results tab for viewing analysis results."""
    
    def __init__(self):
        super().__init__()
        self.results = None
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        self.sub_tabs = QTabWidget()
        layout.addWidget(self.sub_tabs)
        
        # Overview tab
        self.overview_widget = QWidget()
        overview_layout = QVBoxLayout(self.overview_widget)
        
        # Top row: Summary + Buttons
        top_row = QHBoxLayout()
        
        # Summary (left)
        summary_group = QGroupBox("Analysis Summary")
        summary_layout = QVBoxLayout()
        self.summary_label = QLabel("No results loaded. Run analysis first.")
        self.summary_label.setWordWrap(True)
        summary_layout.addWidget(self.summary_label)
        summary_group.setLayout(summary_layout)
        top_row.addWidget(summary_group, 2)
        
        # Buttons (right)
        buttons_group = QGroupBox("Export")
        buttons_layout = QVBoxLayout()
        
        self.open_folder_btn = QPushButton("📁 Open Output Folder")
        self.open_folder_btn.clicked.connect(self._open_folder)
        buttons_layout.addWidget(self.open_folder_btn)
        
        self.export_excel_btn = QPushButton("📊 Export to Excel")
        self.export_excel_btn.clicked.connect(self._export_excel)
        buttons_layout.addWidget(self.export_excel_btn)
        
        self.open_report_btn = QPushButton("📄 Open HTML Report")
        self.open_report_btn.clicked.connect(self._open_report)
        buttons_layout.addWidget(self.open_report_btn)
        
        buttons_layout.addStretch()
        buttons_group.setLayout(buttons_layout)
        top_row.addWidget(buttons_group, 1)
        
        overview_layout.addLayout(top_row)
        
        # Table (double-click to view image)
        table_label = QLabel("Double-click a row to view and edit image details:")
        overview_layout.addWidget(table_label)
        
        self.summary_table = QTableWidget()
        self.summary_table.setColumnCount(6)
        self.summary_table.setHorizontalHeaderLabels(["Filename", "Normal", "ROD", "Artifact", "ROD %", "Total IOD"])
        self.summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.summary_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.summary_table.doubleClicked.connect(self._on_table_double_click)
        overview_layout.addWidget(self.summary_table)
        
        self.sub_tabs.addTab(self.overview_widget, "Overview")
        
        # Statistics tab (merged with Visualizations)
        self.stats_widget = QScrollArea()
        self.stats_widget.setWidgetResizable(True)
        self.stats_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.stats_content = QWidget()
        self.stats_layout = QVBoxLayout(self.stats_content)
        self.stats_widget.setWidget(self.stats_content)
        self.sub_tabs.addTab(self.stats_widget, "Statistics")
    
    def load_results(self, results: dict):
        self.results = results
        film_summary = results['film_summary']
        
        total_normal = film_summary['n_normal'].sum()
        total_rod = film_summary['n_rod'].sum()
        total_artifact = film_summary['n_artifact'].sum()
        mean_rod_frac = film_summary['rod_fraction'].mean()
        
        summary_text = f"""
        <h3>Summary</h3>
        <p><b>Total Films:</b> {len(film_summary)}</p>
        <p><b>Total Deposits:</b> {film_summary['n_total'].sum():.0f}</p>
        <p><b>Normal:</b> {total_normal:.0f} | <b>ROD:</b> {total_rod:.0f} | <b>Artifact:</b> {total_artifact:.0f}</p>
        <p><b>Mean ROD Fraction:</b> {mean_rod_frac*100:.1f}% (±{film_summary['rod_fraction'].std()*100:.1f}%)</p>
        <p><b>Output:</b> {results.get('output_dir', '')}</p>
        """
        self.summary_label.setText(summary_text)
        
        self.summary_table.setRowCount(len(film_summary))
        for i, row in film_summary.iterrows():
            self.summary_table.setItem(i, 0, QTableWidgetItem(str(row['filename'])))
            self.summary_table.setItem(i, 1, QTableWidgetItem(f"{row['n_normal']:.0f}"))
            self.summary_table.setItem(i, 2, QTableWidgetItem(f"{row['n_rod']:.0f}"))
            self.summary_table.setItem(i, 3, QTableWidgetItem(f"{row['n_artifact']:.0f}"))
            self.summary_table.setItem(i, 4, QTableWidgetItem(f"{row['rod_fraction']*100:.1f}%"))
            self.summary_table.setItem(i, 5, QTableWidgetItem(f"{row.get('total_iod', 0):.0f}"))
        
        self._load_statistics_tab(results)
    
    def _load_statistics_tab(self, results: dict):
        """Load combined statistics and visualizations."""
        # Clear existing
        while self.stats_layout.count():
            item = self.stats_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        film_summary = results['film_summary']
        viz_results = results.get('viz_results', {})
        stats_results = results.get('stats_results', {})
        spatial_stats = results.get('spatial_stats', {})
        
        # ===== Visualizations Section =====
        if viz_results:
            viz_group = QGroupBox("📊 Visualizations")
            viz_inner = QVBoxLayout()
            
            # Grid layout for images (2 columns)
            from PySide6.QtWidgets import QGridLayout
            viz_grid = QGridLayout()
            viz_grid.setSpacing(15)
            
            for idx, (name, path) in enumerate(viz_results.items()):
                btn = QPushButton()
                btn.setToolTip(f"Click to enlarge: {name}")
                
                pixmap = QPixmap(path)
                if not pixmap.isNull():
                    scaled = pixmap.scaled(380, 280, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    btn.setIcon(QIcon(scaled))
                    btn.setIconSize(scaled.size())
                    btn.setFixedSize(scaled.size() + QSize(10, 10))
                    btn.clicked.connect(lambda checked, p=path, n=name: self._show_image_dialog(p, n))
                
                container = QWidget()
                container_layout = QVBoxLayout(container)
                container_layout.setContentsMargins(0, 0, 0, 0)
                container_layout.addWidget(btn)
                label = QLabel(self._format_viz_name(name))
                label.setAlignment(Qt.AlignCenter)
                container_layout.addWidget(label)
                
                row = idx // 2
                col = idx % 2
                viz_grid.addWidget(container, row, col)
            
            viz_inner.addLayout(viz_grid)
            viz_group.setLayout(viz_inner)
            self.stats_layout.addWidget(viz_group)
        
        # ===== Descriptive Statistics Section =====
        desc_group = QGroupBox("📈 Descriptive Statistics")
        desc_layout = QVBoxLayout()
        
        desc_text = self._generate_descriptive_stats(film_summary)
        desc_label = QLabel(desc_text)
        desc_label.setWordWrap(True)
        desc_label.setTextFormat(Qt.RichText)
        desc_layout.addWidget(desc_label)
        
        desc_group.setLayout(desc_layout)
        self.stats_layout.addWidget(desc_group)
        
        # ===== Group Comparison Section =====
        if stats_results:
            comp_group = QGroupBox("📉 Group Comparisons")
            comp_layout = QVBoxLayout()
            
            comp_text = self._generate_comparison_stats(stats_results)
            comp_label = QLabel(comp_text)
            comp_label.setWordWrap(True)
            comp_label.setTextFormat(Qt.RichText)
            comp_layout.addWidget(comp_label)
            
            comp_group.setLayout(comp_layout)
            self.stats_layout.addWidget(comp_group)
        
        # ===== Spatial Analysis Section =====
        if spatial_stats:
            spatial_group = QGroupBox("🗺️ Spatial Analysis")
            spatial_layout = QVBoxLayout()
            
            spatial_text = self._generate_spatial_stats(spatial_stats)
            spatial_label = QLabel(spatial_text)
            spatial_label.setWordWrap(True)
            spatial_label.setTextFormat(Qt.RichText)
            spatial_layout.addWidget(spatial_label)
            
            spatial_group.setLayout(spatial_layout)
            self.stats_layout.addWidget(spatial_group)
        
        self.stats_layout.addStretch()
    
    def _format_viz_name(self, name: str) -> str:
        """Format visualization key names for display."""
        # Special mappings for known keys
        name_map = {
            'dashboard': 'Dashboard',
            'pca': 'PCA Analysis',
            'heatmap': 'Feature Heatmap',
            'scatter_matrix': 'Feature Relationships',
            'area_iod': 'Area vs IOD',
            'nnd_histogram': 'Nearest Neighbor Distance',
            'clark_evans': 'Clark-Evans Index',
            'density_map': 'Deposit Density Map',
            'quadrant_plot': 'Quadrant Analysis',
            'violin_total_iod': 'Total IOD Distribution',
            'violin_rod_fraction': 'ROD Fraction Distribution',
            'violin_n_deposits': 'Deposit Count Distribution',
            'violin_mean_area': 'Mean Area Distribution',
        }
        
        if name in name_map:
            return name_map[name]
        
        # Generic formatting: remove prefix, replace underscores, proper case
        formatted = name
        for prefix in ['violin_', 'box_', 'bar_', 'scatter_']:
            if formatted.startswith(prefix):
                formatted = formatted[len(prefix):]
                break
        
        # Special abbreviations that should stay uppercase
        upper_words = {'iod': 'IOD', 'rod': 'ROD', 'nnd': 'NND', 'pca': 'PCA'}
        words = formatted.split('_')
        formatted_words = []
        for w in words:
            if w.lower() in upper_words:
                formatted_words.append(upper_words[w.lower()])
            else:
                formatted_words.append(w.capitalize())
        
        return ' '.join(formatted_words)
    
    def _generate_descriptive_stats(self, film_summary: pd.DataFrame) -> str:
        """Generate detailed descriptive statistics."""
        from scipy import stats as sp_stats
        
        text = "<table style='border-collapse: collapse; width: 100%;'>"
        text += "<tr style='background-color: #3D3D4D;'>"
        text += "<th style='padding: 8px; text-align: left;'>Metric</th>"
        text += "<th style='padding: 8px;'>Mean</th>"
        text += "<th style='padding: 8px;'>SD</th>"
        text += "<th style='padding: 8px;'>Median</th>"
        text += "<th style='padding: 8px;'>IQR</th>"
        text += "<th style='padding: 8px;'>95% CI</th>"
        text += "<th style='padding: 8px;'>Normal?</th>"
        text += "</tr>"
        
        metrics = [
            ('ROD Fraction', film_summary['rod_fraction'] * 100, '%'),
            ('Total Deposits', film_summary['n_total'], ''),
            ('Normal Count', film_summary['n_normal'], ''),
            ('ROD Count', film_summary['n_rod'], ''),
        ]
        
        if 'total_iod' in film_summary.columns:
            metrics.append(('Total IOD', film_summary['total_iod'], ''))
        
        for name, data, unit in metrics:
            data = data.dropna()
            if len(data) < 2:
                continue
            
            mean = data.mean()
            std = data.std()
            median = data.median()
            q1, q3 = data.quantile([0.25, 0.75])
            iqr = q3 - q1
            sem = std / np.sqrt(len(data))
            ci_low = mean - 1.96 * sem
            ci_high = mean + 1.96 * sem
            
            # Normality test
            if len(data) >= 3:
                _, p_norm = sp_stats.shapiro(data[:5000])
                is_normal = "✓" if p_norm > 0.05 else "✗"
            else:
                is_normal = "-"
            
            text += f"<tr>"
            text += f"<td style='padding: 6px;'><b>{name}</b></td>"
            text += f"<td style='padding: 6px; text-align: center;'>{mean:.2f}{unit}</td>"
            text += f"<td style='padding: 6px; text-align: center;'>{std:.2f}</td>"
            text += f"<td style='padding: 6px; text-align: center;'>{median:.2f}</td>"
            text += f"<td style='padding: 6px; text-align: center;'>{iqr:.2f}</td>"
            text += f"<td style='padding: 6px; text-align: center;'>[{ci_low:.2f}, {ci_high:.2f}]</td>"
            text += f"<td style='padding: 6px; text-align: center;'>{is_normal}</td>"
            text += "</tr>"
        
        text += "</table>"
        text += f"<p style='color: #B0B0B0; font-size: 11px;'>n = {len(film_summary)} films | CI = Confidence Interval | Normal? = Shapiro-Wilk p > 0.05</p>"
        
        return text
    
    def _generate_comparison_stats(self, stats_results: dict) -> str:
        """Generate group comparison statistics."""
        text = ""
        
        for metric, result in stats_results.items():
            if 'error' in result:
                text += f"<p><b>{metric}:</b> {result['error']}</p>"
                continue
            
            text += f"<h4 style='color: {Theme.PRIMARY_LIGHT};'>{metric}</h4>"
            
            if 'overall_test' in result:
                # Multiple group comparison
                text += f"<p><b>Test:</b> {result['overall_test']}</p>"
                text += f"<p><b>Statistic:</b> {result['overall_statistic']:.3f}</p>"
                p = result['overall_p_value']
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                text += f"<p><b>p-value:</b> {p:.4f} {sig}</p>"
                
                if result.get('pairwise_comparisons'):
                    text += "<p><b>Pairwise:</b></p><ul>"
                    for pair in result['pairwise_comparisons']:
                        p_corr = pair.get('p_value_corrected', pair['p_value'])
                        sig = "***" if p_corr < 0.001 else "**" if p_corr < 0.01 else "*" if p_corr < 0.05 else ""
                        text += f"<li>{pair['group1_name']} vs {pair['group2_name']}: "
                        text += f"p={p_corr:.4f}{sig}, d={pair['cohens_d']:.2f} ({pair['effect_size']})</li>"
                    text += "</ul>"
            else:
                # Two group comparison
                text += f"<p><b>Groups:</b> {result['group1_name']} (n={result['n1']}) vs {result['group2_name']} (n={result['n2']})</p>"
                text += f"<p><b>Means:</b> {result['mean1']:.3f} ± {result['std1']:.3f} vs {result['mean2']:.3f} ± {result['std2']:.3f}</p>"
                text += f"<p><b>Test:</b> {result['test_name']}</p>"
                
                p = result['p_value']
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                color = Theme.ROD if p < 0.05 else Theme.TEXT_SECONDARY
                text += f"<p><b>p-value:</b> <span style='color: {color};'>{p:.4f} {sig}</span></p>"
                text += f"<p><b>Effect size:</b> Cohen's d = {result['cohens_d']:.2f} ({result['effect_size']})</p>"
        
        text += "<p style='color: #B0B0B0; font-size: 11px;'>* p < 0.05 | ** p < 0.01 | *** p < 0.001</p>"
        return text
    
    def _generate_spatial_stats(self, spatial_stats: dict) -> str:
        """Generate spatial analysis statistics."""
        text = "<table style='border-collapse: collapse;'>"
        
        if 'mean_nnd' in spatial_stats:
            text += f"<tr><td style='padding: 6px;'><b>Mean Nearest Neighbor Distance:</b></td>"
            text += f"<td style='padding: 6px;'>{spatial_stats['mean_nnd']:.1f} px</td></tr>"
        
        if 'mean_clark_evans' in spatial_stats:
            r = spatial_stats['mean_clark_evans']
            pattern = "clustered" if r < 1 else "regular" if r > 1 else "random"
            text += f"<tr><td style='padding: 6px;'><b>Clark-Evans R:</b></td>"
            text += f"<td style='padding: 6px;'>{r:.3f} ({pattern})</td></tr>"
        
        if 'density_per_mm2' in spatial_stats:
            text += f"<tr><td style='padding: 6px;'><b>Deposit Density:</b></td>"
            text += f"<td style='padding: 6px;'>{spatial_stats['density_per_mm2']:.2f} /mm²</td></tr>"
        
        text += "</table>"
        text += "<p style='color: #B0B0B0; font-size: 11px;'>Clark-Evans R: &lt;1 clustered, =1 random, &gt;1 regular</p>"
        return text
    
    def _show_image_dialog(self, path: str, title: str):
        dialog = ImageViewerDialog(path, title, self)
        dialog.exec()
    
    def _on_table_double_click(self, index):
        if not self.results:
            return
        
        row = index.row()
        filename = self.summary_table.item(row, 0).text()
        output_dir = Path(self.results['output_dir'])
        
        # Open LabelingWindow in EDIT_MODE
        if 'deposit_data' in self.results and self.results['deposit_data'] is not None:
            # Find original image
            image_path = None
            stem = Path(filename).stem
            
            # Try input_dir from config
            input_dir = config.get("last_input_dir", "")
            if input_dir:
                for ext in ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.TIF', '.TIFF']:
                    candidate = Path(input_dir) / f"{stem}{ext}"
                    if candidate.exists():
                        image_path = str(candidate)
                        break
            
            # Fallback to annotated image
            if not image_path:
                annotated_path = output_dir / 'annotated' / f"{stem}_annotated.png"
                if annotated_path.exists():
                    image_path = str(annotated_path)
            
            if not image_path:
                QMessageBox.warning(self, "Not Found", f"Original image not found for {filename}")
                return
            
            # Get deposits for this file
            deposits_df = self.results['deposit_data']
            file_deposits = deposits_df[deposits_df['filename'] == filename].copy()
            
            # Load contour data from JSON
            contour_data = {}
            next_group_id = 1
            json_path = output_dir / 'deposits' / f"{stem}.labels.json"
            if not json_path.exists():
                json_path = output_dir / 'deposits' / f"{stem}_deposits.json"
            
            if json_path.exists():
                import json
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        next_group_id = data.get('next_group_id', 1)
                        for dep in data.get('deposits', []):
                            dep_id = dep.get('id', len(contour_data))
                            contour_data[dep_id] = {
                                'contour': dep.get('contour', []),
                                'merged': dep.get('merged', False),
                                'group_id': dep.get('group_id', None)
                            }
                except Exception:
                    pass
            
            # Open LabelingWindow in EDIT_MODE
            from .labeling_gui import LabelingWindow
            
            edit_data = {
                'image_path': image_path,
                'output_dir': str(output_dir),
                'filename': filename,
                'deposits_df': file_deposits,
                'contour_data': contour_data,
                'next_group_id': next_group_id
            }
            
            self._edit_window = LabelingWindow(
                mode=LabelingWindow.MODE_EDIT,
                edit_data=edit_data
            )
            self._edit_window.data_saved.connect(self._reload_results)
            self._edit_window.show()
        else:
            # Fallback to simple viewer
            annotated_path = output_dir / 'annotated' / f"{Path(filename).stem}_annotated.png"
            if annotated_path.exists():
                dialog = ImageViewerDialog(str(annotated_path), filename, self)
                dialog.exec()
            else:
                QMessageBox.warning(self, "Not Found", f"Annotated image not found")
    
    def _reload_results(self):
        """Reload results after editing."""
        if not self.results or 'output_dir' not in self.results:
            return
        
        output_dir = Path(self.results['output_dir'])
        
        # Reload film_summary
        summary_path = output_dir / 'film_summary.csv'
        if summary_path.exists():
            self.results['film_summary'] = pd.read_csv(summary_path)
        
        # Reload deposit_data
        all_deposits_path = output_dir / 'all_deposits.csv'
        if all_deposits_path.exists():
            self.results['deposit_data'] = pd.read_csv(all_deposits_path)
        
        # Refresh display
        self.load_results(self.results)
    
    def _open_folder(self):
        if self.results and 'output_dir' in self.results:
            os.startfile(self.results['output_dir'])
    
    def _export_excel(self):
        if not self.results:
            return
        
        path, _ = QFileDialog.getSaveFileName(self, "Save Excel", "", "Excel (*.xlsx)")
        if path:
            try:
                with pd.ExcelWriter(path) as writer:
                    self.results['film_summary'].to_excel(writer, sheet_name='Film Summary', index=False)
                QMessageBox.information(self, "Success", f"Saved to {path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))
    
    def _open_report(self):
        if self.results and 'output_dir' in self.results:
            report_path = Path(self.results['output_dir']) / 'report.html'
            if report_path.exists():
                import webbrowser
                webbrowser.open(str(report_path))


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SCAT - Spot Classification and Analysis Tool")
        self.setMinimumSize(900, 700)
        
        # Set icon
        icon_path = get_icon_path()
        if icon_path:
            self.setWindowIcon(QIcon(icon_path))
        
        self._setup_ui()
        self._setup_shortcuts()
        self._load_window_state()
    
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Header with settings button
        header_layout = QHBoxLayout()
        
        header = QLabel("SCAT")
        header.setFont(QFont("Arial", 24, QFont.Bold))
        header.setStyleSheet("color: #1a237e; padding: 10px;")
        header_layout.addWidget(header)
        
        header_layout.addStretch()
        
        settings_btn = QPushButton("⚙ Settings")
        settings_btn.clicked.connect(self._open_settings)
        header_layout.addWidget(settings_btn)
        
        layout.addLayout(header_layout)
        
        subtitle = QLabel("Spot Classification and Analysis Tool for Drosophila Excreta")
        subtitle.setStyleSheet("color: #666; padding-left: 10px;")
        layout.addWidget(subtitle)
        
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # Labeling tab
        self.labeling_widget = QWidget()
        labeling_layout = QVBoxLayout(self.labeling_widget)
        launch_labeling = QPushButton("Launch Labeling Window")
        launch_labeling.setMinimumHeight(100)
        launch_labeling.clicked.connect(self._launch_labeling)
        labeling_layout.addWidget(launch_labeling)
        labeling_layout.addStretch()
        self.tabs.addTab(self.labeling_widget, "Labeling")
        
        # Training tab
        self.training_tab = TrainingTab()
        self.tabs.addTab(self.training_tab, "Training")
        
        # Analysis tab
        self.analysis_tab = AnalysisTab()
        self.analysis_tab.analysis_complete.connect(self._on_analysis_complete)
        self.tabs.addTab(self.analysis_tab, "Analysis")
        
        # Results tab
        self.results_tab = ResultsTab()
        self.tabs.addTab(self.results_tab, "Results")
        
        self.statusBar().showMessage("Ready")
    
    def _setup_shortcuts(self):
        # Global shortcuts
        QShortcut(QKeySequence(config.get_shortcut("quit")), self, self.close)
        QShortcut(QKeySequence(config.get_shortcut("run_analysis")), self, self._run_analysis_shortcut)
    
    def _run_analysis_shortcut(self):
        self.tabs.setCurrentWidget(self.analysis_tab)
        self.analysis_tab._run_analysis()
    
    def _open_settings(self):
        dialog = SettingsDialog(self)
        if dialog.exec():
            # Shortcuts changed - would need restart to take effect
            QMessageBox.information(
                self, "Settings Saved",
                "Settings saved. Some shortcut changes may require restart."
            )
    
    def _launch_labeling(self):
        from .labeling_gui import LabelingWindow
        self.labeling_window = LabelingWindow()
        icon_path = get_icon_path()
        if icon_path:
            self.labeling_window.setWindowIcon(QIcon(icon_path))
        self.labeling_window.show()
    
    def _on_analysis_complete(self, results):
        self.results_tab.load_results(results)
        self.tabs.setCurrentWidget(self.results_tab)
    
    def _load_window_state(self):
        w = config.get("window.width", 1200)
        h = config.get("window.height", 800)
        self.resize(w, h)
        if config.get("window.maximized", False):
            self.showMaximized()
    
    def _save_window_state(self):
        config.set("window.width", self.width(), auto_save=False)
        config.set("window.height", self.height(), auto_save=False)
        config.set("window.maximized", self.isMaximized())
    
    def closeEvent(self, event):
        self._save_window_state()
        event.accept()


def _load_custom_fonts():
    """Load custom fonts bundled with the application."""
    fonts_dir = Path(__file__).parent / "resources" / "fonts"
    
    if fonts_dir.exists():
        for font_file in fonts_dir.glob("*.ttf"):
            font_id = QFontDatabase.addApplicationFont(str(font_file))
            if font_id < 0:
                print(f"Warning: Failed to load font {font_file.name}")


def run_gui():
    """Launch the main GUI application."""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Load custom fonts (Noto Sans)
    _load_custom_fonts()
    
    # Set application-wide font
    app_font = QFont("Noto Sans", 10)
    if not QFontDatabase.hasFamily("Noto Sans"):
        # Fallback if Noto Sans not available
        app_font = QFont("Segoe UI", 10)  # Windows fallback
    app.setFont(app_font)
    
    # Apply dark theme
    app.setStyleSheet(Theme.get_app_stylesheet())
    
    icon_path = get_icon_path()
    if icon_path:
        app.setWindowIcon(QIcon(icon_path))
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    run_gui()
