"""
Labeling GUI for deposit annotation with manual editing.
"""

import sys
import json
from pathlib import Path
from typing import List, Optional
import numpy as np
import cv2

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QGraphicsView, QGraphicsScene, 
    QGraphicsPathItem, QGraphicsRectItem, QToolBar, QSplitter, QGroupBox,
    QSpinBox, QDoubleSpinBox, QFormLayout, QTableWidget,
    QTableWidgetItem, QHeaderView, QButtonGroup, QRadioButton, QScrollArea
)
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import (
    QImage, QPixmap, QPainter, QPen, QColor, QBrush, 
    QPainterPath, QAction, QShortcut, QKeySequence, QWheelEvent
)

from .detector import DepositDetector, Deposit
from .features import FeatureExtractor
from .config import config


# =============================================================================
# Theme - Dark mode colors (matching main_gui.py)
# =============================================================================
class Theme:
    """Dark theme colors - DIC microscopy palette."""
    PRIMARY = "#DA4E42"      # DIC2497 - Main accent (red-orange)
    SECONDARY = "#636867"    # DIC540 - Secondary (gray-green)
    
    BG_DARKEST = "#0A0A0A"   # Main background
    BG_DARK = "#121212"      # Card/panel background
    BG_MEDIUM = "#1A1A1A"    # Input fields
    BG_LIGHT = "#242424"     # Buttons, hover states
    
    TEXT_PRIMARY = "#E0E0E0"
    TEXT_SECONDARY = "#A0A0A0"
    TEXT_MUTED = "#666666"
    
    BORDER = "#2A2A2A"
    
    # Deposit colors
    NORMAL = "#4CAF50"
    NORMAL_DARK = "#388E3C"
    ROD = "#F44336"
    ROD_DARK = "#D32F2F"
    ARTIFACT = "#9E9E9E"
    ARTIFACT_DARK = "#757575"
    
    @classmethod
    def get_stylesheet(cls) -> str:
        return f"""
            QMainWindow, QWidget {{
                background-color: {cls.BG_DARKEST};
                color: {cls.TEXT_PRIMARY};
            }}
            QGroupBox {{
                font-weight: bold;
                border: 1px solid {cls.BORDER};
                border-radius: 6px;
                margin-top: 16px;
                padding-top: 16px;
                background-color: {cls.BG_DARK};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px;
                color: {cls.PRIMARY};
            }}
            QPushButton {{
                background-color: {cls.BG_LIGHT};
                border: 1px solid {cls.BORDER};
                border-radius: 4px;
                padding: 6px 12px;
                min-height: 24px;
                color: {cls.TEXT_PRIMARY};
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {cls.SECONDARY};
            }}
            QPushButton:pressed {{
                background-color: {cls.PRIMARY};
            }}
            QSpinBox, QDoubleSpinBox {{
                background-color: {cls.BG_MEDIUM};
                border: 1px solid {cls.BORDER};
                border-radius: 4px;
                padding: 4px 8px;
                min-height: 20px;
                color: {cls.TEXT_PRIMARY};
            }}
            QSpinBox:focus, QDoubleSpinBox:focus {{
                border-color: {cls.PRIMARY};
            }}
            QTableWidget {{
                background-color: {cls.BG_DARK};
                border: 1px solid {cls.BORDER};
                gridline-color: {cls.BORDER};
                color: {cls.TEXT_PRIMARY};
            }}
            QTableWidget::item:selected {{
                background-color: {cls.SECONDARY};
            }}
            QHeaderView::section {{
                background-color: {cls.BG_MEDIUM};
                color: {cls.TEXT_PRIMARY};
                padding: 6px;
                border: none;
                border-bottom: 1px solid {cls.BORDER};
                font-weight: bold;
            }}
            QRadioButton {{
                color: {cls.TEXT_PRIMARY};
                spacing: 8px;
            }}
            QRadioButton::indicator {{
                width: 16px;
                height: 16px;
            }}
            QLabel {{
                color: {cls.TEXT_PRIMARY};
                font-weight: bold;
            }}
            QScrollArea {{
                border: none;
                background-color: transparent;
            }}
            QToolBar {{
                background-color: {cls.BG_DARK};
                border-bottom: 1px solid {cls.BORDER};
                spacing: 4px;
                padding: 4px;
            }}
            QSplitter::handle {{
                background-color: {cls.BORDER};
            }}
        """


# =============================================================================
# Custom Widgets - SpinBox without scroll wheel
# =============================================================================
class NoScrollSpinBox(QSpinBox):
    """SpinBox that ignores wheel events to prevent accidental value changes."""
    def wheelEvent(self, event: QWheelEvent):
        event.ignore()


class NoScrollDoubleSpinBox(QDoubleSpinBox):
    """DoubleSpinBox that ignores wheel events."""
    def wheelEvent(self, event: QWheelEvent):
        event.ignore()


class DepositGraphicsItem(QGraphicsPathItem):
    COLORS = {
        'rod': QColor(255, 0, 0, 150),
        'normal': QColor(0, 255, 0, 150),
        'artifact': QColor(128, 128, 128, 150),
        'unknown': QColor(255, 255, 0, 150)
    }
    
    # Colors for group visualization
    GROUP_COLORS = [
        QColor(255, 165, 0),    # Orange
        QColor(0, 255, 255),    # Cyan
        QColor(255, 0, 255),    # Magenta
        QColor(255, 255, 0),    # Yellow
        QColor(0, 128, 255),    # Light blue
        QColor(255, 128, 0),    # Dark orange
        QColor(128, 0, 255),    # Purple
        QColor(0, 255, 128),    # Mint
    ]
    
    def __init__(self, deposit: Deposit, scale: float = 1.0):
        super().__init__()
        self.deposit = deposit
        self.scale = scale
        self.selected = False
        self._create_path()
        self._update_appearance()
    
    def _create_path(self):
        path = QPainterPath()
        contour = self.deposit.contour.squeeze()
        if len(contour.shape) == 1:
            contour = contour.reshape(1, 2)
        path.moveTo(contour[0][0] * self.scale, contour[0][1] * self.scale)
        for point in contour[1:]:
            path.lineTo(point[0] * self.scale, point[1] * self.scale)
        path.closeSubpath()
        self.setPath(path)
    
    def _update_appearance(self):
        color = self.COLORS.get(self.deposit.label, self.COLORS['unknown'])
        
        # If grouped, use dashed line with group color
        if self.deposit.group_id is not None:
            group_color = self.GROUP_COLORS[self.deposit.group_id % len(self.GROUP_COLORS)]
            pen = QPen(group_color, 4 if self.selected else 3)
            pen.setStyle(Qt.DashLine)
            self.setPen(pen)
        else:
            self.setPen(QPen(color.darker(150), 3 if self.selected else 2))
        
        fill = QColor(color)
        fill.setAlpha(100 if self.selected else 50)
        self.setBrush(QBrush(fill))
    
    def set_label(self, label: str):
        self.deposit.label = label
        self._update_appearance()
    
    def set_selected(self, selected: bool):
        self.selected = selected
        self._update_appearance()
    
    def update_group_visual(self, group_id):
        """Update visual appearance for grouping."""
        self.deposit.group_id = group_id
        self._update_appearance()


class ImageViewer(QGraphicsView):
    MODE_SELECT = 0
    MODE_ADD = 1
    MODE_MERGE = 2
    
    def __init__(self, parent=None):
        super().__init__()
        self.parent_window = parent
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        
        self.pixmap_item = None
        self.deposit_items: List[DepositGraphicsItem] = []
        self.selected_item: Optional[DepositGraphicsItem] = None
        self.selected_items: List[DepositGraphicsItem] = []
        self.scale_factor = 1.0
        
        self.edit_mode = self.MODE_SELECT
        self.drawing = False
        self.start_point = None
        self.rect_item = None
    
    def set_mode(self, mode: int):
        self.edit_mode = mode
        self.selected_items.clear()
        if mode == self.MODE_SELECT:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
        else:
            self.setDragMode(QGraphicsView.NoDrag)
    
    def load_image(self, image: np.ndarray):
        self.scene.clear()
        self.deposit_items.clear()
        self.selected_item = None
        self.selected_items.clear()
        h, w, ch = image.shape
        qimg = QImage(image.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.pixmap_item = self.scene.addPixmap(pixmap)
        self.setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
    
    def add_deposit(self, deposit: Deposit):
        item = DepositGraphicsItem(deposit, self.scale_factor)
        item.setFlag(QGraphicsPathItem.ItemIsSelectable)
        self.scene.addItem(item)
        self.deposit_items.append(item)
    
    def remove_deposit_item(self, item: DepositGraphicsItem):
        if item in self.deposit_items:
            self.deposit_items.remove(item)
            self.scene.removeItem(item)
            if self.selected_item == item:
                self.selected_item = None
            if item in self.selected_items:
                self.selected_items.remove(item)
    
    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)
    
    def mousePressEvent(self, event):
        if self.edit_mode == self.MODE_SELECT:
            item = self.itemAt(event.pos())
            
            # Ctrl+클릭: 다중 선택
            if event.modifiers() & Qt.ControlModifier:
                if isinstance(item, DepositGraphicsItem):
                    if item in self.selected_items:
                        item.set_selected(False)
                        self.selected_items.remove(item)
                    else:
                        item.set_selected(True)
                        self.selected_items.append(item)
                    # 선택 소스를 이미지로 설정
                    if self.parent_window:
                        self.parent_window._selection_source = 'image'
            else:
                # 일반 클릭: 단일 선택
                # 기존 다중 선택 해제
                for sel_item in self.selected_items:
                    sel_item.set_selected(False)
                self.selected_items.clear()
                
                if isinstance(item, DepositGraphicsItem):
                    if self.selected_item and self.selected_item != item:
                        self.selected_item.set_selected(False)
                    self.selected_item = item
                    item.set_selected(True)
                    # 선택 소스를 이미지로 설정
                    if self.parent_window:
                        self.parent_window._selection_source = 'image'
                else:
                    # 빈 곳 클릭: 선택 해제
                    if self.selected_item:
                        self.selected_item.set_selected(False)
                        self.selected_item = None
            
            super().mousePressEvent(event)
            
        elif self.edit_mode == self.MODE_ADD:
            self.drawing = True
            self.start_point = self.mapToScene(event.pos())
            self.rect_item = QGraphicsRectItem()
            self.rect_item.setPen(QPen(QColor(0, 0, 255), 2, Qt.DashLine))
            self.scene.addItem(self.rect_item)
        elif self.edit_mode == self.MODE_MERGE:
            item = self.itemAt(event.pos())
            if isinstance(item, DepositGraphicsItem):
                if item in self.selected_items:
                    item.set_selected(False)
                    self.selected_items.remove(item)
                else:
                    item.set_selected(True)
                    self.selected_items.append(item)
    
    def mouseMoveEvent(self, event):
        if self.edit_mode == self.MODE_ADD and self.drawing:
            current = self.mapToScene(event.pos())
            rect = QRectF(self.start_point, current).normalized()
            self.rect_item.setRect(rect)
        else:
            super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        if self.edit_mode == self.MODE_ADD and self.drawing:
            self.drawing = False
            if self.rect_item:
                rect = self.rect_item.rect()
                self.scene.removeItem(self.rect_item)
                self.rect_item = None
                if rect.width() > 5 and rect.height() > 5 and self.parent_window:
                    self.parent_window._add_deposit_from_rect(rect)
        else:
            super().mouseReleaseEvent(event)


class LabelingWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SCAT - Labeling Tool")
        self.setMinimumSize(1200, 800)
        
        # Apply dark theme
        self.setStyleSheet(Theme.get_stylesheet())
        
        self.image: Optional[np.ndarray] = None
        self.deposits: List[Deposit] = []
        self.current_file: Optional[Path] = None
        # Use sensitive_mode=True to match Analysis tab detection
        self.detector = DepositDetector(sensitive_mode=True)
        self.extractor = FeatureExtractor()
        self.next_id = 0
        self.next_group_id = 1  # For grouping deposits
        
        # Selection source: 'image' or 'table'
        # Used to determine if auto-advance should happen
        self._selection_source = 'image'
        
        self._setup_ui()
        self._setup_shortcuts()
    
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        self.viewer = ImageViewer(parent=self)
        splitter.addWidget(self.viewer)
        
        # Right panel with scroll area
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        right_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        right_scroll.setMinimumWidth(280)
        right_scroll.setMaximumWidth(350)
        
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(8)
        right_layout.setContentsMargins(8, 8, 8, 8)
        
        right_scroll.setWidget(right_panel)
        splitter.addWidget(right_scroll)
        
        # Edit Mode
        edit_group = QGroupBox("Edit Mode")
        edit_layout = QVBoxLayout()
        
        self.mode_group = QButtonGroup()
        self.radio_select = QRadioButton("Select (S)")
        self.radio_select.setChecked(True)
        self.radio_add = QRadioButton("Add Deposit (A)")
        self.radio_merge = QRadioButton("Merge Mode (M)")
        
        self.mode_group.addButton(self.radio_select, ImageViewer.MODE_SELECT)
        self.mode_group.addButton(self.radio_add, ImageViewer.MODE_ADD)
        self.mode_group.addButton(self.radio_merge, ImageViewer.MODE_MERGE)
        self.mode_group.buttonClicked.connect(self._on_mode_changed)
        
        edit_layout.addWidget(self.radio_select)
        edit_layout.addWidget(self.radio_add)
        edit_layout.addWidget(self.radio_merge)
        
        edit_btn_layout = QHBoxLayout()
        self.btn_delete = QPushButton("Delete (D)")
        self.btn_delete.clicked.connect(self._delete_selected)
        edit_btn_layout.addWidget(self.btn_delete)
        
        self.btn_merge = QPushButton("Merge Selected")
        self.btn_merge.clicked.connect(self._merge_selected)
        edit_btn_layout.addWidget(self.btn_merge)
        
        edit_layout.addLayout(edit_btn_layout)
        
        # Group buttons (separate row)
        group_btn_layout = QHBoxLayout()
        self.btn_group = QPushButton("Group (G)")
        self.btn_group.setToolTip("Group selected deposits (keeps individual contours)")
        self.btn_group.clicked.connect(self._group_selected)
        group_btn_layout.addWidget(self.btn_group)
        
        self.btn_ungroup = QPushButton("Ungroup (U)")
        self.btn_ungroup.setToolTip("Remove selected deposits from their group")
        self.btn_ungroup.clicked.connect(self._ungroup_selected)
        group_btn_layout.addWidget(self.btn_ungroup)
        
        edit_layout.addLayout(group_btn_layout)
        
        edit_group.setLayout(edit_layout)
        right_layout.addWidget(edit_group)
        
        # Detection settings
        detect_group = QGroupBox("Detection Settings")
        detect_layout = QFormLayout()
        
        self.min_area_spin = NoScrollSpinBox()
        self.min_area_spin.setRange(1, 1000)
        self.min_area_spin.setValue(20)
        detect_layout.addRow("Min Area:", self.min_area_spin)
        
        self.max_area_spin = NoScrollSpinBox()
        self.max_area_spin.setRange(100, 50000)
        self.max_area_spin.setValue(10000)
        detect_layout.addRow("Max Area:", self.max_area_spin)
        
        self.threshold_spin = NoScrollDoubleSpinBox()
        self.threshold_spin.setRange(0.1, 1.0)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(0.6)
        detect_layout.addRow("ROD Threshold:", self.threshold_spin)
        
        detect_btn = QPushButton("Re-detect")
        detect_btn.clicked.connect(self._detect_deposits)
        detect_layout.addRow(detect_btn)
        detect_group.setLayout(detect_layout)
        right_layout.addWidget(detect_group)
        
        # Labeling
        label_group = QGroupBox("Labeling (1=Normal, 2=ROD, 3=Artifact)")
        label_layout = QVBoxLayout()
        
        btn_layout = QHBoxLayout()
        self.btn_normal = QPushButton("Normal (1)")
        self.btn_normal.setStyleSheet(f"background-color: {Theme.NORMAL}; color: white;")
        self.btn_normal.clicked.connect(lambda: self._set_selected_label("normal"))
        btn_layout.addWidget(self.btn_normal)
        
        self.btn_rod = QPushButton("ROD (2)")
        self.btn_rod.setStyleSheet(f"background-color: {Theme.ROD}; color: white;")
        self.btn_rod.clicked.connect(lambda: self._set_selected_label("rod"))
        btn_layout.addWidget(self.btn_rod)
        
        self.btn_artifact = QPushButton("Artifact (3)")
        self.btn_artifact.setStyleSheet(f"background-color: {Theme.ARTIFACT}; color: white;")
        self.btn_artifact.clicked.connect(lambda: self._set_selected_label("artifact"))
        btn_layout.addWidget(self.btn_artifact)
        label_layout.addLayout(btn_layout)
        
        auto_threshold = QPushButton("Auto (Threshold)")
        auto_threshold.clicked.connect(self._auto_label_threshold)
        label_layout.addWidget(auto_threshold)
        label_group.setLayout(label_layout)
        right_layout.addWidget(label_group)
        
        # Statistics
        stats_group = QGroupBox("Statistics")
        stats_layout = QFormLayout()
        self.label_total = QLabel("0")
        self.label_normal = QLabel("0")
        self.label_rod = QLabel("0")
        self.label_artifact = QLabel("0")
        self.label_unknown = QLabel("0")
        stats_layout.addRow("Total:", self.label_total)
        stats_layout.addRow("Normal:", self.label_normal)
        stats_layout.addRow("ROD:", self.label_rod)
        stats_layout.addRow("Artifact:", self.label_artifact)
        stats_layout.addRow("Unlabeled:", self.label_unknown)
        stats_group.setLayout(stats_layout)
        right_layout.addWidget(stats_group)
        
        # Table
        self.deposit_table = QTableWidget()
        self.deposit_table.setColumnCount(5)
        self.deposit_table.setHorizontalHeaderLabels(["ID", "Area", "Circ", "Hue", "Label"])
        self.deposit_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.deposit_table.itemSelectionChanged.connect(self._on_table_select)
        right_layout.addWidget(self.deposit_table)
        
        right_layout.addStretch()
        
        # Toolbar
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        
        open_action = QAction("Open Image", self)
        open_action.triggered.connect(self._open_image)
        toolbar.addAction(open_action)
        
        save_action = QAction("Save Labels", self)
        save_action.triggered.connect(self._save_labels)
        toolbar.addAction(save_action)
        
        load_action = QAction("Load Labels", self)
        load_action.triggered.connect(self._load_labels)
        toolbar.addAction(load_action)
        
        export_action = QAction("Export Training Data", self)
        export_action.triggered.connect(self._export_training_data)
        toolbar.addAction(export_action)
        
        self.statusBar().showMessage("S: Select, A: Add, Ctrl+Click: Multi-select, M: Merge, D: Delete")
        splitter.setSizes([800, 400])
    
    def _setup_shortcuts(self):
        QShortcut(QKeySequence("1"), self, lambda: self._set_selected_label("normal"))
        QShortcut(QKeySequence("2"), self, lambda: self._set_selected_label("rod"))
        QShortcut(QKeySequence("3"), self, lambda: self._set_selected_label("artifact"))
        QShortcut(QKeySequence("N"), self, self._select_next)
        QShortcut(QKeySequence("P"), self, self._select_prev)
        QShortcut(QKeySequence("Ctrl+S"), self, self._save_labels)
        QShortcut(QKeySequence("D"), self, self._delete_selected)
        QShortcut(QKeySequence("Delete"), self, self._delete_selected)
        QShortcut(QKeySequence("S"), self, lambda: self._set_mode(0))
        QShortcut(QKeySequence("A"), self, lambda: self._set_mode(1))
        # M: merge 실행 (Ctrl+클릭으로 다중 선택 후)
        QShortcut(QKeySequence("M"), self, self._merge_selected)
        # G: group selected, U: ungroup
        QShortcut(QKeySequence("G"), self, self._group_selected)
        QShortcut(QKeySequence("U"), self, self._ungroup_selected)
    
    def _set_mode(self, mode):
        if mode == 0:
            self.radio_select.setChecked(True)
        elif mode == 1:
            self.radio_add.setChecked(True)
        elif mode == 2:
            self.radio_merge.setChecked(True)
        self._on_mode_changed()
    
    def _on_mode_changed(self):
        mode = self.mode_group.checkedId()
        self.viewer.set_mode(mode)
        modes = {0: "Select", 1: "Add (drag rectangle)", 2: "Merge (click deposits)"}
        self.statusBar().showMessage(f"Mode: {modes.get(mode, 'Unknown')}")
    
    def _open_image(self):
        # Use last image directory
        start_dir = config.get("last_image_dir", "")
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", start_dir, 
            "Images (*.tif *.tiff *.png *.jpg)"
        )
        if path:
            # Save directory for next time
            config.set("last_image_dir", str(Path(path).parent))
            self._load_image(Path(path))
    
    def _load_image(self, path: Path):
        from PIL import Image
        self.current_file = path
        self.image = np.array(Image.open(path))
        img = Image.open(path)
        dpi = img.info.get('dpi', (600, 600))[0]
        self.extractor = FeatureExtractor(dpi=dpi)
        self.viewer.load_image(self.image)
        self._detect_deposits()
        self.setWindowTitle(f"SCAT - {path.name}")
    
    def _detect_deposits(self):
        if self.image is None:
            return
        self.detector.min_area = self.min_area_spin.value()
        self.detector.max_area = self.max_area_spin.value()
        self.detector.sensitive_mode = True  # Always match Analysis tab
        self.deposits = self.detector.detect(self.image)
        self.deposits = self.extractor.extract_features(self.image, self.deposits)
        self.next_id = len(self.deposits)
        
        for item in self.viewer.deposit_items:
            self.viewer.scene.removeItem(item)
        self.viewer.deposit_items.clear()
        self.viewer.selected_item = None
        self.viewer.selected_items.clear()
        
        for deposit in self.deposits:
            self.viewer.add_deposit(deposit)
        
        self._update_table()
        self._update_stats()
        self.statusBar().showMessage(f"Detected {len(self.deposits)} deposits")
    
    def _add_deposit_from_rect(self, rect: QRectF):
        if self.image is None:
            return
        
        x, y = int(rect.x()), int(rect.y())
        w, h = int(rect.width()), int(rect.height())
        
        h_img, w_img = self.image.shape[:2]
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = min(w, w_img - x)
        h = min(h, h_img - y)
        
        contour = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
        
        roi = self.image[y:y+h, x:x+w]
        if len(roi) > 0:
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                contour = largest + np.array([x, y])
        
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        bx, by, bw, bh = cv2.boundingRect(contour)
        aspect_ratio = max(bw, bh) / min(bw, bh) if min(bw, bh) > 0 else 1
        
        M = cv2.moments(contour)
        cx = int(M["m10"] / M["m00"]) if M["m00"] > 0 else x + w // 2
        cy = int(M["m01"] / M["m00"]) if M["m00"] > 0 else y + h // 2
        
        deposit = Deposit(
            id=self.next_id, contour=contour,
            x=bx, y=by, width=bw, height=bh,
            area=area, perimeter=perimeter,
            circularity=circularity, aspect_ratio=aspect_ratio,
            centroid=(cx, cy)
        )
        self.next_id += 1
        
        self.deposits.append(deposit)
        self.extractor.extract_features(self.image, [deposit])
        self.viewer.add_deposit(deposit)
        
        self._update_table()
        self._update_stats()
        self.statusBar().showMessage(f"Added deposit {deposit.id}")
    
    def _delete_selected(self):
        deleted = []
        if self.viewer.selected_items:
            for item in self.viewer.selected_items[:]:
                self.deposits.remove(item.deposit)
                deleted.append(item.deposit.id)
                self.viewer.remove_deposit_item(item)
        elif self.viewer.selected_item:
            item = self.viewer.selected_item
            self.deposits.remove(item.deposit)
            deleted.append(item.deposit.id)
            self.viewer.remove_deposit_item(item)
        
        if deleted:
            self._update_table()
            self._update_stats()
            self.statusBar().showMessage(f"Deleted: {deleted}")
    
    def _merge_selected(self):
        if len(self.viewer.selected_items) < 2:
            self.statusBar().showMessage("Select 2+ deposits to merge")
            return
        
        all_points = []
        for item in self.viewer.selected_items:
            contour = item.deposit.contour.squeeze()
            if len(contour.shape) == 1:
                contour = contour.reshape(1, 2)
            all_points.extend(contour.tolist())
        
        hull = cv2.convexHull(np.array(all_points))
        
        area = cv2.contourArea(hull)
        perimeter = cv2.arcLength(hull, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        bx, by, bw, bh = cv2.boundingRect(hull)
        aspect_ratio = max(bw, bh) / min(bw, bh) if min(bw, bh) > 0 else 1
        
        M = cv2.moments(hull)
        cx = int(M["m10"] / M["m00"]) if M["m00"] > 0 else bx + bw // 2
        cy = int(M["m01"] / M["m00"]) if M["m00"] > 0 else by + bh // 2
        
        merged = Deposit(
            id=self.next_id, contour=hull,
            x=bx, y=by, width=bw, height=bh,
            area=area, perimeter=perimeter,
            circularity=circularity, aspect_ratio=aspect_ratio,
            centroid=(cx, cy),
            merged=True  # Mark as merged for U-Net training
        )
        self.next_id += 1
        
        for item in self.viewer.selected_items[:]:
            self.deposits.remove(item.deposit)
            self.viewer.remove_deposit_item(item)
        
        self.deposits.append(merged)
        self.extractor.extract_features(self.image, [merged])
        self.viewer.add_deposit(merged)
        
        self._update_table()
        self._update_stats()
        self.statusBar().showMessage(f"Merged into deposit {merged.id}")
    
    def _group_selected(self):
        """Group selected deposits together (keeps individual contours)."""
        if len(self.viewer.selected_items) < 2:
            self.statusBar().showMessage("Select 2+ deposits to group")
            return
        
        # Assign same group_id to all selected deposits
        group_id = self.next_group_id
        self.next_group_id += 1
        
        grouped_ids = []
        for item in self.viewer.selected_items:
            item.deposit.group_id = group_id
            grouped_ids.append(item.deposit.id)
            # Update visual
            item.update_group_visual(group_id)
        
        self._update_table()
        self.statusBar().showMessage(f"Grouped deposits {grouped_ids} into group {group_id}")
    
    def _ungroup_selected(self):
        """Remove selected deposits from their group."""
        ungrouped_ids = []
        
        items = self.viewer.selected_items if self.viewer.selected_items else (
            [self.viewer.selected_item] if self.viewer.selected_item else []
        )
        
        for item in items:
            if item and item.deposit.group_id is not None:
                item.deposit.group_id = None
                ungrouped_ids.append(item.deposit.id)
                item.update_group_visual(None)
        
        if ungrouped_ids:
            self._update_table()
            self.statusBar().showMessage(f"Ungrouped deposits {ungrouped_ids}")
        else:
            self.statusBar().showMessage("No grouped deposits selected")

    
    def _update_table(self):
        self.deposit_table.setRowCount(len(self.deposits))
        for i, d in enumerate(self.deposits):
            self.deposit_table.setItem(i, 0, QTableWidgetItem(str(d.id)))
            self.deposit_table.setItem(i, 1, QTableWidgetItem(f"{d.area:.0f}"))
            self.deposit_table.setItem(i, 2, QTableWidgetItem(f"{d.circularity:.3f}"))
            self.deposit_table.setItem(i, 3, QTableWidgetItem(f"{d.mean_hue:.1f}"))
            self.deposit_table.setItem(i, 4, QTableWidgetItem(d.label))
    
    def _update_stats(self):
        labels = [d.label for d in self.deposits]
        self.label_total.setText(str(len(labels)))
        self.label_normal.setText(str(labels.count("normal")))
        self.label_rod.setText(str(labels.count("rod")))
        self.label_artifact.setText(str(labels.count("artifact")))
        self.label_unknown.setText(str(labels.count("unknown")))
    
    def _set_selected_label(self, label: str):
        # 다중 선택된 경우 모두 라벨링
        if self.viewer.selected_items:
            for item in self.viewer.selected_items:
                item.set_label(label)
            self._update_table()
            self._update_stats()
            return
        
        # 단일 선택
        if self.viewer.selected_item:
            self.viewer.selected_item.set_label(label)
            self._update_table()
            self._update_stats()
            
            # 테이블에서 선택한 경우에만 다음으로 이동
            if self._selection_source == 'table':
                self._select_next()
    
    def _auto_label_threshold(self):
        threshold = self.threshold_spin.value()
        for item in self.viewer.deposit_items:
            d = item.deposit
            # ROD: elongated AND concentrated (low lightness)
            if d.circularity < threshold and d.mean_lightness < 0.80:
                label = "rod"
            else:
                label = "normal"
            item.set_label(label)
        self._update_table()
        self._update_stats()
    
    def _select_next(self):
        if not self.viewer.deposit_items:
            return
        current_idx = -1
        if self.viewer.selected_item:
            try:
                current_idx = self.viewer.deposit_items.index(self.viewer.selected_item)
            except ValueError:
                pass
        next_idx = (current_idx + 1) % len(self.viewer.deposit_items)
        if self.viewer.selected_item:
            self.viewer.selected_item.set_selected(False)
        self.viewer.selected_item = self.viewer.deposit_items[next_idx]
        self.viewer.selected_item.set_selected(True)
        self.viewer.centerOn(self.viewer.selected_item)
        self.deposit_table.selectRow(next_idx)
    
    def _select_prev(self):
        if not self.viewer.deposit_items:
            return
        current_idx = 0
        if self.viewer.selected_item:
            try:
                current_idx = self.viewer.deposit_items.index(self.viewer.selected_item)
            except ValueError:
                pass
        prev_idx = (current_idx - 1) % len(self.viewer.deposit_items)
        if self.viewer.selected_item:
            self.viewer.selected_item.set_selected(False)
        self.viewer.selected_item = self.viewer.deposit_items[prev_idx]
        self.viewer.selected_item.set_selected(True)
        self.viewer.centerOn(self.viewer.selected_item)
        self.deposit_table.selectRow(prev_idx)
    
    def _on_table_select(self):
        rows = self.deposit_table.selectionModel().selectedRows()
        if rows:
            idx = rows[0].row()
            if 0 <= idx < len(self.viewer.deposit_items):
                if self.viewer.selected_item:
                    self.viewer.selected_item.set_selected(False)
                self.viewer.selected_item = self.viewer.deposit_items[idx]
                self.viewer.selected_item.set_selected(True)
                self.viewer.centerOn(self.viewer.selected_item)
                # 테이블에서 선택했음을 표시
                self._selection_source = 'table'
    
    def _save_labels(self):
        if not self.current_file:
            return
        # Use last label directory, fallback to image directory
        start_dir = config.get("last_label_dir", "")
        if not start_dir:
            start_dir = str(self.current_file.parent)
        default_name = self.current_file.stem + '.labels.json'
        default_path = Path(start_dir) / default_name
        
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Labels", str(default_path), "JSON (*.json)"
        )
        if path:
            # Save directory for next time
            config.set("last_label_dir", str(Path(path).parent))
            
            data = {
                'image_file': str(self.current_file),
                'next_group_id': self.next_group_id,
                'deposits': []
            }
            for d in self.deposits:
                deposit_data = {
                    'id': d.id, 
                    'x': d.centroid[0], 
                    'y': d.centroid[1],
                    'area': d.area, 
                    'circularity': d.circularity, 
                    'label': d.label,
                    'contour': d.contour.tolist(),
                    'merged': d.merged,
                    'group_id': d.group_id
                }
                data['deposits'].append(deposit_data)
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            self.statusBar().showMessage(f"Saved to {path}")
    
    def _load_labels(self):
        # Use last label directory
        start_dir = config.get("last_label_dir", "")
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Labels", start_dir, "JSON (*.json)"
        )
        if path:
            # Save directory for next time
            config.set("last_label_dir", str(Path(path).parent))
            
            with open(path) as f:
                data = json.load(f)
            
            # Load next_group_id if present
            self.next_group_id = data.get('next_group_id', 1)
            
            has_contours = data['deposits'] and 'contour' in data['deposits'][0]
            
            if has_contours:
                for item in self.viewer.deposit_items:
                    self.viewer.scene.removeItem(item)
                self.viewer.deposit_items.clear()
                self.viewer.selected_item = None
                self.deposits.clear()
                
                for saved in data['deposits']:
                    contour = np.array(saved['contour'])
                    d = Deposit(
                        id=saved['id'], contour=contour,
                        x=0, y=0, width=0, height=0,
                        area=saved['area'], perimeter=cv2.arcLength(contour, True),
                        circularity=saved['circularity'], aspect_ratio=1.0,
                        centroid=(saved['x'], saved['y']),
                        merged=saved.get('merged', False),
                        group_id=saved.get('group_id', None)
                    )
                    d.label = saved['label']
                    self.deposits.append(d)
                    self.extractor.extract_features(self.image, [d])
                    self.viewer.add_deposit(d)
                
                self.next_id = max(d.id for d in self.deposits) + 1 if self.deposits else 0
                
                # Update next_group_id based on loaded deposits
                group_ids = [d.group_id for d in self.deposits if d.group_id is not None]
                if group_ids:
                    self.next_group_id = max(max(group_ids) + 1, self.next_group_id)
            else:
                for saved in data['deposits']:
                    for item in self.viewer.deposit_items:
                        d = item.deposit
                        if abs(d.centroid[0] - saved['x']) < 5 and abs(d.centroid[1] - saved['y']) < 5:
                            item.set_label(saved['label'])
                            break
            
            self._update_table()
            self._update_stats()
            self.statusBar().showMessage(f"Loaded from {path}")
    
    def _export_training_data(self):
        if not self.deposits:
            return
        folder = QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if not folder:
            return
        folder = Path(folder)
        for label in ['rod', 'normal', 'artifact']:
            (folder / label).mkdir(exist_ok=True)
        from PIL import Image
        count = 0
        for d in self.deposits:
            if d.label in ['rod', 'normal', 'artifact']:
                patch = d.get_patch(self.image, padding=10)
                filename = f"{self.current_file.stem}_{d.id}.png"
                Image.fromarray(patch).save(folder / d.label / filename)
                count += 1
        self.statusBar().showMessage(f"Exported {count} patches to {folder}")


def run_labeling_gui():
    app = QApplication(sys.argv)
    window = LabelingWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_labeling_gui()
