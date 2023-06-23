from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QAction
from PySide6.QtMultimedia import QMediaDevices
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QGroupBox,
                               QHBoxLayout, QLineEdit, QPushButton, QSlider,
                               QSpinBox, QStackedWidget, QVBoxLayout, QWidget)
from superqt import QDoubleRangeSlider

from ui import DoubleSpinBox


class SideBar(QWidget):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._parent = self.parent()
    self.model_dir = Path("models")
    self.model_dir.mkdir(exist_ok=True)
    self.create_input_panel()
    self.create_model_panel()
    self.create_roi_panel()

    layout = QVBoxLayout()
    layout.addWidget(self.input_gb)
    layout.addWidget(self.model_gb)
    layout.addWidget(self.roi_gb)
    layout.addStretch(1)
    self.setLayout(layout)

    self._connect()

  def create_model_panel(self):
    self.model_gb = QGroupBox("Model")
    self.conf_slider = QSlider(Qt.Orientation.Horizontal, self.model_gb)
    self.conf_sb = QDoubleSpinBox()
    self.trkr_slider = QSlider(Qt.Orientation.Horizontal, self.model_gb)
    self.trkr_sb = QSpinBox()
    self.model_cb = QComboBox()
    self.model_cb.addItems([model.name for model in self.model_dir.iterdir() if model.suffix.casefold() == ".onnx"])
    self.trkr_reset_btn = QPushButton("Reset Tracker")
    self.coco_chk = QCheckBox("COCO")

    self.conf_slider.setRange(0, 100)
    self.conf_slider.setValue(55)
    self.trkr_slider.setRange(0, 200)
    self.trkr_slider.setValue(50)

    self.conf_sb.setRange(0, 1)
    self.conf_sb.setValue(self.conf_slider.value()/100)
    self.conf_sb.setSingleStep(0.01)
    self.conf_sb.setButtonSymbols(self.conf_sb.buttonSymbols().NoButtons)
    self.trkr_sb.setRange(0, 200)
    self.trkr_sb.setValue(self.trkr_slider.value())
    self.trkr_sb.setButtonSymbols(self.trkr_sb.buttonSymbols().NoButtons)

    conf_layout = QHBoxLayout()
    conf_layout.addWidget(self.conf_slider)
    conf_layout.addWidget(self.conf_sb)
    trkr_layout = QHBoxLayout()
    trkr_layout.addWidget(self.trkr_slider)
    trkr_layout.addWidget(self.trkr_sb)
    mdel_layout = QHBoxLayout()
    mdel_layout.addWidget(self.model_cb)
    mdel_layout.addStretch(1)
    mdel_layout.addWidget(self.coco_chk)

    layout = QFormLayout()
    layout.addRow("Model:", mdel_layout)
    layout.addRow("Conf. threshold:", conf_layout)
    layout.addRow("Tracker threshold:", trkr_layout)
    layout.addWidget(self.trkr_reset_btn)
    self.model_gb.setLayout(layout)

  def create_input_panel(self):
    self.input_gb = QGroupBox("Media source")

    self.source_cb = QComboBox()
    self.source_cb.addItems(["Camera", "File", "URL"])

    self.camera_widget = QWidget()
    self.camera_cb = QComboBox()
    self.cameras = QMediaDevices.videoInputs()
    self.camera_cb.addItems([camera.description() for camera in self.cameras])
    camera_layout = QFormLayout()
    camera_layout.addRow("Camera:", self.camera_cb)
    self.camera_widget.setLayout(camera_layout)

    self.url_widget = QWidget()
    self.url_le = QLineEdit()
    self.url_pb = QPushButton("Open URL")
    url_layout = QFormLayout()
    url_layout.addRow("URL:", self.url_le)
    url_layout.addWidget(self.url_pb)
    self.url_widget.setLayout(url_layout)

    self.file_widget = QWidget()
    self.file_pb = QPushButton("Browse...")
    file_layout = QFormLayout()
    file_layout.addWidget(self.file_pb)
    self.file_widget.setLayout(file_layout)

    self.sources_widget = QStackedWidget()
    self.sources_widget.addWidget(self.camera_widget)
    self.sources_widget.addWidget(self.file_widget)
    self.sources_widget.addWidget(self.url_widget)

    layout = QVBoxLayout()
    layout.addWidget(self.source_cb)
    layout.addWidget(self.sources_widget)
    self.input_gb.setLayout(layout)

  def create_roi_panel(self):
    self.roi_gb = QGroupBox("Region of Interest")

    self.roi_t_slider = QDoubleRangeSlider(Qt.Orientation.Horizontal, self.roi_gb)
    self.roi_b_slider = QDoubleRangeSlider(Qt.Orientation.Horizontal, self.roi_gb)
    self.roi_l_slider = QDoubleRangeSlider(Qt.Orientation.Horizontal, self.roi_gb)
    self.roi_r_slider = QDoubleRangeSlider(Qt.Orientation.Horizontal, self.roi_gb)
    self.full_frame_chk = QCheckBox("Full frame")

    self.roi_t_slider.setRange(0, 1)
    self.roi_b_slider.setRange(0, 1)
    self.roi_l_slider.setRange(0, 1)
    self.roi_r_slider.setRange(0, 1)

    self.roi_t_slider.setValue((0.33, 0.63))
    self.roi_b_slider.setValue((0.17, 0.71))
    self.roi_l_slider.setValue((0.3, 0.86))
    self.roi_r_slider.setValue((0.3, 0.86))

    self.ul_x_sb = DoubleSpinBox(value=self.roi_t_slider.value()[0])
    self.ur_x_sb = DoubleSpinBox(value=self.roi_t_slider.value()[1])
    self.bl_x_sb = DoubleSpinBox(value=self.roi_l_slider.value()[0])
    self.br_x_sb = DoubleSpinBox(value=self.roi_l_slider.value()[1])
    self.ul_y_sb = DoubleSpinBox(value=self.roi_b_slider.value()[0])
    self.bl_y_sb = DoubleSpinBox(value=self.roi_b_slider.value()[1])
    self.ur_y_sb = DoubleSpinBox(value=self.roi_r_slider.value()[0])
    self.br_y_sb = DoubleSpinBox(value=self.roi_r_slider.value()[1])

    t_layout = QHBoxLayout()
    t_layout.addWidget(self.roi_t_slider)
    t_layout.addWidget(self.ul_x_sb)
    t_layout.addWidget(self.ur_x_sb)
    l_layout = QHBoxLayout()
    l_layout.addWidget(self.roi_l_slider)
    l_layout.addWidget(self.ul_y_sb)
    l_layout.addWidget(self.bl_y_sb)

    b_layout = QHBoxLayout()
    b_layout.addWidget(self.roi_b_slider)
    b_layout.addWidget(self.bl_x_sb)
    b_layout.addWidget(self.br_x_sb)
    r_layout = QHBoxLayout()
    r_layout.addWidget(self.roi_r_slider)
    r_layout.addWidget(self.ur_y_sb)
    r_layout.addWidget(self.br_y_sb)

    slayout = QFormLayout()
    slayout.addRow("top:", t_layout)
    slayout.addRow("left:", l_layout)
    slayout.addRow("bottom:", b_layout)
    slayout.addRow("right:", r_layout)
    layout = QVBoxLayout()
    layout.addLayout(slayout)
    layout.addWidget(self.full_frame_chk)
    self.roi_gb.setLayout(layout)

  def _connect(self):
    self.camera_cb.currentIndexChanged.connect(self.on_camera_changed)
    self.source_cb.currentIndexChanged.connect(self.on_source_changed)
    self.conf_slider.valueChanged.connect(self.on_conf_changed)
    self.conf_sb.valueChanged.connect(lambda x: self.conf_slider.setValue(int(x*100)))
    self.trkr_slider.valueChanged.connect(self.on_trkr_changed)
    self.trkr_sb.valueChanged.connect(lambda x: self.trkr_slider.setValue(x))
    self.file_pb.clicked.connect(self.on_file_pb_clicked)
    self.url_pb.clicked.connect(self.on_url_pb_clicked)
    self.model_cb.currentTextChanged.connect(self.on_model_changed)
    self.coco_chk.toggled.connect(self._parent.frame_widget.on_coco_toggled)
    self.full_frame_chk.toggled.connect(self.on_full_frame_toggled)
    self.trkr_reset_btn.clicked.connect(self.on_trkr_reset_clicked)
    self.roi_t_slider.valueChanged.connect(self.on_roi_t_changed)
    self.roi_b_slider.valueChanged.connect(self.on_roi_b_changed)
    self.roi_l_slider.valueChanged.connect(self.on_roi_l_changed)
    self.roi_r_slider.valueChanged.connect(self.on_roi_r_changed)
    self.ul_x_sb.valueChanged.connect(self.on_ul_x_changed)
    self.ur_x_sb.valueChanged.connect(self.on_ur_x_changed)
    self.bl_x_sb.valueChanged.connect(self.on_bl_x_changed)
    self.br_x_sb.valueChanged.connect(self.on_br_x_changed)
    self.ul_y_sb.valueChanged.connect(self.on_ul_y_changed)
    self.bl_y_sb.valueChanged.connect(self.on_bl_y_changed)
    self.ur_y_sb.valueChanged.connect(self.on_ur_y_changed)
    self.br_y_sb.valueChanged.connect(self.on_br_y_changed)

  @Slot()
  def on_full_frame_toggled(self, state):
    if not state:
      self._parent.frame_worker.engine.roi = self.old_roi
    else:
      self.old_roi = [
        self.roi_t_slider.value(), self.roi_b_slider.value(),
        self.roi_l_slider.value(), self.roi_r_slider.value()
      ]
      self._parent.frame_worker.engine.roi = [(0, 1)] * 4

    self.ul_x_sb.setDisabled(state)
    self.ur_x_sb.setDisabled(state)
    self.bl_x_sb.setDisabled(state)
    self.br_x_sb.setDisabled(state)
    self.ul_y_sb.setDisabled(state)
    self.ur_y_sb.setDisabled(state)
    self.bl_y_sb.setDisabled(state)
    self.br_y_sb.setDisabled(state)
    self.roi_t_slider.setDisabled(state)
    self.roi_b_slider.setDisabled(state)
    self.roi_l_slider.setDisabled(state)
    self.roi_r_slider.setDisabled(state)

  @Slot()
  def on_camera_changed(self, value):
    self._parent.frame_widget.setCamera(self.cameras[value])

  @Slot()
  def on_source_changed(self, value):
    self.sources_widget.setCurrentIndex(value)
    self._parent.frame_widget.change_mode(value)

  @Slot(str)
  def on_model_changed(self, value: str):
    model = self.model_dir / value
    if not model.is_file():
      return
    self._parent.frame_worker.set_model(
      model_path = str(model),
      conf = self.conf_slider.value()/100,
      roi = [
        self.roi_t_slider.value(), self.roi_b_slider.value(),
        self.roi_l_slider.value(), self.roi_r_slider.value()
      ] if not self.full_frame_chk.isChecked() else [(0, 1)] * 4
    )

  @Slot()
  def on_file_pb_clicked(self, value):
    file_url, file_type = QFileDialog.getOpenFileUrl(self, "Open Video", "/", "Videos (*.mp4 *.m4v *.mkv *.avi *.flv *.mov *.webm);; All Files (*.*)")
    self._parent.frame_widget.setSource(file_url)

  @Slot()
  def on_url_pb_clicked(self, value):
    pass

  @Slot()
  def on_trkr_reset_clicked(self, value):
    self._parent.frame_worker.tracker.reset()
    # self.parent().parent().table.clear_table()

  @Slot()
  def on_trkr_changed(self, value):
    self.trkr_sb.setValue(value)
    self._parent.frame_worker.tracker.thres = value

  @Slot()
  def on_conf_changed(self, value):
    val = value/100
    self.conf_sb.setValue(val)
    self._parent.frame_worker.engine.conf = val

  @Slot()
  def on_roi_t_changed(self, value):
    self.ul_x_sb.setValue(value[0])
    self.ur_x_sb.setValue(value[1])
    self._parent.frame_worker.engine.roi = [
      value, self.roi_b_slider.value(),
      self.roi_l_slider.value(), self.roi_r_slider.value()
    ]

  @Slot()
  def on_roi_b_changed(self, value):
    self.bl_x_sb.setValue(value[0])
    self.br_x_sb.setValue(value[1])
    self._parent.frame_worker.engine.roi = [
      self.roi_t_slider.value(), value,
      self.roi_l_slider.value(), self.roi_r_slider.value()
    ]

  @Slot()
  def on_roi_l_changed(self, value):
    self.ul_y_sb.setValue(value[0])
    self.bl_y_sb.setValue(value[1])
    self._parent.frame_worker.engine.roi = [
      self.roi_t_slider.value(), self.roi_b_slider.value(),
      value, self.roi_r_slider.value()
    ]

  @Slot()
  def on_roi_r_changed(self, value):
    self.ur_y_sb.setValue(value[0])
    self.br_y_sb.setValue(value[1])
    self._parent.frame_worker.engine.roi = [
      self.roi_t_slider.value(), self.roi_b_slider.value(),
      self.roi_l_slider.value(), value
    ]

  @Slot()
  def on_ul_x_changed(self, value):
    self.roi_t_slider.setValue((value, self.roi_t_slider.value()[1]))
    self.ur_x_sb.setMinimum(value)

  @Slot()
  def on_ur_x_changed(self, value):
    self.roi_t_slider.setValue((self.roi_t_slider.value()[0], value))
    self.ul_x_sb.setMaximum(value)

  @Slot()
  def on_bl_x_changed(self, value):
    self.roi_b_slider.setValue((value, self.roi_b_slider.value()[1]))
    self.br_x_sb.setMinimum(value)

  @Slot()
  def on_br_x_changed(self, value):
    self.roi_b_slider.setValue((self.roi_b_slider.value()[0], value))
    self.bl_x_sb.setMaximum(value)

  @Slot()
  def on_ul_y_changed(self, value):
    self.roi_l_slider.setValue((value, self.roi_l_slider.value()[1]))
    self.bl_y_sb.setMinimum(value)

  @Slot()
  def on_bl_y_changed(self, value):
    self.roi_l_slider.setValue((self.roi_l_slider.value()[0], value))
    self.ul_y_sb.setMaximum(value)

  @Slot()
  def on_ur_y_changed(self, value):
    self.roi_r_slider.setValue((value, self.roi_r_slider.value()[1]))
    self.br_y_sb.setMinimum(value)

  @Slot()
  def on_br_y_changed(self, value):
    self.roi_r_slider.setValue((self.roi_r_slider.value()[0], value))
    self.ur_y_sb.setMaximum(value)
