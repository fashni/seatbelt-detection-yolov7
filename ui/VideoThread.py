import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, Slot, QThread
from PySide6.QtWidgets import QLabel, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QFormLayout


class VideoThread(QThread):
  frameChanged = Signal(np.ndarray)
  def __init__(self, source=0):
    super().__init__()
    self._run_flag = False
    self._cap = None
    self.setSource(source)

  def cap(self):
    return self._cap

  def setSource(self, source):
    if self._run_flag:
      return
    self.source = source
    if self._cap is not None:
      self._cap.release()
    self._cap = cv2.VideoCapture(self.source)

  def changeState(self, state):
    self._run_flag = state

  def run(self):
    self._cap = cv2.VideoCapture(self.source)
    while self._run_flag:
      ret, cv_img = self._cap.read()
      if ret:
        self.frameChanged.emit(cv_img)
    # shut down capture system

  def stop(self):
    """Sets run flag to False and waits for thread to finish"""
    self._run_flag = False
    self._cap.release()
    self.wait()


class FrameWidget(QWidget):
  def __init__(self, worker=None, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.init_ui()
    self.yolo_thread = VideoThread()
    self._connect()

  def init_ui(self):
    self.video_widget = QLabel()
    self.start_button = QPushButton("Start")
    self.stop_button = QPushButton("Stop")
    self.process_button = QPushButton("Process Frame")

    self.start_button.setCheckable(True)
    self.process_button.setCheckable(True)

    button_layout1 = QHBoxLayout()
    button_layout1.addWidget(self.start_button)
    button_layout1.addWidget(self.stop_button)
    button_layout2 = QVBoxLayout()
    button_layout2.addLayout(button_layout1)
    button_layout2.addWidget(self.process_button)

    layout = QVBoxLayout()
    layout.addWidget(self.video_widget)
    layout.addLayout(button_layout2)
    self.setLayout(layout)

  def _connect(self):
    self.yolo_thread.frameChanged.connect(self._update)
    self.start_button.toggled.connect(self.on_start_toggled)
    self.stop_button.clicked.connect(self.on_stop_clicked)

  @Slot(np.ndarray)
  def _update(self, frame):
    qt_img = self.convert_cv_qt(frame)
    self.image_label.setPixmap(qt_img)

  @Slot(bool)
  def on_start_toggled(self, state):
    self.yolo_thread.changeState(state)

  @Slot()
  def on_stop_clicked(self):
    self.yolo_thread.stop()


