import os
from PySide6.QtCore import Slot
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QMessageBox, QPushButton,
                               QSplitter, QStatusBar, QVBoxLayout, QWidget)

from ui import FrameWidget, ResultTable, SideBar
from utils.worker import YoloFrameWorker


class MainWidget(QWidget):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.status_bar = None
    self.table = ResultTable()
    self.frame_worker = YoloFrameWorker(conf=0.55, result_table=self.table)
    self.frame_widget = FrameWidget(worker=self.frame_worker)
    self.left_sidebar = SideBar(self)

    self.right_sidebar = QWidget()
    self.clear_data_pb = QPushButton("Clear Data")
    self.save_data_pb = QPushButton("Save Data")
    button_layout = QHBoxLayout()
    button_layout.addStretch(1)
    button_layout.addWidget(self.save_data_pb)
    button_layout.addWidget(self.clear_data_pb)
    r_layout = QVBoxLayout()
    r_layout.addWidget(self.table)
    r_layout.addLayout(button_layout)
    self.right_sidebar.setLayout(r_layout)

    # QWidget Layout
    self.layout = QHBoxLayout()
    self.splitter = QSplitter()
    self.splitter.addWidget(self.left_sidebar)
    self.splitter.addWidget(self.frame_widget)
    self.splitter.addWidget(self.right_sidebar)
    self.splitter.setSizes([250, 500, 250])
    self.layout.addWidget(self.splitter)

    # Set the layout to the QWidget
    self.setLayout(self.layout)

    self.left_sidebar.on_model_changed(self.left_sidebar.model_cb.currentText())
    self.save_data_pb.clicked.connect(self.save_data)
    self.clear_data_pb.clicked.connect(self.clear_data)

  def set_status_bar(self, status_bar: QStatusBar):
    providers = [
      prov.replace("ExecutionProvider", "")
      for prov in self.frame_worker.engine.providers
    ]
    self.status_bar = status_bar
    self.file_lbl = QLabel()
    self.backend_lbl = QLabel()
    self.exprov_lbl = QLabel()

    self.file_lbl.setStatusTip("Active Media")
    self.backend_lbl.setStatusTip("Media Backend")
    self.exprov_lbl.setStatusTip("Active Execution Providers")

    self.status_bar.addPermanentWidget(self.file_lbl)
    self.status_bar.addPermanentWidget(self.backend_lbl)
    self.status_bar.addPermanentWidget(self.exprov_lbl)

    self.backend_lbl.setText(os.getenv("QT_MEDIA_BACKEND", ""))
    self.exprov_lbl.setText(", ".join(providers))

  @Slot()
  def save_data(self):
    filename = self.table.save_data()
    if filename is not None:
      ret = QMessageBox.information(self, "Data saved succesfully", f"{filename}")
      return

  @Slot()
  def clear_data(self):
    ret = QMessageBox.question(self, "Are you sure?", "All data and tracker id will be deleted.")
    if ret == QMessageBox.No:
      return
    self.table.clear_table()
    self.table._data = None
    self.frame_worker.tracker.reset()

  def on_close(self):
    self.frame_widget.on_close()

  def closeEvent(self, event):
    self.on_close()
    super().closeEvent(event)
