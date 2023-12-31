import sys

from PySide6.QtCore import Slot
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import (QApplication, QMainWindow, QMessageBox,
                               QStyleFactory)

from cfg import get_configs, set_environment_variables
from ui import MainWidget

set_environment_variables()
configs = get_configs()

class MainWindow(QMainWindow):
  def __init__(self, widget):
    QMainWindow.__init__(self)
    self.setWindowTitle("Seatbelt Detection with Yolov7")

    self.menu = self.menuBar()
    self.file_menu = self.menu.addMenu("File")
    exit_action = QAction("Exit", self)
    exit_action.setShortcut("Ctrl+Q")
    exit_action.triggered.connect(self.exit_app)
    self.file_menu.addAction(exit_action)

    self._theme_group = QActionGroup(self)
    self._theme_group.setExclusive(True)
    self.theme_menu = self.menu.addMenu("Theme")
    for theme in QStyleFactory.keys():
      theme_action = QAction(theme, self._theme_group)
      theme_action.setCheckable(True)
      theme_action.setData(theme)
      if theme.casefold() == configs.get("theme"):
        theme_action.setChecked(True)
      self.theme_menu.addAction(theme_action)
    self._theme_group.triggered.connect(self._changeStyle)

    self.statusBar().showMessage("Ready")
    widget.set_status_bar(self.statusBar())
    self.setCentralWidget(widget)

  def check_model(self):
    if self.centralWidget().left_sidebar.model_cb.count() > 0:
      return
    QMessageBox.warning(
      self.centralWidget(),
      "No Model", "No ONNX model found in the model directory",
      buttons=QMessageBox.Close
    )
    self.exit_app(True)
    sys.exit()

  @Slot(QAction)
  def _changeStyle(self, style: QAction):
    self.changeStyle(style.data())

  def changeStyle(self, style_name):
    QApplication.setStyle(QStyleFactory.create(style_name))
    QApplication.setPalette(QApplication.style().standardPalette())

  @Slot()
  def exit_app(self, checked):
    self.centralWidget().on_close()
    QApplication.quit()

  def closeEvent(self, event):
    self.exit_app(True)
    super().closeEvent(event)


if __name__ == "__main__":
  app = QApplication(sys.argv)
  QApplication.setStyle(QStyleFactory.create(configs.get("theme")))
  QApplication.setPalette(QApplication.style().standardPalette())

  widget = MainWidget()
  window = MainWindow(widget)
  window.resize(1000, 600)
  window.show()
  window.check_model()

  sys.exit(app.exec())
