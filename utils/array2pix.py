import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap

def array2pix(cv_img):
  rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
  h, w, ch = rgb_image.shape
  bytes_per_line = ch * w
  convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
  p = convert_to_Qt_format.scaled(rgb_image.shape[0], cv_img.shape[1], Qt.AspectRatioMode.KeepAspectRatio)
  return QPixmap.fromImage(p)
