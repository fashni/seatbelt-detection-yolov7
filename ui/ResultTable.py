import json
from datetime import datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (QHeaderView, QLabel, QTableWidget,
                               QTableWidgetItem)

from utils import array2pix


# (idx, max(recs['n_passenger']), max(recs['n_seat_belt']))
class ResultTable(QTableWidget):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.items = 0
    self.setColumnCount(3)
    self.setHorizontalHeaderLabels(["id", "n_ps", "n_sb"])
    self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    self._data = None

  def update(self, cboxes_id: npt.NDArray, frame_id: int, frame: npt.NDArray[np.uint8]):
    if self._data is None:
      self._data = {}
    for cbox in cboxes_id:
      box = cbox[:4]
      oid, n_ps, n_sb = cbox[4:].astype(int).tolist()
      img = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :]
      if oid not in self._data.keys():
        self._data[oid] = {
          "frame_id": [frame_id],
          "bbox": [box.tolist()],
          "n_passenger": [n_ps],
          "n_seatbelt": [n_sb],
          "image": [img.tolist()]
        }
        continue

      if "frame_id" in self._data[oid].keys():
        self._data[oid]["frame_id"] += [frame_id]
      else:
        self._data[oid]["frame_id"] = [frame_id]

      if "bbox" in self._data[oid].keys():
        self._data[oid]["bbox"] += [box.tolist()]
      else:
        self._data[oid]["bbox"] = [box.tolist()]

      if "n_passenger" in self._data[oid].keys():
        self._data[oid]["n_passenger"] += [n_ps]
      else:
        self._data[oid]["n_passenger"] = [n_ps]

      if "n_seatbelt" in self._data[oid].keys():
        self._data[oid]["n_seatbelt"] += [n_sb]
      else:
        self._data[oid]["n_seatbelt"] = [n_sb]

      if "image" in self._data[oid].keys():
        self._data[oid]["image"] += [img.tolist()]
      else:
        self._data[oid]["image"] = [img.tolist()]

    # self.fill_table()
    # print(self._data)
    # with open("runs/result.json", "w") as f:
    #   json.dump(self._data, f, indent=2)

  def fill_table(self, data=None):
    data = self._data if not data else data
    if data is None:
      return
    self.clear_table()
    for idx in reversed(list(data.keys())):
      recs = data[idx]

      # lbl = QLabel(self)
      # lbl.setPixmap(array2pix(recs["image"][np.argmax([im.size for im in recs["image"]])]))
      # lbl.setAlignment(Qt.AlignCenter)

      id_item = QTableWidgetItem(f"{idx:d}")
      id_item.setTextAlignment(Qt.AlignCenter)
      n_ps_item = QTableWidgetItem(f"{max(recs['n_passenger'])}")
      n_ps_item.setTextAlignment(Qt.AlignCenter)
      n_sb_item = QTableWidgetItem(f"{max(recs['n_seatbelt'])}")
      n_sb_item.setTextAlignment(Qt.AlignCenter)
      self.insertRow(self.items)
      self.setItem(self.items, 0, id_item)
      self.setItem(self.items, 1, n_ps_item)
      self.setItem(self.items, 2, n_sb_item)
      # self.setCellWidget(self.items, 1, lbl)
      self.items += 1

  def clear_table(self):
    self.setRowCount(0)
    self.items = 0

  def save_data(self):
    if self._data is None:
      return
    outdir = Path("runs")
    outdir.mkdir(exist_ok=True)
    fn = outdir / f"result_{datetime.timestamp(datetime.now())}.json"
    with fn.open("w") as f:
      json.dump(self._data, f)
    return str(fn)
