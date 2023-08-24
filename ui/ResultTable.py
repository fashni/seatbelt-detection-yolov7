import json
from datetime import datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHeaderView, QTableWidget, QTableWidgetItem

from utils import rgb8_to_jpeg


# (idx, max(recs['n_passenger']), max(recs['n_seat_belt']))
class ResultTable(QTableWidget):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.items = 0
    self.setColumnCount(3)
    self.setHorizontalHeaderLabels(["id", "n_ps", "n_sb"])
    self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    self.clear_data()
    self._save = False

  @property
  def save(self):
    return self._save

  @save.setter
  def save(self, state: bool):
    self._save = state

  def update(self, cboxes_id: npt.NDArray, frame_id: int, frame: npt.NDArray[np.uint8]):
    if self._data is None:
      self._data = {}
      self._img_id = {}
      self.outdir = Path("runs")
      self.outdir.mkdir(exist_ok=True)
      self.rundir = self.outdir / f"result_{datetime.timestamp(datetime.now())}"
      self.rundir.mkdir(exist_ok=True)

    for cbox in cboxes_id:
      box = cbox[:4]
      oid, n_ps, n_sb = cbox[4:].astype(int).tolist()
      img = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :]
      oid_dir = self.rundir / f"{oid}"
      if oid not in self._data.keys():
        self._img_id[oid] = 0
        self._data[oid] = {
          "frame_id": [frame_id],
          "bbox": [box.tolist()],
          "n_passenger": [n_ps],
          "n_seatbelt": [n_sb],
          "image": [f"{oid}/{self._img_id[oid]}_{n_ps}_{n_sb}.jpg"]
        }
        if self.save:
          oid_dir.mkdir(exist_ok=True)
          with (oid_dir / f"{self._img_id[oid]}_{n_ps}_{n_sb}.jpg").open("wb") as f:
            f.write(rgb8_to_jpeg(img, 100))
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

      self._img_id[oid] += 1
      img_filename = f"{self._img_id[oid]}_{n_ps}_{n_sb}.jpg"
      if "image" in self._data[oid].keys():
        self._data[oid]["image"] += [f"{oid}/{img_filename}"]
      else:
        self._data[oid]["image"] = [f"{oid}/{img_filename}"]

      if self.save:
        oid_dir.mkdir(exist_ok=True)
        with (oid_dir / img_filename).open("wb") as f:
          f.write(rgb8_to_jpeg(img, 100))

    if self.save:
      with (self.rundir / "data.json").open("w") as f:
        json.dump(self._data, f)

  def fill_table(self, data=None):
    data = self._data if not data else data
    if data is None:
      return
    self.clear_table()
    for idx in reversed(list(data.keys())):
      recs = data[idx]
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
      self.items += 1

  def clear_table(self):
    self.setRowCount(0)
    self.items = 0

  def clear_data(self):
    self._data = None
    self._img_id = None

