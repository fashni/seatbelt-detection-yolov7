import time
from typing import Optional

import numpy as np
import numpy.typing as npt
import qimage2ndarray
from PySide6 import QtCore, QtGui
from PySide6.QtMultimedia import QVideoFrame
from araviq6.array2qvideoframe import array2qvideoframe

from utils import (bbox_iou, draw_detections, draw_detections_id, draw_fps,
                   draw_roi, hex2rgb)
from utils.tracker import EuclideanDistTracker
from utils.yolo import YoloInfer


class VideoFrameWorker(QtCore.QObject):
  videoFrameProcessed = QtCore.Signal(QVideoFrame, np.ndarray)
  def __init__(self, parent=None):
    super().__init__(parent)
    self._ready = True

  def ready(self) -> bool:
    return self._ready

  def runProcess(self, frame: QVideoFrame):
    self._ready = False

    qimg = frame.toImage()  # must assign to avoid crash
    array = self.imageToArray(qimg)
    processedArray = self.processArray(array)
    processedFrame = self.arrayToVideoFrame(processedArray, frame)

    self.videoFrameProcessed.emit(processedFrame, processedArray)
    self._ready = True

  def imageToArray(self, image: QtGui.QImage) -> npt.NDArray[np.uint8]:
    if image.isNull():
      ret = np.empty((0, 0, 0), dtype=np.uint8)
    else:
      ret = qimage2ndarray.rgb_view(image, byteorder=None)
    return ret

  def processArray(self, array: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    return array

  def arrayToVideoFrame(self, array: npt.NDArray[np.uint8], hintFrame: QVideoFrame) -> QVideoFrame:
    if array.size == 0:
      ret = hintFrame
    else:
      ret = array2qvideoframe(array)
      ret.map(hintFrame.mapMode())
      ret.setStartTime(hintFrame.startTime())
      ret.setEndTime(hintFrame.endTime())
    return ret


class VideoFrameProcessor(QtCore.QObject):
  _processRequested = QtCore.Signal(QVideoFrame)
  videoFrameProcessed = QtCore.Signal(QVideoFrame, np.ndarray)

  def __init__(self, parent=None):
    super().__init__(parent)
    self._worker = None
    self._skipIfRunning = True

    self._processorThread = QtCore.QThread()
    self._processorThread.start()

  def worker(self) -> Optional[VideoFrameWorker]:
    return self._worker

  def setWorker(self, worker: Optional[VideoFrameWorker]):
    oldWorker = self.worker()
    if oldWorker is not None:
      self._processRequested.disconnect(oldWorker.runProcess)
      oldWorker.videoFrameProcessed.disconnect(self.videoFrameProcessed)
    self._worker = worker
    if worker is not None:
      self._processRequested.connect(worker.runProcess)
      worker.videoFrameProcessed.connect(self.videoFrameProcessed)
      worker.moveToThread(self._processorThread)

  def skipIfRunning(self) -> bool:
    return self._skipIfRunning

  def setSkipIfRunning(self, flag: bool):
    self._skipIfRunning = flag

  @QtCore.Slot(QVideoFrame)
  def processVideoFrame(self, frame: QVideoFrame):
    worker = self.worker()
    if worker is not None:
      if worker.ready() or not self.skipIfRunning():
        self._processRequested.emit(frame)
    else:
      self.videoFrameProcessed.emit(frame)

  def stop(self):
    self._processorThread.quit()
    self._processorThread.wait()


class YoloFrameWorker(VideoFrameWorker):
  def __init__(self, model_path=None, conf=0.25, result_table=None, verbose=False, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._enabled = False
    self._coco = False
    self.result_table = result_table
    self.has_model = model_path is not None
    self.engine = YoloInfer(model_path, conf, verbose)
    self.tracker = EuclideanDistTracker(thres=50)
    self.frame_id = 0
    self.prev_frame_time = 0
    self.new_frame_time = 0
    self.colors = [hex2rgb("23aaf2"), hex2rgb("f9394a"), hex2rgb("18b400")]

  def set_model(self, model_path: str, conf=0.25, roi=None):
    self.engine.init_model(model_path)
    self.engine.conf = conf
    if roi is not None:
      self.engine.roi = roi
    self.has_model = True

  @property
  def enabled(self):
    return self._enabled

  @enabled.setter
  def enabled(self, state: bool):
    self._enabled = state

  @property
  def coco(self):
    return self._coco
  
  @coco.setter
  def coco(self, state: bool):
    self._coco = state

  def update_table(self, cars_id, img):
    self.result_table.update(cars_id, self.frame_id, img.copy())
    self.result_table.fill_table()

  def postprocess(self, img, boxes, scores, class_ids, cars_id=None):
    self.new_frame_time = time.perf_counter()
    if boxes[class_ids==0].size > 0 and cars_id is not None:
      res_img = draw_detections_id(img, cars_id[:, :5], scores[class_ids==0], class_ids[class_ids==0], colors=self.colors)
      draw_fps(res_img, 1/(self.new_frame_time-self.prev_frame_time))
    elif self._coco:
      res_img = draw_detections(img.copy(), boxes, scores, class_ids, self.engine.class_names, coco=True)
      draw_fps(res_img, 1/(self.new_frame_time-self.prev_frame_time))
    else:
      draw_fps(img, 1/(self.new_frame_time-self.prev_frame_time))
      res_img = img
    self.prev_frame_time = self.new_frame_time
    self.frame_id += 1

    return res_img

  def processArray(self, image: np.ndarray) -> np.ndarray:
    if image.size == 0:
      return image

    img = image.copy()
    if not self.has_model:
      return img

    self.engine.get_roi_vertices(img)
    draw_roi(img, self.engine.verts, color=hex2rgb("#ffd96a"))
    if self.has_model and not self._enabled:
      self.new_frame_time = time.perf_counter()
      draw_fps(img, 1/(self.new_frame_time-self.prev_frame_time))
      self.prev_frame_time = self.new_frame_time
      return img

    classes = None if self._coco else [0, 3]
    res = self.engine.detect([image], classes=classes, full=False)
    boxes, scores, class_ids = res
    # cars_id = self.tracker.update(boxes[class_ids==0])

    if boxes[0].size == 0 or self._coco:
      return self.postprocess(img, boxes[0], scores[0], class_ids[0])

    cars = boxes[0][class_ids[0]==0]
    windshields = boxes[0][class_ids[0]==3]

    if cars.size == 0:
      return self.postprocess(img, boxes[0], scores[0], class_ids[0])

    cars_id = np.zeros((cars.shape[0], cars.shape[1]+3))
    cars_id[:, :5] = self.tracker.update(cars)

    if windshields.size == 0:
      self.update_table(cars_id, image)
      return self.postprocess(img, boxes[0], scores[0], class_ids[0], cars_id)

    windshields_id = -np.ones((windshields.shape[0], windshields.shape[1]+1))
    windshields_id[:, :-1] = windshields

    invalid = []
    for idx, w in enumerate(windshields_id):
      car_index = bbox_iou(np.expand_dims(w[:4], 0), cars[:, :4])
      if car_index.sum()==0:
        invalid.append(idx)
        continue
      w[-1] = cars_id[np.argmax(car_index), 4]
    windshields_id = np.delete(windshields_id, invalid, 0)

    wboxes = [box for box in windshields_id.astype(int)]
    ws_imgs = [img[box[1]:box[3], box[0]:box[2], :] for box in windshields_id.astype(int)]
    ps_ress = self.engine.detect(ws_imgs, classes=[1, 2], full=True)
    ps_boxes, ps_scores, ps_class_ids = ps_ress
    for wbox, ws, ps_box, ps_score, ps_class_id in zip(wboxes, ws_imgs, ps_boxes, ps_scores, ps_class_ids):
      draw_detections(ws, ps_box, ps_score, ps_class_id, class_names=self.engine.class_names)
      tmp = cars_id[cars_id[:, 4]==wbox[-1]]
      tmp[0, 6] = ps_class_id[ps_class_id == 2].size
      tmp[0, 5] = ps_class_id[ps_class_id == 1].size
      cars_id[cars_id[:, 4] == wbox[-1]] = tmp

    self.update_table(cars_id, image)
    return self.postprocess(img, boxes[0], scores[0], class_ids[0], cars_id)
