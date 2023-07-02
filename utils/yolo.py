import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from cfg import get_configs
from utils import (draw_detections, draw_roi, expand_boxes, hex2bgr, letterbox,
                   scale_coords)


class YoloInfer:
  def __init__(self, path=None, conf=0.35, verbose=False):
    providers = get_configs()["providers"]
    self.providers = [provider for provider in providers if provider in ort.get_available_providers()]
    self.roi = [(0, 1)] * 4
    self.conf = conf
    self.verbose = verbose
    self.class_names = None
    if path is not None:
      self.init_model(path)

  def __call__(self, image, classes=None, full=True):
    return self.detect(image, classes, full)

  @property
  def roi(self):
    return self._roi

  @roi.setter
  def roi(self, value):
    _roi = np.array(value)
    _roi[2:] = _roi[2:].T
    self._roi = np.r_[_roi[[0, 2, 1, 3]].T[:, :2], _roi[[0, 2, 1, 3]].T[:, 2:][[1, 0]]]

  def init_model(self, path: str):
    classes_path = Path(path).with_suffix(".txt")
    if classes_path.is_file():
      with classes_path.open("r") as f:
        self.class_names = [line.replace("\n", "") for line in f.readlines()]

    self.session = ort.InferenceSession(path, providers=self.providers)
    self.providers = self.session.get_providers()
    self.get_input_details()
    self.get_output_details()
    self.warmup()

  def warmup(self):
    print("Warming up...")
    warmup_data = np.zeros((self.input_h, self.input_w, 3)).astype(np.uint8)
    self.detect(warmup_data, full=True)
    print("Ready")

  def get_input_details(self):
    inputs = self.session.get_inputs()
    self.input_names = [inp.name for inp in inputs]
    self.input_shape = inputs[0].shape
    self.input_h, self.input_w = self.input_shape[2:]

  def get_output_details(self):
    outputs = self.session.get_outputs()
    self.output_names = [outp.name for outp in outputs]

  def get_roi_vertices(self, image: np.ndarray) -> np.ndarray:
    self.img_h, self.img_w = image.shape[:2]
    self.verts = (self._roi * [self.img_w, self.img_h]).astype(int)

  def get_roi(self, image: np.ndarray) -> np.ndarray:
    self.get_roi_vertices(image)
    self.roi_l, self.roi_t, self.roi_w, self.roi_h = cv2.boundingRect(self.verts)
    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.int32(self.verts)], (255,)*image.shape[2])
    masked = cv2.bitwise_and(image, mask)
    return masked[self.roi_t:self.roi_t+self.roi_h, self.roi_l:self.roi_l+self.roi_w, :]

  def preprocess(self, image: np.ndarray, full=True) -> np.ndarray:
    start = time.perf_counter()
    self.img_h, self.img_w = image.shape[:2]

    roi = image.copy() if full else self.get_roi(image)

    # roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi, _, _ = letterbox(roi, (self.input_h, self.input_w), auto=False)
    roi = roi / 255.0
    roi = roi.transpose(2, 0, 1)

    if self.verbose:
      print(f"Preprocess time: {(time.perf_counter() - start)*1000:.2f} ms")

    return roi[np.newaxis, :, :, :].astype(np.float32)

  def inference(self, input_tensor: np.ndarray) -> np.ndarray:
    start = time.perf_counter()
    outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

    if self.verbose:
      print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")

    return outputs

  def detect(self, image: np.ndarray, classes=None, full=True):
    input_tensor = self.preprocess(image, full)
    outputs = self.inference(input_tensor)

    return self.postprocess(outputs, classes, full)

  def postprocess(self, outputs: np.ndarray, classes=None, full=True):
    start = time.perf_counter()

    if full:
      self.roi_h = self.img_h
      self.roi_w = self.img_w

    out = outputs[0]
    if classes is not None:
      if isinstance(classes, int):
        classes = [classes]
      out = out[(out[:, 5:6] == np.array(classes)).any(1)]

    scores = out[:, -1]
    predictions = out[:, [0, 5, 1, 2, 3, 4]]

    valid_scores = scores > self.conf
    predictions = predictions[valid_scores, :]
    scores = scores[valid_scores]

    if scores.size == 0:
      if self.verbose:
        print(f"Postprocess time: {(time.perf_counter() - start)*1000:.2f} ms")
      return np.array([]), np.array([]), np.array([])

    # batch_numbers = predictions[:, 0]
    class_ids = predictions[:, 1].astype(int)
    boxes = scale_coords((self.input_h, self.input_w), predictions[:, 2:], (self.roi_h, self.roi_w))

    if not full:
      expand_boxes(boxes, (self.roi_l, self.roi_t))

    if self.verbose:
      print(f"Postprocess time: {(time.perf_counter() - start)*1000:.2f} ms")

    return boxes, scores, class_ids


if __name__ == "__main__":
  from tracker import EuclideanDistTracker

  from utils import bbox_iou, cv2_imshow, draw_detections_id

  tracker = EuclideanDistTracker()
  colors = [hex2bgr("23aaf2"), hex2bgr("f9394a"), hex2bgr("18b400")]
  verbose = True
  img = cv2.imread("media/test (1).png")
  img_h, img_w = img.shape[:2]
  path = "models/seatbelt-tiny.onnx"
  engine = YoloInfer(path, verbose=verbose, conf=0.35)
  # engine.roi = [(0, 1), (0, 1), (0, 1), (0, 1)]
  # engine.roi = [(0.42, 0.52), (0.41, 0.58), (0.37, 0.86), (0.37, 0.86)]
  engine.roi = [(0.33, 0.63), (0.17, 0.71), (0.3, 0.86), (0.3, 0.86)] # [top, bottom, left, right]

  res = engine.detect(img, classes=[0, 3], full=False)
  boxes, scores, class_ids = res

  if boxes.size > 0:
    cars = boxes[class_ids==0]
    windshields = boxes[class_ids==3]
    if cars.size > 0:
      cars_id = np.zeros((cars.shape[0], cars.shape[1]+3))
      cars_id[:, :5] = tracker.update(cars)

      if windshields.size > 0:
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

        windshields_imgs = []
        for wbox in windshields_id:
          box = wbox.astype(int)
          ws = img[box[1]:box[3], box[0]:box[2], :]
          ps_res = engine.detect(ws, classes=[1, 2], full=True)
          ps_boxes, ps_scores, ps_class_ids = ps_res
          draw_detections(ws, *ps_res, class_names=engine.class_names)
          windshields_imgs.append((int(wbox[-1]), ps_res, ws))
          tmp = cars_id[cars_id[:, 4]==wbox[-1]]
          tmp[0, 6] = ps_class_ids[ps_class_ids == 2].size
          tmp[0, 5] = ps_class_ids[ps_class_ids == 1].size
          cars_id[cars_id[:, 4] == wbox[-1]] = tmp

  print(boxes.shape)
  print(cars.shape)
  print(windshields.shape)
  print(cars_id.shape)
  print(windshields_id.shape)

  for box in cars_id:
    print(box.astype(int))

  # print(res)
  # res = engine(img)
  # res = engine(img)
  start = time.perf_counter()
  draw_roi(img, engine.verts, 2, hex2bgr("#ffd96a"))
  #render = draw_detections_id(img, *res)
  render = draw_detections_id(img, cars_id[:, :5], scores[class_ids==0], class_ids[class_ids==0], colors=colors)
  # render = draw_detections_id(render, windshields_id, scores[class_ids==3], class_ids[class_ids==3])
  if verbose:
    print(f"Rendering time: {(time.perf_counter() - start)*1000:.2f} ms")
  cv2_imshow(cv2.resize(render, (img_w//2, img_h//2)))

# h_img, w_img = img.shape[:2]
# verts = np.vstack((np.hstack([x.value for x in roi_xs])*w_img, np.hstack([y.value for y in roi_ys])*h_img)).T.astype(int)
# verts[[2, 3]] = verts[[3,2]]
# l,t,w,h = cv2.boundingRect(verts)

# mask = np.zeros(img.shape, dtype=np.uint8)
# cv2.fillPoly(mask, [np.int32(verts)], (255,)*img.shape[2])
# masked = cv2.bitwise_and(img, mask)
# roi = masked[t:t+h, l:l+w, :]
