import random

import cv2
import numpy as np


MAX_NCLASS = 100

rng = np.random.default_rng(3)
def_colors = rng.uniform(0, 255, size=(MAX_NCLASS+1, 3))

def draw_roi(frame, verts=None, thickness=None, color=None):
  tl = thickness or round(0.002 * (frame.shape[0] + frame.shape[1]) / 2) + 1
  color = color or def_colors[-1]
  cv2.polylines(frame, [np.int32(verts)], True, color, tl, cv2.LINE_AA)


def draw_fps(image, fps):
  tl = round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
  cv2.putText(image, f"{fps:.2f} fps", (tl, tl*10), 0, tl/3, [0, 0, 0], tl+3, cv2.LINE_AA)
  cv2.putText(image, f"{fps:.2f} fps", (tl, tl*10), 0, tl/3, [255, 255, 255], max(tl - 1, 1), cv2.LINE_AA)


def draw_detections(image, boxes, scores, class_ids, class_names=None, colors=None, coco=False):
  det_img = image
  if class_names is None:
    class_names = np.arange(MAX_NCLASS+1)
  if colors is None:
    colors = def_colors
  for box, score, class_id in zip(boxes, scores, class_ids):
    color = colors[class_id]
    plot_one_box(
      x = box,
      img = det_img,
      color = color,
      label = f"{class_names[class_id]}: {score:.2f}" if coco else f"{class_id}",
      line_thickness = 3
    )
  return det_img


def draw_detections_id(image, boxes, scores, class_ids, colors=None):
  det_img = image.copy()
  if colors is None:
    colors = def_colors
  for box, score, class_id in zip(boxes, scores, class_ids):
    color = colors[class_id]
    plot_one_box(box[:-1], det_img, color, f"id: {int(box[-1]):d}; {score:.2f}")
  return det_img


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
  # Plots one bounding box on image img
  tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
  if color is None:
    color = [random.randint(0, 255) for _ in range(3)]
  c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
  cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
  if label:
    tf = max(tl - 1, 1)  # font thickness
    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
  # Resize and pad image while meeting stride-multiple constraints
  shape = img.shape[:2]  # current shape [height, width]
  if isinstance(new_shape, int):
    new_shape = (new_shape, new_shape)

  # Scale ratio (new / old)
  r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
  if not scaleup:  # only scale down, do not scale up (for better test mAP)
    r = min(r, 1.0)

  # Compute padding
  ratio = r, r  # width, height ratios
  new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
  dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
  if auto:  # minimum rectangle
    dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
  elif scaleFill:  # stretch
    dw, dh = 0.0, 0.0
    new_unpad = (new_shape[1], new_shape[0])
    ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

  dw /= 2  # divide padding into 2 sides
  dh /= 2

  if shape[::-1] != new_unpad:  # resize
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
  top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
  left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
  img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
  return img, ratio, (dw, dh)


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
  # Rescale coords (xyxy) from img1_shape to img0_shape
  if ratio_pad is None:  # calculate from img0_shape
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
  else:
    gain = ratio_pad[0][0]
    pad = ratio_pad[1]

  coords[:, ::2] -= pad[0]  # x padding
  coords[:, 1::2] -= pad[1]  # y padding
  coords[:, :4] /= gain
  clip_coords(coords, img0_shape)
  return coords


def clip_coords(boxes, img_shape):
  # Clip bounding xyxy bounding boxes to image shape (height, width)
  boxes[:, ::2] = boxes[:, ::2].clip(0, img_shape[1])  # x1, x2
  boxes[:, 1::2] = boxes[:, 1::2].clip(0, img_shape[0])  # y1, y2


def expand_boxes(boxroi, roi_tl):
  roi_x, roi_y = roi_tl
  boxroi[:, ::2] += roi_x
  boxroi[:, 1::2] += roi_y


def hex2rgb(hex):
  return tuple(int(hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))


def hex2bgr(hex):
  return tuple(int(hex.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))


def cv2_imshow(image):
  cv2.imshow("", image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()


def bbox_iou(box1, box2):
  # Get the coordinates of bounding boxes
  b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
  b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

  # Get the coordinates of the intersection rectangle
  inter_rect_x1 = np.maximum(b1_x1, b2_x1)
  inter_rect_y1 = np.maximum(b1_y1, b2_y1)
  inter_rect_x2 = np.minimum(b1_x2, b2_x2)
  inter_rect_y2 = np.minimum(b1_y2, b2_y2)
  # Intersection area
  inter_rect_x = inter_rect_x2 - inter_rect_x1 + 1
  inter_rect_y = inter_rect_y2 - inter_rect_y1 + 1
  inter_area = np.minimum(inter_rect_x.max(), np.maximum(inter_rect_x, 0)) * \
               np.minimum(inter_rect_y.max(), np.maximum(inter_rect_y, 0))

  # Union Area
  b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
  b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

  iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

  return iou

def rgb8_to_jpeg(array, quality=95):
  param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
  status, buffer = cv2.imencode('.jpg', array[:, :, ::-1], param)
  if status:
    return bytes(buffer)
  return None

def xywh2xyxy(x):
  # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
  y = np.copy(x)
  y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
  y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
  y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
  y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
  return y
