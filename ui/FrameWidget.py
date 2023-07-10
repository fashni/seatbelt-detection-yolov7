from PySide6.QtCore import QUrl, Slot
from PySide6.QtMultimedia import (QCamera, QCameraDevice, QMediaCaptureSession,
                                  QMediaDevices, QMediaPlayer, QVideoFrame,
                                  QVideoSink)
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import QCheckBox, QPushButton, QVBoxLayout, QWidget
from araviq6 import MediaController

from utils.worker import VideoFrameProcessor, VideoFrameWorker


class FrameWidget(QWidget):
  def __init__(self, worker=None, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.mode = 0
    self.init_ui()
    self.init_camera()
    self.init_video()
    self.set_layout()
    self.set_worker(worker)
    self.set_pipeline()
    self.change_mode(0) # 0: camera, 1: file, 2: url

  def init_ui(self):
    self.frame_processor = VideoFrameProcessor()
    self.video_sink = QVideoSink()
    self.video_widget = QVideoWidget()
    self.process_button = QPushButton("Process Frame")
    self.skip_frame_button = QCheckBox("Skip Frames")
    self.skip_frame_button.setChecked(True)

  def init_camera(self):
    self.cam_start = False
    self.capture_session = QMediaCaptureSession()
    self.camera_button = QPushButton("Start Camera")
    self.setCamera(QMediaDevices.defaultVideoInput())

  def init_video(self):
    self.video_player = QMediaPlayer()
    self.media_controller = MediaController()

  def set_layout(self):
    self.video_widget.setMinimumHeight(500)
    layout = QVBoxLayout()
    layout.addWidget(self.video_widget)
    layout.addWidget(self.skip_frame_button)
    layout.addWidget(self.media_controller)
    layout.addWidget(self.camera_button)
    layout.addWidget(self.process_button)
    layout.addStretch(1)
    self.setLayout(layout)

  def set_worker(self, worker: VideoFrameWorker):
    self.worker = worker
    self.frame_processor.setWorker(self.worker)

  def set_pipeline(self):
    self.video_sink.videoFrameChanged.connect(self.frame_processor.processVideoFrame)
    self.frame_processor.videoFrameProcessed.connect(self.display_frame)
    self.frame_processor.setWorker(self.worker)

    self.media_controller.setPlayer(self.video_player)
    self.process_button.setCheckable(True)
    self.process_button.toggled.connect(self.on_process_btn_toggled)
    self.camera_button.setCheckable(True)
    self.camera_button.toggled.connect(self.on_camera_btn_toggled)
    self.skip_frame_button.toggled.connect(self.on_skip_frame_btn_toggled)

  def is_stream(self) -> bool:
    if self.mode == 0:
      return  self.camera.isActive()
    elif self.mode == 1:
      return self.video_player.playbackState() != QMediaPlayer.PlaybackState.PlayingState
    return False

  def on_skip_frame_btn_toggled(self, state: bool):
    self.frame_processor.setSkipIfRunning(state)

  def on_camera_btn_toggled(self, state: bool):
    self.camera_button.setText("Stop Camera" if state else "Start Camera")
    if state and not self.camera.isActive():
      self.cam_start = True
      self.camera.start()
    else:
      self.cam_start = False
      self.camera.stop()

  def on_coco_toggled(self, state: bool):
    self.worker.coco = state

  def on_process_btn_toggled(self, state: bool):
    self.worker.enabled = state
    if self.is_stream() and self.worker.has_model:
      self.frame_processor.processVideoFrame(self.video_sink.videoFrame())

  def setSource(self, url: QUrl):
    if self.mode == 1:
      self.video_player.setSource(url)

  @Slot(QCameraDevice)
  def setCamera(self, camera):
    if self.mode != 0:
      return
    self.camera = QCamera(camera)
    self.capture_session.setCamera(self.camera)
    if self.cam_start:
      self.camera.start()

  def change_mode(self, mode: int):
    self.mode = mode
    if self.mode == 0:
      self.capture_session.setVideoSink(self.video_sink)
      self.media_controller.setHidden(True)
      self.camera_button.setHidden(False)
    elif self.mode == 1:
      self.video_player.setVideoSink(self.video_sink)
      self.media_controller.setHidden(False)
      self.camera_button.setHidden(True)

  @Slot(QVideoFrame)
  def display_frame(self, frame: QVideoFrame):
    self.video_widget.videoSink().setVideoFrame(frame)

  def on_close(self):
    self.frame_processor.stop()

  def closeEvent(self, event):
    self.on_close()
    super().closeEvent(event)
