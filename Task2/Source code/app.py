import cv2
import time
import numpy as np
from ultralytics import YOLO


class VideoStream:
    def __init__(self, source=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


class FrameSkipper:
    def __init__(self, skip_n=3):
        self.skip_n = max(1, skip_n)
        self.counter = 0

    def should_process(self):
        self.counter += 1
        if self.counter >= self.skip_n:
            self.counter = 0
            return True
        return False


class PersonTracker:
    def __init__(self, model_path="yolov8n.onnx", conf=0.5, device="cpu",
                 tracker_cfg="bytetrack.yaml", person_class_id=0):
        self.model = YOLO(model_path, task="detect")
        self.conf = conf
        self.device = device
        self.tracker_cfg = tracker_cfg
        self.person_class_id = person_class_id

    def track(self, frame):
        results = self.model.track(
            frame,
            conf=self.conf,
            device=self.device,
            persist=True,
            classes=[self.person_class_id],
            tracker=self.tracker_cfg,
            verbose=False,
        )
        return results[0]

    @staticmethod
    def extract_tracks(result):
        tracks = []
        if result.boxes is None or result.boxes.id is None:
            return tracks

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()

        for box, track_id, conf in zip(boxes, ids, confs):
            x1, y1, x2, y2 = box.astype(int)
            tracks.append({
                "id": int(track_id),
                "bbox": (x1, y1, x2, y2),
                "conf": float(conf),
            })
        return tracks


class Annotator:
    def __init__(self, box_color=(0, 255, 0), text_color=(255, 255, 255)):
        self.box_color = box_color
        self.text_color = text_color

    def draw_tracks(self, frame, tracks):
        for t in tracks:
            x1, y1, x2, y2 = t["bbox"]
            track_id = t["id"]
            conf = t["conf"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), self.box_color, 2)

            label = f"ID:{track_id} Person {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), self.box_color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.text_color, 2)

        return frame

    @staticmethod
    def draw_count(frame, count):
        cv2.putText(
            frame,
            f"Persons Tracked: {count}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        return frame


class FPSMeter:
    def __init__(self):
        self.prev_time = time.time()
        self.fps = 0.0

    def update(self):
        current_time = time.time()
        elapsed = current_time - self.prev_time
        self.prev_time = current_time
        if elapsed > 0:
            self.fps = 1.0 / elapsed
        return self.fps

    def draw(self, frame):
        cv2.putText(
            frame,
            f"FPS: {self.fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        return frame


class App:
    def __init__(self, source=0, skip_n=3, model_path="yolov8n.onnx",
                 conf=0.5, device="cpu", tracker_cfg="bytetrack.yaml"):
        self.stream = VideoStream(source=source)
        self.skipper = FrameSkipper(skip_n=skip_n)
        self.tracker = PersonTracker(
            model_path=model_path,
            conf=conf,
            device=device,
            tracker_cfg=tracker_cfg,
            person_class_id=0,
        )
        self.annotator = Annotator()
        self.fps_meter = FPSMeter()
        self.window_name = "Person Detection & ByteTrack Tracking"
        self.last_display_frame = None

    def run(self):
        while True:
            ret, frame = self.stream.read()
            if not ret:
                break

            if self.skipper.should_process():
                result = self.tracker.track(frame)
                tracks = self.tracker.extract_tracks(result)
                display_frame = frame.copy()
                display_frame = self.annotator.draw_tracks(display_frame, tracks)
                display_frame = self.annotator.draw_count(display_frame, len(tracks))
                self.last_display_frame = display_frame
            else:
                display_frame = frame.copy() if self.last_display_frame is not None else frame

            self.fps_meter.update()
            display_frame = self.fps_meter.draw(display_frame)

            cv2.putText(
                display_frame,
                "Press Q to Quit",
                (10, display_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

            cv2.imshow(self.window_name, display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cleanup()

    def cleanup(self):
        self.stream.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = App(
        source=0,
        skip_n=3,
        model_path="yolov8n.pt",
        conf=0.5,
        device="cpu",
        tracker_cfg="bytetrack.yaml",
    )
    app.run()