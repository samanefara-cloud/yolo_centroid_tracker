# با ویدیو رایتر برای ذخیره کردن ویدیو
import cv2
import numpy as np
import threading
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from ultralytics import YOLO
from distance_centroid_tracker import DistanceCentroidTracker


class YOLOCentroidTrackerApp:
    def __init__(self, source=0, fps=40, model_name="yolov8n.pt", conf_threshold=0.5):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise Exception(f"Error: Cannot open source: {source}")
        
        print(f"Loading YOLO model: {model_name} ...")
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        print(f"YOLO model loaded (threshold: {conf_threshold})")
        
        self.tracker = DistanceCentroidTracker(max_disappeared=10, max_distance=100)
        
        self.fps = fps
        self.interval = 1.0 / fps
        self.timer = None
        self.running = True
        
        self.current_frame = None
        self.frame_height = 0
        self.frame_width = 0
        self.tracked_objects = {}
        
        # ========== ویدیو خروجی ==========
        self.output_path = "output_video.mp4"
        self.video_writer = None
        
        plt.ion()
        self.fig, (self.ax_video, self.ax_info) = plt.subplots(1, 2, figsize=(14, 6))
        self.fig.canvas.manager.set_window_title("YOLO + Centroid Tracker with Noise Filter")
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        self.ax_video.set_title("YOLO Detection + Centroid Tracking")
        self.ax_info.set_title("Information Panel")
        self.ax_info.axis('off')
        
        print("=" * 50)
        print("YOLO + Centroid Tracker with Noise Filter")
        print(f"Source: {source}")
        print(f"FPS: {fps}")
        print(f"Model: {model_name}")
        print(f"Confidence threshold: {conf_threshold}")
        print("Press 'q' to quit")
        print("=" * 50)
    
    def on_key(self, event):
        if event.key == 'q':
            print("Exiting...")
            self.running = False
            if self.timer:
                self.timer.cancel()
            plt.close()
    
    def draw_tracked_boxes(self):
        colors = plt.cm.tab20.colors
        for obj_id, bbox in self.tracked_objects.items():
            x, y, w, h = bbox
            color = colors[obj_id % len(colors)]
            
            rect = Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
            self.ax_video.add_patch(rect)
            
            self.ax_video.text(x, y - 5, f"ID:{obj_id}", fontsize=9, color=color,
                             weight='bold', bbox=dict(facecolor='black', alpha=0.6, pad=1))
            
            cx = x + w // 2
            cy = y + h // 2
            self.ax_video.plot(cx, cy, 'ro', markersize=3)
    
    def draw_info_panel(self):
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        info_text = f"""
YOLO + Centroid Tracker with Noise Filter
----------------------------------------
Tracked objects: {len(self.tracked_objects)}
Total registered: {self.tracker.next_object_id}
Target FPS: {self.fps}
Confidence threshold: {self.conf_threshold}

Active objects:
"""
        if len(self.tracked_objects) == 0:
            info_text += "  (No objects detected)\n"
        else:
            for obj_id, bbox in list(self.tracked_objects.items())[:8]:
                x, y, w, h = bbox
                info_text += f"  ID {obj_id}: ({x}, {y}, {w}, {h})\n"
            if len(self.tracked_objects) > 8:
                info_text += f"  ... and {len(self.tracked_objects)-8} more\n"
        
        info_text += """
Keys:
  q : quit
"""
        self.ax_info.text(0.5, 0.95, info_text, transform=self.ax_info.transAxes,
                         fontsize=10, verticalalignment='top', horizontalalignment='center',
                         fontfamily='monospace', bbox=dict(facecolor='lightgray', alpha=0.9))
        self.ax_info.set_xlim(0, 1)
        self.ax_info.set_ylim(0, 1)
    
    # ===============================================
    # Noise filtering functions
    # ===============================================
    def iou(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        ax1, ay1, ax2, ay2 = x1, y1, x1 + w1, y1 + h1
        bx1, by1, bx2, by2 = x2, y2, x2 + w2, y2 + h2
        
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        
        if inter_x2 < inter_x1 or inter_y2 < inter_y1:
            return 0.0
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        
        return inter_area / (box1_area + box2_area - inter_area)
    
    def nms(self, detections, iou_threshold=0.5):
        if len(detections) == 0:
            return []
        
        filtered = []
        for box in detections:
            is_duplicate = False
            for kept in filtered:
                if self.iou(box, kept) > iou_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered.append(box)
        return filtered
    
    def filter_boundary_boxes(self, detections, margin=1):
        filtered = []
        for x, y, w, h in detections:
            if x < margin or y < margin:
                continue
            if x + w > self.frame_width - margin or y + h > self.frame_height - margin:
                continue
            filtered.append((x, y, w, h))
        return filtered
    
    def filter_by_size(self, detections, min_area=100, max_area_ratio=0.95):
        filtered = []
        max_area = self.frame_width * self.frame_height * max_area_ratio
        for x, y, w, h in detections:
            area = w * h
            if area < min_area or area > max_area:
                continue
            filtered.append((x, y, w, h))
        return filtered
    
    def filter_by_aspect_ratio(self, detections, min_ratio=0.2, max_ratio=5):
        filtered = []
        for x, y, w, h in detections:
            if h == 0:
                continue
            ratio = w / h
            if ratio < min_ratio or ratio > max_ratio:
                continue
            filtered.append((x, y, w, h))
        return filtered
    
    def advanced_noise_filter(self, detections):
        if len(detections) == 0:
            return detections
        
        result = detections
        result = self.filter_boundary_boxes(result, margin=1)
        result = self.filter_by_size(result, min_area=100, max_area_ratio=0.95)
        result = self.filter_by_aspect_ratio(result, min_ratio=0.2, max_ratio=5)
        result = self.nms(result, iou_threshold=0.5)
        return result
    
    # ===============================================
    # Main update loop
    # ===============================================
    def update_frame(self):
        if not self.running:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            print("Video ended or camera disconnected")
            self.running = False
            if self.timer:
                self.timer.cancel()
            return
        
        self.current_frame = frame
        self.frame_height, self.frame_width = frame.shape[:2]
        
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        detections_raw = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    w = x2 - x1
                    h = y2 - y1
                    detections_raw.append((x1, y1, w, h))
        
        detections = self.advanced_noise_filter(detections_raw)
        self.tracked_objects = self.tracker.update(detections)
        
        # ========== رسم روی فریم برای ذخیره ویدیو ==========
        frame_display = frame.copy()
        for obj_id, bbox in self.tracked_objects.items():
            x, y, w, h = bbox
            cv2.rectangle(frame_display, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(frame_display, f"ID:{obj_id}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cx = x + w // 2
            cy = y + h // 2
            cv2.circle(frame_display, (cx, cy), 3, (0, 0, 255), -1)
        
        cv2.putText(frame_display, f"Objects: {len(self.tracked_objects)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # ========== ذخیره در فایل ویدیویی ==========
        if self.video_writer:
            self.video_writer.write(frame_display)
        
        # ========== نمایش در matplotlib ==========
        frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
        self.ax_video.clear()
        self.ax_video.imshow(frame_rgb)
        self.ax_video.set_title(f"YOLO + Centroid Tracker ({len(self.tracked_objects)} objects)")
        self.ax_video.set_xlim(0, self.frame_width)
        self.ax_video.set_ylim(self.frame_height, 0)
        
        self.draw_tracked_boxes()
        self.draw_info_panel()
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        if self.running:
            self.timer = threading.Timer(self.interval, self.update_frame)
            self.timer.start()
    
    def run(self):
        try:
            ret, first_frame = self.cap.read()
            if ret:
                self.frame_height, self.frame_width = first_frame.shape[:2]
            
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # مقداردهی VideoWriter برای ذخیره خروجی
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.output_path, fourcc, self.fps,
                                                (self.frame_width, self.frame_height))
            print(f"✅ Video will be saved to: {self.output_path}")
            
            self.timer = threading.Timer(self.interval, self.update_frame)
            self.timer.start()
            
            plt.ioff()
            plt.show(block=True)
        
        except Exception as e:
            print(f"Error: {e}")
        
        finally:
            print("Cleaning up...")
            if hasattr(self, 'cap') and self.cap.isOpened():
                self.cap.release()
            if self.timer:
                self.timer.cancel()
            if self.video_writer:
                self.video_writer.release()
                print(f"✅ Video saved: {self.output_path}")
            cv2.destroyAllWindows()
            plt.close('all')
            print("Program finished")


if __name__ == "__main__":
    try:
        app = YOLOCentroidTrackerApp(source=0, fps=30, model_name="yolov8n.pt", conf_threshold=0.5)
        app.run()
    
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    
    except Exception as e:
        print(f"Unexpected error: {e}")