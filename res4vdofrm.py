import cv2
import mediapipe as mp
import numpy as np
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
import sys
import matplotlib.pyplot as plt

# MediaPipe Face Detection and Face Mesh modules
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

class FaceDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Real-time Face Detection and Face Mesh with Respiration Rate")
        self.setFixedSize(800, 600)  
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)
        self.video_label = QLabel(self)
        self.layout.addWidget(self.video_label)

        # Set the window background color to white 
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(255, 255, 255)) 
        self.setPalette(palette)
        self.face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        self.face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.capture = cv2.VideoCapture(0)

        #  respiration rate variables
        self.respiration_rates = []
        self.time_points = []
        self.plot_initialized = False

    def update_frame(self):
        ret, frame = self.capture.read()

        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Perform face detection
            results_detection = self.face_detection.process(frame_rgb)
            if results_detection.detections:
                for detection in results_detection.detections:
                    ih, iw, _ = frame_rgb.shape
                    bboxC = detection.location_data.relative_bounding_box
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                    # Calculate ROI coordinates for forehead and cheeks
                    forehead_x = x
                    forehead_y = y
                    forehead_w = w
                    forehead_h = int(h * 0.25)  

                    cheek_x = x
                    cheek_y = y + int(h * 0.75)  
                    cheek_w = w
                    cheek_h = int(h * 0.25) 

                    # Extraction  ROI for forehead and cheeks
                    forehead_roi = frame_rgb[forehead_y:forehead_y + forehead_h, forehead_x:forehead_x + forehead_w]
                    cheek_roi = frame_rgb[cheek_y:cheek_y + cheek_h, cheek_x:cheek_x + cheek_w]

                    # Calculating  the RGB values for the ROIs
                    forehead_rgb_mean = np.mean(forehead_roi, axis=(0, 1))
                    cheek_rgb_mean = np.mean(cheek_roi, axis=(0, 1))

                    # Calculating the respiration rate
                    respiration_rate = np.mean(forehead_rgb_mean) - np.mean(cheek_rgb_mean)
                    self.respiration_rates.append(respiration_rate)
                    self.time_points.append(len(self.respiration_rates))

                    # graph
                    if self.plot_initialized:
                        self.ax.clear()
                    else:
                        plt.ion()  #  plotting
                        self.fig, self.ax = plt.subplots()
                        self.ax.set_xlabel('Time')
                        self.ax.set_ylabel('Respiration Rate')
                        self.plot_initialized = True

                    self.ax.plot(self.time_points, self.respiration_rates, marker='o', linestyle='-')
                    self.ax.set_title('Real-time Respiration Rate')
                    self.ax.set_xlim(0, max(10, len(self.respiration_rates)))
                    self.ax.set_ylim(min(0, min(self.respiration_rates)), max(1, max(self.respiration_rates)))

                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()

        
            h, w, c = frame.shape
            window_h, window_w = self.video_label.height(), self.video_label.width()
            aspect_ratio = w / h  # Calculate the aspect ratio
            if aspect_ratio > (window_w / window_h):
                new_w = window_w
                new_h = int(window_w / aspect_ratio)
            else:
                new_h = window_h
                new_w = int(window_h * aspect_ratio)
            frame_resized = cv2.resize(frame, (new_w, new_h))
            
            
            frame_with_fixed_size = np.zeros((window_h, window_w, 3), dtype=np.uint8)
            
            # Calculate the position to paste the resized frame
            y_offset = (window_h - new_h) // 2
            x_offset = (window_w - new_w) // 2
            frame_with_fixed_size[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = frame_resized

            # BGR format
            bytes_per_line = 3 * window_w
            q_image = QImage(frame_with_fixed_size.data, window_w, window_h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image.rgbSwapped())  
            self.video_label.setPixmap(pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceDetectionApp()
    window.show()
    sys.exit(app.exec_())
