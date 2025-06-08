from time import sleep
from picamera import PiCamera
from PIL import Image
import cv2
from io import BytesIO
import numpy as np
import pandas as pd
from datetime import datetime
import sys
sys.path.append('/home/timh/codingProjects/src/car')
import Telemetry

class Camera:
    def __init__(self):
        self.camera = PiCamera()
        self.camera.resolution = (1024, 768)
        self.camera.start_preview()
        self.camera.shutter_speed = self.camera.exposure_speed
        self.camera.exposure_mode = 'off'
        # Camera warm-up time
        sleep(2)
        self.camera_ready = True
        print("Camera is ready for use")

    # does not save photo
    def take_photo_auto(self):
        if self.camera_ready:
            self.stream = BytesIO()
            self.image.capture(self.stream, format='jpeg')
            self.stream.seek(0)
            self.image = Image.open(self.stream)
            self.image_array = np.array(self.image)
        else:
            raise Exception("Camera is not ready, auto mode")

    def calc_steering_angle(self, left, right):
        steering_angle = "0"
        if left != 0:
            steering_angle = "-1"
        elif right != 0:
            steering_angle = "1"
        return steering_angle

    def take_photo_training(self, forward, left, right):
        if self.camera_ready:
            print("Photo taken")
            self.stream = BytesIO()
            self.image = self.camera.capture(self.stream, format='jpeg')
            self.stream.seek(0)
            self.image = Image.open(self.stream)
            self.image_array = np.array(self.image)
            self.processed_image = self.img_preprocess(self.image_array)
            self.pilImage = Image.fromarray(self.processed_image)
            self.image_name = 'training_' + str(datetime.now()) +'.jpg'
            self.pilImage.save('../camera/training/' + self.image_name)
            print("start")
            self.steering_angle = self.calc_steering_angle(left, right)
            print("end")
            self.df = pd.DataFrame([[self.image_name, forward, self.steering_angle]], columns=["img", "forward", "steering_angle"])
            self.df.to_csv('../camera/training_data.csv', mode='a', index=False, header=False)
            print("Photo saved and processed")
        else:
            raise Exception("Camera is not ready, training mode")
