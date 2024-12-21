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

    def img_preprocess(self, image):
        print("preprocessing image")
        self.img = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        return self.img

    # does not save photo
    def take_photo_auto(self):
        if self.camera_ready:
            self.stream = BytesIO()
            self.image.capture(self.stream, format='jpeg')
            self.stream.seek(0)
            self.image = Image.open(self.stream)
            self.image_array = np.array(self.image)
            self.processed_image = self.img_preprocess(self.image_array)
        else:
            raise Exception("Camera is not ready, auto mode")

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
            self.image_name = '../camera/training/foo' + str(datetime.now()) +'.jpg'
            self.image.save(self.image_name)
            self.df = pd.DataFrame({self.image_name, forward, left, right})
            self.df.to_csv('training_data.csv', index=False)
        else:
            raise Exception("Camera is not ready, training mode")

