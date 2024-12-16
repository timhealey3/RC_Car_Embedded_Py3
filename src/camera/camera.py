from time import sleep
from picamera import PiCamera
from PIL import Image
import cv2
from io import BytesIO
import numpy as np

def img_preprocess(image):
    print("preprocessing image")
    img = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    return img

# does not save photo
def take_photo_auto(camera_ready, camera):
    if camera_ready:
        stream = BytesIO()
        image.capture(stream, format='jpeg')
        stream.seek(0)
        image = Image.open(stream)
        image_array = np.array(image)
        processed_image = img_preprocess(image_array)
    else:
        raise Exception("Camera is not ready, auto mode")

def take_photo_training(camera_ready, camera):
    if camera_ready:
        print("Photo taken")
        stream = BytesIO()
        image = camera.capture(stream, format='jpeg')
        stream.seek(0)
        image = Image.open(stream)
        image_array = np.array(image)
        processed_image = img_preprocess(image_array) 
        pilImage = Image.fromarray(processed_image)
        image.save('training/foo_rg.jpg')
    else:
        raise Exception("Camera is not ready, training mode")

def init_camera():
    camera = PiCamera()
    camera.resolution = (1024, 768)
    camera.start_preview()
    camera.shutter_speed = camera.exposure_speed
    camera.exposure_mode = 'off'
    # Camera warm-up time
    sleep(2)
    camera_ready = True
    print("Camera is ready for use")
    return camera_ready, camera
