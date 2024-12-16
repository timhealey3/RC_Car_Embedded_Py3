import camera

def main():
    camera_ready, cameraObj = camera.init_camera()
    camera.take_photo_training(camera_ready, cameraObj)

main()
