import cv2

class Camera:

    def __init__(self, index: int = 0):
        self.camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)

        if not self.camera.isOpened():
            raise RuntimeError(f"Cannot open camera {index}")

    def read(self):
        success, frame = self.camera.read()

        if not success:
            return None

        return frame

    def release(self):
        self.camera.release()