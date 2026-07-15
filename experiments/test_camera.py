import cv2


for index in range(5):

    camera = cv2.VideoCapture(index)

    if camera.isOpened():
        print(f"Camera found at index {index}")
        camera.release()

    else:
        print(f"No camera at index {index}")