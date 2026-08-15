import cv2

print("OpenCV:", cv2.__version__)
print("cv2 location:", cv2.__file__)
print("CascadeClassifier:", hasattr(cv2, "CascadeClassifier"))

print("haar path:", cv2.data.haarcascades)

detector_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

face_detector = cv2.CascadeClassifier(detector_path)

print("Detector created:", face_detector)
print("Detector empty:", face_detector.empty())