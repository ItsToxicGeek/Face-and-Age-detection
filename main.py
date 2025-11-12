import cv2
import math

# Paths to model files
faceProto = "deploy.prototxt"
faceModel = "res10_300x300_ssd_iter_140000_fp16.caffemodel"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

# Load network models
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Model mean values and label lists
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-10)', '(10-15)', '(15-17)', '(17-20)', '(20-22)', '(22-25)', '(25-28)', '(28-30)',
            '(30-40)', '(40-60)', '(60-100)']
genderList = ['Male', 'Female']

# Helper function for face detection
def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight, frameWidth = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2),
                          (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, bboxes


# --- Create window that always stays on top ---
cv2.namedWindow("Age and Gender Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Age and Gender Detection", cv2.WND_PROP_TOPMOST, 1)

# Start video capture
video = cv2.VideoCapture(0)
padding = 20

while True:
    ret, frame = video.read()
    if not ret:
        break

    frameFace, bboxes = getFaceBox(faceNet, frame)
    if not bboxes:
        cv2.putText(frameFace, "No face detected", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    for bbox in bboxes:
        face = frame[max(0, bbox[1]-padding):min(bbox[3]+padding, frame.shape[0]-1),
                     max(0, bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                     MODEL_MEAN_VALUES, swapRB=False)
        
        # Predict gender
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        # Predict age
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        label = f"{gender}, {age}"
        cv2.putText(frameFace, label, (bbox[0], bbox[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    # Display output
    cv2.imshow("Age and Gender Detection", frameFace)

    # Press 'q' or click close (X) to exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty("Age and Gender Detection", cv2.WND_PROP_VISIBLE) < 1:
        print("Program ended safely.")
        break

# Cleanup
video.release()
cv2.destroyAllWindows()
