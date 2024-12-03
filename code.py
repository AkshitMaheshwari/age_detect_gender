import cv2
import streamlit as st
import numpy as np


def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes


# Model files
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load networks
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Streamlit UI
st.title("Age and Gender Detection")

# Choose detection mode
mode = st.sidebar.selectbox("Choose a mode", ("Real-Time Detection", "Image Upload"))

if mode == "Real-Time Detection":
    run = st.checkbox('Start Detection')
    quit_app = st.button('Quit')

    if run:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera not accessible.")
                break

            resultImg, faceBoxes = highlightFace(faceNet, frame)
            if not faceBoxes:
                st.warning("No face detected")

            for faceBox in faceBoxes:
                face = frame[max(0, faceBox[1] - 20):min(faceBox[3] + 20, frame.shape[0] - 1),
                             max(0, faceBox[0] - 20):min(faceBox[2] + 20, frame.shape[1] - 1)]

                blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                genderNet.setInput(blob)
                genderPreds = genderNet.forward()
                gender = genderList[genderPreds[0].argmax()]

                ageNet.setInput(blob)
                agePreds = ageNet.forward()
                age = ageList[agePreds[0].argmax()]

                cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            stframe.image(cv2.cvtColor(resultImg, cv2.COLOR_BGR2RGB), channels="RGB")

            if quit_app:
                break

        cap.release()
        st.write("Detection stopped.")

elif mode == "Image Upload":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)  # Decode the image

        resultImg, faceBoxes = highlightFace(faceNet, image)
        if not faceBoxes:
            st.warning("No face detected")

        for faceBox in faceBoxes:
            face = image[max(0, faceBox[1] - 20):min(faceBox[3] + 20, image.shape[0] - 1),
                         max(0, faceBox[0] - 20):min(faceBox[2] + 20, image.shape[1] - 1)]

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]

            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]

            cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        st.image(cv2.cvtColor(resultImg, cv2.COLOR_BGR2RGB), channels="RGB", caption="Processed Image")

st.write("Select a mode from the sidebar to start.")
