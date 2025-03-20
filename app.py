import streamlit as st
import cv2
import numpy as np
import pytesseract
import pandas as pd
import os

# Load YOLO model
weights_path = r"C:\Users\User\internship projects\Project 10\Custom_OCR\yolo_config\yolov3.weights"
cfg_path = r"C:\Users\User\internship projects\Project 10\Custom_OCR\yolo_config\yolov3.cfg"

net = cv2.dnn.readNet(weights_path, cfg_path)


# Function to detect objects and extract text
def process_image(image):
    image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)
    height, width, _ = image.shape

    # Prepare image for YOLO
    blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    detections = net.forward(output_layers)

    extracted_text = []
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x, center_y, w, h = (obj[:4] * [width, height, width, height]).astype(int)
                x, y = center_x - w // 2, center_y - h // 2

                # Crop detected region
                cropped = image[y:y+h, x:x+w]

                # Preprocess for Tesseract
                gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (3,3), 0)
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Extract text
                text = pytesseract.image_to_string(thresh, config="--psm 6")
                extracted_text.append({"Detected Region": f"({x}, {y}, {w}, {h})", "Extracted Text": text.strip()})

    return extracted_text

# Streamlit UI
st.title("Custom OCR: Extract Text from Lab Reports")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Processing..."):
        extracted_data = process_image(uploaded_file)

    if extracted_data:
        df = pd.DataFrame(extracted_data)
        st.write("### Extracted Text in Table Format:")
        st.dataframe(df)

        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=csv, file_name="extracted_text.csv", mime="text/csv")

    else:
        st.warning("No text detected. Try another image.")
        
        
        
image_path = "/content/drive/MyDrive/Custom_OCR/test.jpg"
net = cv2.dnn.readNet(weights_path, cfg_path)
image = cv2.imread(image_path)
height, width = image.shape[:2]
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
detections = net.forward(output_layers)

if len(detections) == 0:
    print("No objects detected!")
else:
    print("Objects detected:", len(detections))
