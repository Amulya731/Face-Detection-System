import cv2
import os

# --- Configuration ---
# Ensure these files are in the same folder as this script
proto_path = r"C:\Users\hp\OneDrive\Desktop\internship proj\second month project\face detection system\gender_deploy.prototxt"
model_path = r"C:\Users\hp\OneDrive\Desktop\internship proj\second month project\face detection system\gender_net.caffemodel"

# --- Initialization ---
if not os.path.exists(proto_path) or not os.path.exists(model_path):
    print(f"Error: Model files not found. Please ensure {proto_path} and {model_path} exist.")
    exit()

# Load the cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the gender detection deep learning model
gender_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
GENDER_LIST = ['Male', 'Female']

def get_gender(face_img):
    # Preprocess the face image for the model
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.426, 87.769, 114.89), swapRB=False)
    gender_net.setInput(blob)
    preds = gender_net.forward()
    return GENDER_LIST[preds[0].argmax()]

# --- Main Video Loop ---
cap = cv2.VideoCapture(0)

print("Starting detection... Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    
    for (x, y, w, h) in faces:
        # Extract the face region
        face_roi = frame[y:y+h, x:x+w]
        
        # Predict the gender
        gender = get_gender(face_roi)
        
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Display the gender label with "face" appended
        label = f"{gender} face"
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show the resulting frame
    cv2.imshow('Face & Gender Detection', frame)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()