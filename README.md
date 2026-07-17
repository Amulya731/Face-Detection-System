# Face-Detection-System
This Python script is a real-time face and gender detection application that relies on the OpenCV library (cv2). It captures a live video feed from a webcam, identifies human faces in the frame, and predicts whether each face is male or female.

Step-by-Step Breakdown :
Initialization and Setup: The script sets up paths for a pre-trained deep learning Caffe model (gender_deploy.prototxt and gender_net.caffemodel) which handles the gender classification. It then loads OpenCV's standard Haar cascade classifier (haarcascade_frontalface_default.xml) for locating faces.  

The Gender Prediction Function (get_gender): This function accepts an image of a cropped face, converts it into a 227x227 "blob" (a pre-processed format required by deep learning models), and feeds it into the Caffe neural network. The network outputs predictions, returning whichever gender—"Male" or "Female"—scores the highest probability.

Video Capture: The code initiates a video stream from the default webcam (cv2.VideoCapture(0)) and runs a continuous loop to process the feed frame-by-frame. 

Face Detection Pipeline: Inside the loop, each color frame is converted to grayscale, which optimizes the Haar cascade's ability to detect faces efficiently.  

Bounding Boxes and Labels: When faces are detected, the script isolates the facial region (the Region of Interest, or ROI) and passes it to the get_gender function. It then draws a green rectangle (cv2.rectangle) around the face and writes a text label above it indicating the result (e.g., "Male face" or "Female face").  

Display and Exit: The script displays the processed frames in a window titled "Face & Gender Detection". The program listens for a keyboard input and will safely exit the loop, release the camera, and close all windows if the 'q' key is pressed. 
