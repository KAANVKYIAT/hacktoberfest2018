static_image_mode: It is used to specify whether the input images must be treated as static images or as a video stream. The default value is False.
model_complexity: It is used to specify the complexity of the pose landmark model: 0, 1, or 2. As the model complexity of the model increases the landmark accuracy and latency increase. The default value is 1.
smooth_landmarks: This parameter is used to reduce the jitter in the prediction by filtering pose landmarks across different input images. The default value is True.
min_detection_confidence: It is used to specify the minimum confidence value with which the detection from the person-detection model needs to be considered as successful. Can specify a value in [0.0,1.0]. The default value is 0.5.
min_tracking_confidence: It is used to specify the minimum confidence value with which the detection from the landmark-tracking model must be considered as successful. Can specify a value in [0.0,1.0]. The default value is 0.5.
STEP-3: Detecting Face and Hand landmarks from the image. Holistic model processes the image and produces landmarks for Face, Left Hand, Right Hand and also detects the Pose of the 

Capture the frames continuously from the camera using OpenCV.
Convert the BGR image to an RGB image and make predictions using initialized holistic model.
The predictions made by the holistic model are saved in the results variable from which we can access the landmarks using results.face_landmarks, results.right_hand_landmarks, results.left_hand_landmarks respectively.
Draw the detected landmarks on the image using the draw_landmarks function from drawing utils.
Display the resulting Image.
# (0) in VideoCapture is used to connect to your computer's default camera
capture = cv2.VideoCapture(0)
 
# Initializing current time and precious time for calculating the FPS
previousTime = 0
currentTime = 0
 
while capture.isOpened():
    # capture frame by frame
    ret, frame = capture.read()
 
    # resizing the frame for better view
    frame = cv2.resize(frame, (800, 600))
 
    # Converting the from BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 
    # Making predictions using holistic model
    # To improve performance, optionally mark the image as not writable to
    # pass by reference.
    image.flags.writable = False
    results = holistic_model.process(image)
    image.flags.writable = True
 
    # Converting back the RGB image to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
 
    # Drawing the Facial Landmarks
    mp_drawing.draw_landmarks(
      image,
      results.face_landmarks,
      mp_holistic.FACE_CONNECTIONS,
      mp_drawing.DrawingSpec(
        color=(255,0,255),
        thickness=1,
        circle_radius=1
      ),
      mp_drawing.DrawingSpec(
        color=(0,255,255),
        thickness=1,
        circle_radius=1
      )
    )
 
    # Drawing Right hand Land Marks
    mp_drawing.draw_landmarks(
      image,
      results.right_hand_landmarks,
      mp_holistic.HAND_CONNECTIONS
    )
 
    # Drawing Left hand Land Marks
    mp_drawing.draw_landmarks(
      image,
      results.left_hand_landmarks,
      mp_holistic.HAND_CONNECTIONS
    )
     
    # Calculating the FPS
    currentTime = time.time()
    fps = 1 / (currentTime-previousTime)
    previousTime = currentTime
     
    # Displaying FPS on the image
    cv2.putText(image, str(int(fps))+" FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
 
    # Display the resulting image
    cv2.imshow("Facial and Hand Landmarks", image)
 
    # Enter key 'q' to break the loop
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
 
# When all the process is done
# Release the capture and destroy all windows
capture.release()
cv2.destroyAllWindows()
