import cv2
import dlib
import face_recognition



detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:\\Users\\Dell 7410\\Desktop\\intermediate project\\shape_predictor_68_face_landmarks.dat")

known_faces = {
    "Person1": face_recognition.load_image_file("C:\\Users\\Dell 7410\\Desktop\\intermediate project\\elon.jpg"),
    "Person2": face_recognition.load_image_file("C:\\Users\\Dell 7410\\Desktop\\intermediate project\\mark.jpg"),
    # Add more known faces as needed
}
# Function to perform face recognition
def recognize_face(frame):
    # Convert the frame to RGB format (required by face_recognition library)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find faces in the frame using Dlib
    faces = detector(rgb_frame)

    # Iterate through each detected face
    for face in faces:
        # Get facial landmarks
        landmarks = face_recognition.face_landmarks(rgb_frame)

        # Draw a rectangle around the detected face
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw facial landmarks on the frame
        for landmark_type, landmarks_list in landmarks[0].items():
            for landmark in landmarks_list:
                cv2.circle(frame, landmark, 2, (255, 0, 0), -1)

    return frame


video_path = "C:\\Users\\Dell 7410\\Desktop\\intermediate project\\WhatsApp Video 2024-03-12 at 4.47.15 PM.mp4"
cap = cv2.VideoCapture(video_path)


#C:\Users\Dell 7410\Desktop\intermediate project\WhatsApp Video 2024-03-12 at 4.47.15 PM.mp4
while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    if not ret:
        # Break the loop if the video is over or an error occurs
        break

    # Perform face recognition on the frame
    frame = recognize_face(frame)

    # Display the frame
    cv2.imshow("Facial Recognition", frame)
    k = cv2.waitKey(1)
    if k == 27:  # Press 'ESC' to exit
      break

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV window
cap.release()
cv2.destroyAllWindows()