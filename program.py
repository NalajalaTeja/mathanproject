#suspect detecting
import os  
import cv2   #import modules
import face_recognition
import datetime
import winsound

TOLERANCE = 0.5
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "cnn"
video = cv2.VideoCapture(0)

known_faces = []
known_names = []
for file in os.listdir("known"):  #known folder
    # image = read_img("known/" + file)
    image = face_recognition.load_image_file("known/" + file)
    encoding = face_recognition.face_encodings(image)[0]
    known_faces.append(encoding)
    known_names.append(file.split('.')[0])

while True:

    ret, image = video.read()
    locations = face_recognition.face_locations(image, model="MODEL")
    encoding = face_recognition.face_encodings(image, locations)

    for face_encoding, (top,right,bottom,left) in zip(encoding, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        match = "Unknown"
        if True in results:
            match = known_names[results.index(True)]
            print(f"Match found:  {match}")
            winsound.Beep(500,500)
            print(datetime.datetime.now())

        # Draw a box around the face
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), FRAME_THICKNESS)

        # Draw a label with a name below the face
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, match, (left + 6, bottom - 6), font, 1.0, (50, 100, 200), 1)

    cv2.imshow(file, image)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
cv2.destroyAllWindows()
video.release()