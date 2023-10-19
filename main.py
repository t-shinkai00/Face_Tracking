import cv2
import mediapipe as mp


class FaceDetector:
    def __init__(self):
        self.model = mp.solutions.face_mesh.FaceMesh()

    def detect(self, image):
        return self.model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = FaceDetector()

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            print("Ignoring empty frame")
            continue
        img.flags.writeable = False  # Mark the image as not writeable to pass by reference.
        img = cv2.flip(img, 1)
        res = detector.detect(img)

        img.flags.writeable = True  # Draw the face detection annotations on the image.
        # print(res.multi_face_landmarks)
        if res.multi_face_landmarks:
            for landmark in res.multi_face_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    image=img,
                    landmark_list=landmark,
                    connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1),
                    connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
                )

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()