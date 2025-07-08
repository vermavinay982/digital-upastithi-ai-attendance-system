import cv2
import mediapipe as mp
import os 

# pip install mediapipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

"""
[label_id: 0
score: 0.727693319
location_data {
    format: RELATIVE_BOUNDING_BOX
    relative_bounding_box {
        xmin: 0.386143744
        ymin: 0.125001788
        width: 0.218552649
        height: 0.152030259
    }
    relative_keypoints {
        x: 0.450964689
        y: 0.170615926
    }
    relative_keypoints {
        x: 0.543260157
        y: 0.16806753
    }
    relative_keypoints {
        x: 0.491054177
        y: 0.197286278
    }
    relative_keypoints {
        x: 0.493746459
        y: 0.230891243
    }
    relative_keypoints {
        x: 0.406727493
        y: 0.197568938
    }
    relative_keypoints {
        x: 0.607000053
        y: 0.192785621
    }
}
]
"""

def crop_face(image, relative_bounding_box):
    h, w, c = image.shape
    xmin, ymin, width, height = relative_bounding_box.xmin, relative_bounding_box.ymin, relative_bounding_box.width, relative_bounding_box.height
    # print(xmin, ymin, width, height)
    xmin, ymin, width, height = int(xmin*w), int(ymin*h), int(width*w), int(height*h)
    # print(xmin, ymin, width, height)
    crop = image[ymin:ymin+height, xmin:xmin+width]
    cv2.imwrite('test.png', crop)
    return crop

def detect_faces(np_images=[]):
    # For static images:
    op_folder = 'op/'
    os.makedirs(op_folder, exist_ok=True)
    faces = []
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.0) as face_detection:
        for idx, image in enumerate(np_images):
            # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
            results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Draw face detections of each face.
            if not results.detections:
                print("No Faces Found")
                faces.append([])
                continue

            for detection in results.detections:
                crop = crop_face(image, detection.location_data.relative_bounding_box)
                faces.append(crop)

            annotated_image = image.copy()
            for detection in results.detections:
                print("Face Found")
                # print('Nose tip:')
                # print(mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
                mp_drawing.draw_detection(annotated_image, detection)
            cv2.imwrite(op_folder + str(idx) + '.png', annotated_image)
    return faces

def detect_faces_video():
    # For webcam input:
    cap = cv2.VideoCapture(0)
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image)

            # Draw the face detection annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(image, detection)
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()

if __name__=="__main__":
    detect_faces_video()