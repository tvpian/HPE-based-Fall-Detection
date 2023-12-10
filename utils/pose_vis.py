import mediapipe as mp
import os
import cv2

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

mp_drawing = mp.solutions.drawing_utils
drawing_styles = mp.solutions.drawing_styles

dataset_folder = "./data/fall-01-cam0-rgb"

for root, dirs, files in os.walk(dataset_folder):
    file_list = sorted(files)
    for file_name in file_list:
        file_path = os.path.join(root, file_name)

        if not file_path.endswith(".png"):
            continue

        frame = cv2.imread(file_path)
        results = pose.process(frame)

        annotated_frame = frame.copy()
        mp_drawing.draw_landmarks(
            annotated_frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style(),
        )

        cv2.imshow("Pose Estimation", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
