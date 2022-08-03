import cv2
import mediapipe as mp
import pandas as pd
from utils import make_landmark_timestep, draw_landmark_on_image

# Doc anh tu webcame
cap = cv2.VideoCapture(0)

# Khoi tao thu vien mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

skeleton_landmark_list = []
labels = "HANDSWING"
no_of_frames = 1000

while len(skeleton_landmark_list) <= no_of_frames: 
    ret, frame = cap.read()

    if ret: 
        # Nhan dien pose
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_skeleton = pose.process(frame_rgb)

        if results_skeleton.pose_landmarks:
            # ghi nhan thong so khung xuong
            lm = make_landmark_timestep(results_skeleton)
            skeleton_landmark_list.append(lm)

            # ve khung xuong len anh
            frame = draw_landmark_on_image(mpDraw, results_skeleton, frame)

        cv2.imshow("image", frame)

        if cv2.waitKey(1) == ord('q'):
            break

# write vao file csv
# df = pd.DataFrame(skeleton_landmark_list)
# df.to_csv(f"../data/{labels}.txt")

cap.release()
cv2.destroyAllWindows()