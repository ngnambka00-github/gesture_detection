import cv2
import os
import mediapipe as mp
import pandas as pd
from utils import make_landmark_timestep, draw_landmark_on_image, draw_class_on_image

from hydra import initialize, compose
from omegaconf import OmegaConf

import warnings
warnings.filterwarnings('ignore')

with initialize(config_path="../configs/"):
    data_cfg = compose(config_name="data_path")
data_cfg = OmegaConf.create(data_cfg)
HOME_PATH = "../"

body_swing_data_path = os.path.join(HOME_PATH, data_cfg.data.body_swing)
hand_left_swing_data_path = os.path.join(HOME_PATH, data_cfg.data.hand_left_swing)
hand_right_swing_data_path = os.path.join(HOME_PATH, data_cfg.data.hand_right_swing)
hand_two_swing_data_path = os.path.join(HOME_PATH, data_cfg.data.hand_two_swing)

# Doc anh tu webcame
cap = cv2.VideoCapture(0)

# Khoi tao thu vien mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

skeleton_landmark_list = []
labels_path = body_swing_data_path
# labels_path = hand_left_swing_data_path
# labels_path = hand_right_swing_data_path
# labels_path = hand_two_swing_data_path

no_of_frames = 1000
i = 0
warmup_frame = 60 # co them thoi gian de chua bi cho camera


while True: 
    ret, frame = cap.read()

    # Nhan dien pose
    frame_rgb = None
    results_skeleton = None
    if ret: 
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_skeleton = pose.process(frame_rgb)

    i = i + 1
    if i > warmup_frame and ret: 

        if results_skeleton.pose_landmarks:
            # ghi nhan thong so khung xuong
            lm = make_landmark_timestep(results_skeleton)
            skeleton_landmark_list.append(lm)

            # ngung viec thu thap du lieu
            if len(skeleton_landmark_list) >= no_of_frames: 
                break

            # ve khung xuong len anh
            frame = draw_landmark_on_image(mpDraw, results_skeleton, frame)
            frame = draw_class_on_image("Start collecting data", frame)

    if i <= warmup_frame and ret:
        frame = draw_class_on_image("Initialization ...", frame)

    if ret: 
        cv2.imshow("image", frame)
        if cv2.waitKey(1) == ord('q'):
            break

# write vao file csv
df = pd.DataFrame(skeleton_landmark_list)
df.to_csv(labels_path)

cap.release()
cv2.destroyAllWindows()