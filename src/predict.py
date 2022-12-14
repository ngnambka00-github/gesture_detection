import cv2
import os
import mediapipe as mp
from utils import make_landmark_timestep, draw_landmark_on_image, draw_class_on_image
import numpy as np
import keras
import threading

from hydra import initialize, compose
from omegaconf import OmegaConf

import warnings
warnings.filterwarnings('ignore')

with initialize(config_path="../configs/"):
    data_cfg = compose(config_name="data_path")
data_cfg = OmegaConf.create(data_cfg)
HOME_PATH = "../"
best_model_path = os.path.join(HOME_PATH, data_cfg.model.best_model)

# Doc anh tu webcame
cap = cv2.VideoCapture(0)

# Khoi tao thu vien mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

# load model
model = keras.models.load_model(best_model_path)

label = "Initialization ... "
skeleton_landmark_list = []
no_of_frames = 1000
n_time_steps = 10

def detect(model, lm_list):
    global label
    label_list = ["BODY_SWING", "HAND_LEFT_SWING", "HAND_RIGHT_SWING", "HAND_TWO_SWING"]
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)

    results = model.predict(lm_list)[0]
    idx_max_confident = results.argmax()
    label = label_list[idx_max_confident]
    
    return label

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

            if len(skeleton_landmark_list) == n_time_steps:
                # dua vao model nhan dien 
                t1 = threading.Thread(target=detect, args=(model, skeleton_landmark_list))
                t1.start()
                skeleton_landmark_list = []

    if ret: 
        # ve khung xuong len anh
        frame = draw_landmark_on_image(mpDraw, results_skeleton, frame)
        frame = draw_class_on_image(label, frame)
        cv2.imshow("image", frame)

        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()