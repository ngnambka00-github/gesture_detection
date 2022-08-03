import mediapipe as mp
import cv2


# results: chua toa do cac diem tren khung xuong
def make_landmark_timestep(results):
    # print(results.pose_landmarks.landmark)

    c_lm = []
    for idx, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)

    return c_lm


def draw_landmark_on_image(mp_draw, results, image):
    # ve cac duong loi
    mp_draw.draw_landmarks(image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

    # ve cac diem nut tai cac vi tri skeleton
    for idx, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = image.shape
        # print(idx, lm)

        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    return image

def draw_class_on_image(label, image):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (0, 255, 0)
    thichness = 2
    lineType = 2
    cv2.putText(image, label, bottomLeftCornerOfText, font, fontScale, fontColor, thichness, lineType)

    return image


