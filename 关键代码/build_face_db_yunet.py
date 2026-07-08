
import cv2

import numpy as np

import os

import sys

from hobot_dnn import pyeasy_dnn as dnn



# 路径配置

DET_MODEL_PATH = "/userdata/yunet.onnx"

REC_MODEL_PATH = "/userdata/w600k_r50.bin"

FRIEND_DIR = "/root/fly/ai/friends"

GOAL_DIR = "/root/fly/ai/goal"

OUTPUT_FILE = "/root/fly/ai/face_db.npy"



print("加载识别模型...")

rec_model = dnn.load(REC_MODEL_PATH)

detector = cv2.FaceDetectorYN.create(DET_MODEL_PATH, "", (320, 320), 0.6)



def detect_faces(img):

    h, w, _ = img.shape

    detector.setInputSize((w, h))

    _, faces = detector.detect(img)

    if faces is None:

        return []

    boxes = []

    for face in faces:

        x, y, w_box, h_box, score = face[:5]

        x1, y1, x2, y2 = int(x), int(y), int(x+w_box), int(y+h_box)

        boxes.append([x1, y1, x2, y2, score])

    return boxes



def align_face(img, box):

    x1, y1, x2, y2, _ = box

    if x1 < 0: x1 = 0

    if y1 < 0: y1 = 0

    if x2 > img.shape[1]: x2 = img.shape[1]

    if y2 > img.shape[0]: y2 = img.shape[0]

    face = img[y1:y2, x1:x2]

    if face.size == 0:

        return None

    return cv2.resize(face, (112, 112))



def extract_feature(face_img):

    img_np = face_img.astype(np.float32)

    img_np = (img_np - 127.5) * 0.007843137

    img_np = np.transpose(img_np, (2, 0, 1))

    input_tensor = np.expand_dims(img_np, axis=0)

    outputs = rec_model[0].forward(input_tensor)

    return outputs[0].buffer.flatten()



def main():

    db = {"friend": [], "goal": []}

    

    for label, folder in [("friend", FRIEND_DIR), ("goal", GOAL_DIR)]:

        if not os.path.exists(folder):

            print(f"警告：{folder} 不存在，跳过")

            continue

        for filename in os.listdir(folder):

            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):

                continue

            img_path = os.path.join(folder, filename)

            img = cv2.imread(img_path)

            if img is None:

                print(f"无法读取 {img_path}")

                continue

            boxes = detect_faces(img)

            if len(boxes) == 0:

                print(f"{img_path} 未检测到人脸，跳过")

                continue

            # 取面积最大的框

            largest = max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))

            aligned = align_face(img, largest)

            if aligned is None:

                continue

            feat = extract_feature(aligned)

            db[label].append(feat)

            print(f"已提取 {label} 特征：{filename}")

    

    np.save(OUTPUT_FILE, db, allow_pickle=True)

    print(f"特征库已保存至 {OUTPUT_FILE}")

    print(f"friend 数量: {len(db['friend'])}, goal 数量: {len(db['goal'])}")



if __name__ == "__main__":

    main()

