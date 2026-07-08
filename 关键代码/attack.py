import cv2
import numpy as np
import time
import os
import threading
from hobot_vio import libsrcampy
from hobot_dnn import pyeasy_dnn as dnn
import Hobot.GPIO as GPIO

# -------------------------- 人脸识别配置 --------------------------
DET_MODEL_PATH = "/userdata/yunet.onnx"
REC_MODEL_PATH = "/userdata/w600k_r50.bin"
DB_PATH = "/root/fly/ai/face_db.npy"
CAM_WIDTH = 1920
CAM_HEIGHT = 1080
INTERVAL = 2  # 人脸识别检测间隔（秒）
THRESHOLD = 0.5  # 相似度阈值

# -------------------------- 舵机配置 --------------------------
PWM_CHIP = 0          # pwmchip0
PWM_CHANNEL = 1       # pwm1，对应物理引脚33
PERIOD_NS = 20000000  # 50Hz = 20ms (舵机标准频率)
# 舵机动作参数：0°=0.5ms(初始位)，90°=1.5ms(旋转90°)，180°=2.5ms
SERVO_INIT_ANGLE = 0      # 舵机初始角度
SERVO_ROTATE_ANGLE = 90   # 识别到目标后旋转的角度
SERVO_RESPONSE_TIME = 0.15 # 舵机最大速度下转动到位的最小等待时间（可根据舵机型号调整）

# -------------------------- LED爆闪配置 --------------------------
LED_PIN = 37          # 对应物理37号引脚 / GPIO26
BLINK_DURATION = 2    # 爆闪持续时间（秒）
BLINK_INTERVAL = 0.1  # 爆闪亮灭间隔（秒）

# -------------------------- 全局逻辑配置 --------------------------
ACTION_LOCK_TIME = 3  # 触发后全局锁定时间（秒），3秒内不重复执行任何动作

# -------------------------- 全局变量 --------------------------
last_trigger_time = 0  # 最后一次动作触发时间


# -------------------------- 舵机控制函数 --------------------------
def run_sys_cmd(cmd):
    """执行系统命令，屏蔽错误输出"""
    os.system(cmd + " 2>/dev/null")

def init_servo():
    """初始化舵机PWM配置"""
    run_sys_cmd(f"echo {PWM_CHANNEL} > /sys/class/pwm/pwmchip{PWM_CHIP}/export")
    time.sleep(0.1)
    
    with open(f"/sys/class/pwm/pwmchip{PWM_CHIP}/pwm{PWM_CHANNEL}/period", "w") as f:
        f.write(str(PERIOD_NS))
    
    with open(f"/sys/class/pwm/pwmchip{PWM_CHIP}/pwm{PWM_CHANNEL}/enable", "w") as f:
        f.write("1")
    
    set_servo_angle(SERVO_INIT_ANGLE)
    print(f"舵机初始化完成，初始角度：{SERVO_INIT_ANGLE}°")

def set_servo_angle(angle):
    """设置舵机角度（0°~180°），直接给定目标角度即以硬件最大速度转动"""
    duty_ns = int((0.5 + angle / 180.0 * 2.0) * 1_000_000)
    with open(f"/sys/class/pwm/pwmchip{PWM_CHIP}/pwm{PWM_CHANNEL}/duty_cycle", "w") as f:
        f.write(str(duty_ns))

def servo_trigger_action():
    """舵机触发动作：最大速度转90°，到位后归位"""
    set_servo_angle(SERVO_ROTATE_ANGLE)
    print(f"[{time.strftime('%H:%M:%S')}] 舵机旋转至 {SERVO_ROTATE_ANGLE}°")
    time.sleep(SERVO_RESPONSE_TIME)  # 等待舵机转动到位
    
    set_servo_angle(SERVO_INIT_ANGLE)
    print(f"[{time.strftime('%H:%M:%S')}] 舵机归位至 {SERVO_INIT_ANGLE}°")

def release_servo():
    """释放舵机PWM资源"""
    set_servo_angle(SERVO_INIT_ANGLE)
    time.sleep(0.5)
    
    with open(f"/sys/class/pwm/pwmchip{PWM_CHIP}/pwm{PWM_CHANNEL}/enable", "w") as f:
        f.write("0")
    run_sys_cmd(f"echo {PWM_CHANNEL} > /sys/class/pwm/pwmchip{PWM_CHIP}/unexport")
    
    print("舵机PWM资源已释放")


# -------------------------- LED爆闪控制函数 --------------------------
def init_led():
    """初始化LED GPIO引脚"""
    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)
    GPIO.setup(LED_PIN, GPIO.OUT, initial=GPIO.LOW)
    print(f"LED引脚初始化完成，物理引脚 {LED_PIN}")

def blink_action():
    """LED爆闪动作：持续BLINK_DURATION秒，结束后自动熄灭"""
    end_time = time.time() + BLINK_DURATION
    print(f"[{time.strftime('%H:%M:%S')}] LED开始爆闪，持续 {BLINK_DURATION} 秒")
    
    while time.time() < end_time:
        GPIO.output(LED_PIN, GPIO.HIGH)
        time.sleep(BLINK_INTERVAL)
        GPIO.output(LED_PIN, GPIO.LOW)
        time.sleep(BLINK_INTERVAL)
    
    GPIO.output(LED_PIN, GPIO.LOW)
    print(f"[{time.strftime('%H:%M:%S')}] LED爆闪结束")

def release_led():
    """释放LED GPIO资源"""
    GPIO.output(LED_PIN, GPIO.LOW)
    GPIO.cleanup()
    print("LED GPIO资源已释放")


# -------------------------- 联动动作触发 --------------------------
def trigger_all_actions():
    """多线程同步启动舵机动作与LED爆闪，不阻塞主识别循环"""
    threading.Thread(target=servo_trigger_action, daemon=True).start()
    threading.Thread(target=blink_action, daemon=True).start()


# -------------------------- 人脸识别函数 --------------------------
def extract_feature(face_img):
    """提取人脸特征"""
    img = cv2.resize(face_img, (112, 112))
    img_np = img.astype(np.float32)
    img_np = (img_np - 127.5) * 0.007843137
    img_np = np.transpose(img_np, (2, 0, 1))
    input_tensor = np.expand_dims(img_np, axis=0)
    outputs = rec_model[0].forward(input_tensor)
    return outputs[0].buffer.flatten()

def cosine_similarity(a, b):
    """计算余弦相似度"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def recognize(feat):
    """识别人脸（仅比对goal特征）"""
    best_score = 0.0
    for db_feat in goal_feats:
        sim = cosine_similarity(feat, db_feat)
        if sim > best_score:
            best_score = sim
    if best_score > THRESHOLD:
        return "goal", best_score
    else:
        return "unknown", best_score


# -------------------------- 主程序 --------------------------
if __name__ == "__main__":
    # 1. 加载人脸识别模型
    print("加载识别模型...")
    try:
        rec_model = dnn.load(REC_MODEL_PATH)
        detector = cv2.FaceDetectorYN.create(DET_MODEL_PATH, "", (320, 320), 0.6)
    except Exception as e:
        print(f"模型加载失败：{e}")
        exit(1)
    
    # 2. 加载人脸特征库
    if not os.path.exists(DB_PATH):
        print("特征库不存在，请先运行 build_face_db_yunet.py")
        exit(1)
    db = np.load(DB_PATH, allow_pickle=True).item()
    goal_feats = db.get("goal", [])
    if len(goal_feats) == 0:
        print("警告：goal 特征为空，请录入目标照片并构建特征库")
    
    # 3. 初始化摄像头
    cam = libsrcampy.Camera()
    if cam.open_cam(0, -1, 30, CAM_WIDTH, CAM_HEIGHT) < 0:
        print("打开摄像头失败")
        exit(1)
    
    # 4. 初始化舵机
    try:
        init_servo()
    except Exception as e:
        print(f"舵机初始化失败：{e}")
        cam.close_cam()
        exit(1)
    
    # 5. 初始化LED
    try:
        init_led()
    except Exception as e:
        print(f"LED初始化失败：{e}")
        release_servo()
        cam.close_cam()
        exit(1)
    
    # 6. 主循环
    print("开始实时识别 + 舵机/LED联动，按 Ctrl+C 停止")
    try:
        while True:
            img_data = cam.get_img()
            if img_data is None:
                time.sleep(0.1)
                continue
            img_np = np.frombuffer(img_data, dtype=np.uint8).reshape((CAM_HEIGHT*3//2, CAM_WIDTH))
            frame = cv2.cvtColor(img_np, cv2.COLOR_YUV2BGR_NV12)
            
            h, w, _ = frame.shape
            detector.setInputSize((w, h))
            _, faces = detector.detect(frame)
            
            if faces is not None:
                face = faces[0]
                x, y, w_box, h_box, score = face[:5]
                x1, y1, x2, y2 = int(x), int(y), int(x+w_box), int(y+h_box)
                face_roi = frame[y1:y2, x1:x2]
                
                if face_roi.size > 0:
                    feat = extract_feature(face_roi)
                    label, conf = recognize(feat)
                    print(f"{time.strftime('%H:%M:%S')} 识别结果: {label} (相似度 {conf:.2f})")
                    
                    # 识别到目标且不在冷却期，触发联动
                    if label == "goal" and (time.time() - last_trigger_time) > ACTION_LOCK_TIME:
                        print(f"{time.strftime('%H:%M:%S')} 触发联动动作！")
                        last_trigger_time = time.time()  # 立即上锁，防止重复触发
                        trigger_all_actions()
            else:
                print(f"{time.strftime('%H:%M:%S')} 未检测到人脸")
            
            time.sleep(INTERVAL)
    
    except KeyboardInterrupt:
        print("\n用户停止程序")
    except Exception as e:
        print(f"程序异常：{e}")
    finally:
        print("释放资源中...")
        cam.close_cam()
        release_servo()
        release_led()
        print("程序正常退出")