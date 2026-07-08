# 人脸识别攻击与防御系统

基于地平线（Horizon Robotics）嵌入式平台的实时人脸识别系统，联动舵机与LED实现物理交互。

## 项目概述

本项目实现了一套智能人脸识别系统，持续监控摄像头画面。当检测到目标人脸（"goal"）时，触发联动动作：舵机旋转模拟物理响应，LED爆闪作为视觉警示。

## 文件结构

| 文件 | 说明 |
|------|------|
| `attack.py` | 主程序：实时识别 + 舵机/LED联动控制 |
| `build_face_db_yunet.py` | 从图片目录构建人脸特征库 |
| `face_db.npy` | 生成的人脸特征库文件 |
| `yunet.onnx` | YuNet人脸检测模型（ONNX格式） |
| `w600k_r50.bin` | W600K-R50人脸识别模型（地平线BPU格式） |

## 硬件要求

- **平台**：地平线（Horizon Robotics）嵌入式开发板（如X3、X5系列）
- **摄像头**：MIPI摄像头，通过`libsrcampy`驱动（1920x1080 @ 30fps）
- **舵机**：连接至PWM通道1（物理33号引脚）
- **LED**：连接至GPIO物理37号引脚（GPIO26）

## 依赖环境

- 地平线DNN SDK（`hobot_dnn`）
- 地平线VIO SDK（`hobot_vio`）
- 地平线GPIO SDK
- 支持`FaceDetectorYN`的OpenCV（opencv-contrib）
- NumPy

## 快速开始

### 1. 构建人脸特征库

首先准备好人脸图片，构建特征库。

将目标人物照片放入 `/root/fly/ai/goal/`，其他人照片放入 `/root/fly/ai/friends/`。

```bash
python build_face_db_yunet.py
```

运行后生成 `face_db.npy`，包含提取的人脸特征向量。

### 2. 运行主程序

```bash
python attack.py
```

系统将持续执行以下流程：
- 从摄像头采集画面（1920x1080，每2秒采样一次）
- 使用YuNet检测人脸
- 使用W600K-R50模型提取人脸特征
- 与goal特征库进行余弦相似度比对（阈值0.5）
- 若识别到目标人脸且冷却期已过（3秒），触发联动动作：
  - **舵机**：以硬件最大速度从0°旋转至90°后归位
  - **LED**：爆闪2秒（亮灭间隔100ms）

按 **Ctrl+C** 停止程序。

## 配置参数

所有可调参数定义在 `attack.py` 文件顶部：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `DET_MODEL_PATH` | `/userdata/yunet.onnx` | 人脸检测模型路径 |
| `REC_MODEL_PATH` | `/userdata/w600k_r50.bin` | 人脸识别模型路径 |
| `DB_PATH` | `/root/fly/ai/face_db.npy` | 特征库路径 |
| `CAM_WIDTH` | 1920 | 摄像头画面宽度 |
| `CAM_HEIGHT` | 1080 | 摄像头画面高度 |
| `INTERVAL` | 2 | 检测间隔（秒） |
| `THRESHOLD` | 0.5 | 余弦相似度阈值 |
| `SERVO_ROTATE_ANGLE` | 90 | 触发时舵机旋转角度 |
| `SERVO_RESPONSE_TIME` | 0.15 | 舵机到位等待时间（秒） |
| `BLINK_DURATION` | 2 | LED爆闪持续时间（秒） |
| `BLINK_INTERVAL` | 0.1 | LED亮灭间隔（秒） |
| `ACTION_LOCK_TIME` | 3 | 动作冷却时间（秒） |

## 工作原理

```
摄像头 → YuNet人脸检测 → 人脸裁剪(112x112) → W600K特征提取
                                                    ↓
                                          余弦相似度比对特征库
                                                    ↓
                                          相似度 > 0.5？→ 舵机 + LED联动
```

### 识别流程

1. **采集**：通过 `libsrcampy` 从MIPI摄像头获取RGB画面
2. **检测**：YuNet ONNX模型检测人脸边界框
3. **特征提取**：人脸区域缩放为112×112，归一化后经W600K-R50模型输出特征向量
4. **识别**：计算采集特征与goal特征库的余弦相似度，判定为"goal"或"unknown"
5. **联动**：识别到目标且在冷却期外时，舵机与LED在独立线程中同步触发，不阻塞主循环

## 许可证

本项目仅用于教学与实验目的。
