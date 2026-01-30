"""
功能：加载训练好的模型，进行目标检测并统计数量
"""
import cv2
from ultralytics import YOLO
import os

#### 配置参数
# 指向你训练好的权重文件
WEIGHTS_PATH = 'runs/train/recognition/weights/best.pt'
SOURCE = 'inference/videos/test.mp4'                     # 输入源: 可以是视频路径，或者 '0' 代表摄像头
CONFIDENCE_THRESHOLD = 0.5                               # 置信度阈值，大于此值才显示
WINDOW_SIZE = (960, 540)                                 # 显示窗口大小 (宽度, 高度)


def main():
    # 检查权重文件是否存在
    if not os.path.exists(WEIGHTS_PATH):
        print(f" 错误：未找到训练好的权重文件 {WEIGHTS_PATH}")
        print("请先运行 train.py 进行训练。")
        return

###### 加载训练好的模型
    print(f" 加载模型: {WEIGHTS_PATH}")
    model = YOLO(WEIGHTS_PATH)

###### 设置推理源
    # 如果 SOURCE 是 '0'，则调用摄像头

    if SOURCE.isdigit():
        cap = cv2.VideoCapture(int(SOURCE))
    else:
        if not os.path.exists(SOURCE):
            print(f" 错误：未找到文件 {SOURCE}")
            return
        cap = cv2.VideoCapture(SOURCE)

    print(" 开始检测与计数... 按 'q' 键退出")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(" 视频播放结束或无法读取帧")
            break

        # 4. 进行推理
        # 注意：这里传入 frame 而不是文件路径，以避免重复读取文件
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

        # 5. 绘制结果
        # results[0].plot() 会自动在帧上画出框和标签
        annotated_frame = results[0].plot()

        # --- 核心计数逻辑 ---
        count = len(results[0].boxes)  # 获取检测到的物体数量
        cv2.putText(annotated_frame, f'Total Count: {count}', (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5, cv2.LINE_AA)

        # 调整显示窗口大小
        display_frame = cv2.resize(annotated_frame, WINDOW_SIZE)

        # 6. 显示画面
        cv2.imshow('YOLO Object Counting', display_frame)

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print(" 程序结束")


if __name__ == '__main__':
    main()


