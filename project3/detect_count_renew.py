"""
增强版
功能：
1. 使用追踪器(Tracker)为每个物体分配唯一ID，实现去重。
2. 实时显示画面和计数。
3. 视频结束时，打印详细统计报告。
"""
import cv2
from ultralytics import YOLO
import os
from collections import defaultdict


# --- 配置区 ---
WEIGHTS_PATH = 'runs/train/recognition/weights/best.pt'   # 训练好的模型路径
SOURCE = 'inference/videos/test.mp4'  # 输入视频/图片路径
CONFIDENCE_THRESHOLD = 0.5  # 置信度阈值
WINDOW_SIZE = (960, 540)  # 显示窗口大小


def recognition():
#####  检查模型文件
    if not os.path.exists(WEIGHTS_PATH):
        print(f" 模型权重未找到: {WEIGHTS_PATH}")
        print("请先运行 train.py 或检查路径。")
        return

#####  加载模型
    # 加载检测权重并初始化追踪器
    model = YOLO(WEIGHTS_PATH)

#####  打开视频源
    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        print(f" 无法打开视频文件: {SOURCE}")
        return

    print(" 开始检测... (按 'q' 键提前退出)")

    # --- 核心数据结构 ---
#####  total_count: 记录每个类别在整个视频中出现的总次数（去重后的）
##### 使用集合(set)存储每个类别对应的物体ID，自动去重
    appearance_log = defaultdict(set)

#####  current_count: 记录当前画面中可见的物体数量（用于实时显示）
    current_objects = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # --- YOLO推理与追踪 ---
        # 注: persist=True 保持追踪ID的连续性
        # tracker='bytetrack.yaml' 使用ByteTrack算法，适合去重计数
        results = model.track(frame, conf=CONFIDENCE_THRESHOLD, persist=True, tracker='bytetrack.yaml', verbose=False)

        # --- 数据提取 ---
        # 获取当前帧的检测框、类别和追踪ID
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()  # 坐标
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)  # 类别ID
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)  # 追踪ID

            # 清空当前画面计数
            current_objects = {}

            # 遍历当前画面中的每一个检测结果
            for box, cls_id, track_id in zip(boxes, class_ids, track_ids):
                class_name = model.names[cls_id]  # 获取类别名称，如 'person', 'car'

                # --- 去重逻辑 ---
                # 如果这个物体的ID还没有被记录过，说明是新出现的，计入总数
                if track_id not in appearance_log[class_name]:
                    appearance_log[class_name].add(track_id)

                # --- 实时画面计数 ---
                # 统计当前画面中该类物体有多少个
                current_objects[class_name] = current_objects.get(class_name, 0) + 1

        # --- 绘图 ---
        # 1. 绘制检测框和追踪结果
        annotated_frame = results[0].plot()

        # 2. 绘制实时计数信息 (当前画面中的数量)
        y_offset = 50
        for obj_name, count in current_objects.items():
            text = f'Now {obj_name}: {count}'
            cv2.putText(annotated_frame, text, (50, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            y_offset += 50

        # 3. 绘制累计计数 (已出现过的总数)
        y_offset += 20
        for obj_name, id_set in appearance_log.items():
            text = f'Total {obj_name}: {len(id_set)}'
            cv2.putText(annotated_frame, text, (50, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            y_offset += 50

        # --- 显示 ---
        display_frame = cv2.resize(annotated_frame, WINDOW_SIZE)
        cv2.imshow('Object Counting with Tracking', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- 打印最终报告 ---
    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 50)
    print(" 检测完成！最终统计报告：")
    print("=" * 50)

    total_all = 0
    for class_name, id_set in appearance_log.items():
        count = len(id_set)
        total_all += count
        print(f" [{class_name.upper()}]: 共检测到 {count} 个")

    print("-" * 30)
    print(f" 总计检测到不同物体: {total_all} 个")
    print("=" * 50)
    print("程序结束")


if __name__ == '__main__':
    recognition()