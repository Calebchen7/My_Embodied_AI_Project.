from ultralytics import YOLO

##### 加载预训练模型
model = YOLO('yolov8n.pt')                    #下载好的权重路径填在这里

##### 进行推理 (可以是本地图片路径)
results = model.predict(
    source='E:\code\yolov8\inference\images',    # 图片路径
    save=True,                              # 保存结果图片
    imgsz=640,                              # 推理尺寸
    conf=0.25,                              # 置信度阈值
    device='0'                              # 使用GPU (如果可用)
)

##### 打印检测到的物体类别
for result in results:
    boxes = result.boxes  # 边框
    print(f"检测到 {len(boxes)} 个物体")
    print("类别:", result.names)