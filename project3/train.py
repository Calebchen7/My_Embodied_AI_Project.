"""
train.py
功能：使用 YOLOv8 训练自定义数据集,
原本电脑上已经安装了YOLOv8,并且看的教程是YOLOv8的，所以以下的训练都用了YOLOv8；
注：请确保 my_data 文件夹和 data.yaml 路径正确()
"""

from ultralytics import YOLO
import os

#  配置参数
DATA_CONFIG = 'E:\code\yolov8\mydata.yaml'     # 数据配置文件路径
PRETRAINED_MODEL = 'yolov8n.pt'                # 使用官方预训练的 轻量级 nano 模型
EPOCHS = 150                                   # 训练轮数
IMG_SIZE = 640                                 # 输入图片大小
BATCH_SIZE = 16                                # 批次大小 (根据显存调整，-1为自动)
PROJECT_NAME = 'runs/train'                    # 保存项目文件夹
NAME = 'recognition'                           # 实验名称
WORKER_SIZE =1                                 #数据加载的线程数（多线程加速数据读取）



def recognition():
    # 检查数据配置文件是否存在
    if not os.path.exists(DATA_CONFIG):
        print(f" 错误：未找到数据配置文件 {DATA_CONFIG}")
        print("请确保 my_data 文件夹和 data.yaml 在当前目录下。")
        return

###加载模型
    # 如果本地有之前中断保存的权重，可以接着训练；否则加载官方预训练权重
    if os.path.exists(PRETRAINED_MODEL):
        print(f" 加载预训练权重: {PRETRAINED_MODEL}")
        model = YOLO(PRETRAINED_MODEL)
    else:
        print(" 未找到预训练权重，正在自动下载 yolov8n.pt...")
        model = YOLO('yolov8n.pt')                         # 会自动下载

#### 开始训练
    print("开始训练...")

    try:
        results = model.train(
            data=DATA_CONFIG,      # 数据集配置
            epochs=EPOCHS,         # 训练轮数
            imgsz=IMG_SIZE,       # 图像尺寸
            batch=BATCH_SIZE,     #批量大小（每次迭代送入模型的图片数）
            workers=WORKER_SIZE,  #数据加载的线程数（多线程加速数据读取）
            project=PROJECT_NAME, # 保存路径
            name=NAME,            # 实验名
            exist_ok=True,        # 允许覆盖同名实验
            device='0'            # 选择GPU
        )
        print("训练完成！")
        print(f"模型已保存在: {PROJECT_NAME}/{NAME}")
    except Exception as e:
        print(f" 训练过程中发生错误: {e}")

if __name__ == '__main__':
    recognition()