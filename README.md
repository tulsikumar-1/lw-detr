# Object Detection for Traffic Monitoring

Object detection is a crucial task in computer vision, involving not only identifying objects in an image or video but also locating them using bounding boxes. This goes beyond basic image classification by providing the precise location of objects. Object detection plays a vital role in numerous real-world applications, such as video surveillance, facial recognition, and autonomous driving. It is indispensable for tasks like identifying vehicles, pedestrians, and other objects in a traffic environment. Despite significant advancements, challenges remain in improving real-time performance and detecting small or overlapping objects.

## Cross-road Traffic Monitoring Dataset

This research utilizes the "Traffic Monitoring" dataset for object detection in traffic environments. The dataset includes images categorized into five different vehicle types:

1. Truck
2. Motorbike
3. Bus
4. Car
5. Bicycle

The dataset is split into the following sections:
- **Training**: 2,440 images (79%)
- **Validation**: 567 images (18%)
- **Testing**: 71 images (3%)

![518_predictions](https://github.com/user-attachments/assets/1a0856ff-460d-4b80-b350-3441d9326f2e)
<img src="![518_predictions](https://github.com/user-attachments/assets/1a0856ff-460d-4b80-b350-3441d9326f2e)" width="500"/>


## Model Performance Comparison

The following table summarizes the performance of various object detection models on the dataset, comparing parameters such as model size, inference time, FLOPs, and accuracy (mAP):

| **Model**     | **Params (M)** | **Inference Time (ms)** |  **GFLOPs**  | **mAP@50** | **mAP@50-95** |
|---------------|----------------|-------------------------|--------------|------------|---------------|
| Tiny          | 9.285          | 21.50                   |   11.2       | 0.835      | 0.391         |
| Tiny (Pruned) | 9.285          | 18.45                   |   11.2       | 0.833      | 0.381         |
| KD Tiny       | 9.285          | 21.50                   |   11.2       | 0.850      | 0.423         |
| Medium        |12.497          | 26.75                   |   18.6       | 0.845      | 0.442         |
| Large         |27.075          | 29.48                   |   42.8       | 0.851      | 0.538         |
| YOLOv8n       | 3.15           |  4.40                   |    8.9       | 0.826      | 0.573         |
| RT-DETR-l     |32.97           |  7.20                   |  108.3       | 0.833      | 0.580         |
