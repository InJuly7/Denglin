# Denglin Program
GPU：可提供32TOPS@ INT8，8TFLOPS@ FP16算力支撑，可支持H264/H265视频解码，32路1080P@30FPS;

CPU：8核64位处理器，采用ARM架构，四核Cortex-A76和四核Cortex-A55，以及单独的NEON协处理器，主频最高2.4GHz;

Driver version 4.6.2;

|模型| 前处理(ms) | 推理(ms) | 后处理(ms) |
|------|------|------|------|
| Yolov8s | 2.50514 ms | 16.807 ms | 9.152 ms |
| Yolov11s| 2.50514 ms | 20.957 ms | 8.467 ms |
| Yolov5s | 2.50062 ms | 9.363 ms | 10.885 ms |
| SeAFusion |  20.423 ms | 20.432 ms | 17.614 ms |
