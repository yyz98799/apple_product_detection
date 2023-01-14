# 简介
通过[mmdetection](https://github.com/open-mmlab/mmdetection)框架，训练并识别部分苹果产品。并采用[ConvNeXt](https://github.com/facebookresearch/ConvNeXt)识别给定的直播场景。
# 安装
1. 参考[mmdetection安装指南](https://mmdetection.readthedocs.io/en/stable/)，安装mmdetection
2. 安装flask  
```pip install flask```
# 运行
## 服务器启动
切换至```apple_demo```目录，运行```python phone_server.py```。
## 请求json示意
```json
{
    "img_base64": "iVBORw0KGgoAAAANSUhEUgAAADwAAAAuCAMAAABDPIrQAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAGUExURf///wAAAFXC034AAAACdFJOU/8A5bcwSgAAAAlwSFlzAAAXEQAAFxEByibzPwAAABlJREFUSEvtwQENAAAAwqD3T20PBwQAAOdqCvYAAQopjw8AAAAASUVORK5CYII=", 
    "debug_mode": 1
}
```
## 响应json示意
```json
{
    "scene_class": 1,
    "results": [
        {
            "center": false,
            "lockscreen": true,
            "reflection": false,
            "type": "phone_front",
            "probability": 0.318,
            "position": [
                115,
                424,
                310,
                825
            ]
        },
        {
            "center": false,
            "lockscreen": false,
            "reflection": false,
            "type": "phone_back",
            "probability": 0.788,
            "position": [
                390,
                387,
                610,
                828
            ]
        }
    ]
}
```
# 注意事项
1. 改动了mmdetection框架中coco数据集类别与类别数量，如需要运行coco数据集的识别需要自行恢复
2. ConvNeXt模型配置文件为```apple_demo/model.py```