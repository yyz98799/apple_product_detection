from mmdet.apis import init_detector, inference_detector
import cv2
import numpy as np
import os
import json
import time
from io import BytesIO

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify
import base64

from model import convnext_tiny as create_model

app = Flask(__name__)

# 产品分类类别和场景分类类别
class_name = ["phone_front", "phone_back", "ipad", "apple_watch", "airpods"]
sense_class_name = ["other", "iphone_comp", "iphone_demo", "iphone_soip",
                    "watch_comp", "watch_demo", "ipad_demo", "airpods_comp", "single"]


# 场景分类、目标检测、黑屏检测，返回json字符串
def operation(img_detect, debug_mode):
    """ response sample
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
    """

    dect_result = []
    rects = []
    res_dict = {}
    f = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    result = inference_detector(dete_model, img_detect)


    # 场景检测部分
    sense_cls = Image.fromarray(cv2.cvtColor(img_detect, cv2.COLOR_BGR2RGB))
    # [N, C, H, W]
    sense_cls_tensor = data_transform(sense_cls)
    # expand batch dimension
    sense_cls_tensor = torch.unsqueeze(sense_cls_tensor, dim=0)

    with torch.no_grad():
        # predict class
        output = torch.squeeze(sense_model(sense_cls_tensor.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    sense_pre_class = class_sense[str(predict_cla)]
    sense_pre_prob = predict[predict_cla].numpy()
    sense_pre_class_index = sense_class_name.index(sense_pre_class)
    if sense_pre_class_index == 8:
        sense_pre_class_index = 0

    if sense_pre_class_index == 1:
        result_mask = inference_detector(mask_model, img_detect)
        mask = np.zeros((result_mask[1][0][0].shape[0], result_mask[1][0][0].shape[1]), dtype=bool)
        # 65: remote, 67: cell phone
        mask_class_list = [67, 65]

        # result：类型为tuple，包含两项：
        # result[0]：长度为80的二维ndarray List(对应MS COCO的80个类别)
        # result[0][0]：二维ndarray，每行长度为5，对应各个mask的坐标(x1, y1, x2, y2)以及置信度
        # result[1]：长度为80的List(对应MS COCO的80个类别)
        # result[1][0]：二维ndarray List，每个mask对应一个二维ndarray
        # result[1][0][0]：二维ndarray，数据类型为bool，大小为输入图片像素，值为True表示此像素包括在mask内
        for mask_class in mask_class_list:
            for index, i in enumerate(result_mask[0][mask_class]):
                if i[4] > score_thr_mask:
                    mask = mask | result_mask[1][mask_class][index]

        uint8_mask = mask.astype(np.uint8) * 255
        edges = cv2.Canny(uint8_mask, 100, 200)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rectangles = []
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            if min(rect[1][0], rect[1][1]) < 140:
                continue
            rect_points = cv2.boxPoints(rect)
            rect_points = np.int64(rect_points)
            rectangles.append(rect_points)
            center, box, angle = rect
            angle = abs(angle)
            if angle > 45:
                angle = 90 - angle
            rects.append([rect[0][0], rect[0][1], rect[1][0], rect[1][1], round(angle, 3)])

        # masklist = {mask_class: [] for mask_class in mask_class_list}
        # for mask_class in mask_class_list:
        #     for index, i in enumerate(result_mask[0][mask_class]):
        #         if i[4] > score_thr_mask:
        #             masklist[mask_class].append(index)
        # flist = np.array([[False] * width] * height)
        # for mask_class in mask_class_list:
        #     if masklist[mask_class]:
        #         for index in masklist[mask_class]:
        #             k = np.array([(i or j) for i, j in np.nditer([flist, result_mask[1][mask_class][index]])])
        #             flist = k.reshape(height, width)

    for class_index, class_det in enumerate(result):
        class_curr = class_name[class_index]
        bbox_result = [class_det]
        bboxes = np.vstack(bbox_result)
        # scores = [s1 s2 s3 s4 ...]
        scores = bboxes[:, -1]
        # 剔除低置信度结果
        # inds = [True False ...]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        for i, box in enumerate(bboxes):
            res_dict = {'lockscreen': False, 'reflection': False}

            left, upper, right, lower = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            pos = [left, upper, right, lower]
            pos_tuple = (left, upper, right, lower)

            prob = round(float(scores[i]), 3)
            res_dict['type'] = class_curr
            res_dict['probability'] = prob
            res_dict['position'] = pos

            # # 分类部分
            # OpenCV [upper:lower, left:right]
            # cut_frame = img_detect[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            # img_cls = Image.fromarray(cv2.cvtColor(cut_frame, cv2.COLOR_BGR2RGB))

            # plt (left, upper, right, lower)
            img_cls = sense_cls.crop(pos_tuple)
            # plt.imshow(img_cls)

            # ---lock screen dect---
            if class_index == 0:
                # [N, C, H, W]
                img_cls = data_transform(img_cls)
                # expand batch dimension
                img_cls = torch.unsqueeze(img_cls, dim=0)

                with torch.no_grad():
                    # predict class
                    output = torch.squeeze(cl_model(img_cls.to(device))).cpu()
                    predict = torch.softmax(output, dim=0)
                    predict_cla = torch.argmax(predict).numpy()

                pre_class = class_indict[str(predict_cla)]
                pre_prob = predict[predict_cla].numpy()

                if pre_class == "locked":
                    res_dict['lockscreen'] = True
                else:
                    res_dict['lockscreen'] = False

                # print_res = "class: {}   prob: {:.3}".format(pre_class, pre_prob)
                # plt.title(print_res)
            # ----------------------

            if debug_mode == 1:
                mem_save = BytesIO()
                plt.savefig(mem_save, format='png')
                img_base64 = base64.b64encode(mem_save.getvalue()).decode('utf8')
                res_dict['img_base64'] = img_base64
                mem_save.close()
            dect_result.append(res_dict)
    # response = {'scene_class': sense_pre_class_index, 'results': dect_result, 'mask': flist.tolist()}
    response = {'scene_class': sense_pre_class_index, 'results': dect_result, 'rects': rects}
    # torch.cuda.empty_cache()
    return response


# 若非部署于子网服务器须考虑加入鉴权
@app.route('/iphone_api', methods=['POST'])
def api():
    # ---ascii -> OpenCV image---
    # 捕捉客户端数据
    data = request.get_data().decode('utf-8')
    # 将string转换为dict
    data = json.loads(data)
    # 获取dict中'img'标签的数据
    image_b64 = data["img_base64"]
    # image_b64 = data["img_base64"].replace(' ', '+')
    # base64->数组
    image_decode = base64.b64decode(image_b64)
    # 数组->Ascii码
    nparr = np.fromstring(image_decode, np.uint8)
    # Ascii码->图像
    img_detect = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # -------------------------
    # 如果图像过大，调整为720*1280
    if img_detect.shape[0] > height:
        img_detect = cv2.resize(img_detect, (width, height))
    # debug mode, 0为不返回图片base64, 1为返回
    debug_mode = data["debug_mode"]

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    # 存储传入图片
    # cv2.imwrite(save_file_path + timestamp + ".png", img_detect)

    # 检测主函数
    # results = operation(img_detect, debug_mode)

    # json init
    # res_return = {'results': results}
    res_return = operation(img_detect, debug_mode)

    # 存储传入json
    # in_json = json.dumps(data)
    # json_in = open(save_file_path + timestamp + "in" + ".json", "w")
    # json_in.write(in_json)
    # json_in.close()
    # 存储返回json
    res_json = json.dumps(res_return)
    # jw = open(save_file_path + timestamp + "out" + ".json", "w")
    # jw.write(res_json)
    # jw.close()
    return jsonify(res_return)


# class FlaskApp(Flask):
#     def __init__(self, *args, **kwargs):
#         super(FlaskApp, self).__init__(*args, **kwargs)
#         self._load_model()
#
#     def _load_model(self):


if __name__ == '__main__':
    save_file_path = "phone_api_out/"
    if not os.path.exists(save_file_path):
        os.mkdir(save_file_path)

    # 目标检测模型
    checkpoint_file = 'weights/ddod_r50_fpn_1x_apple_epoch_12.pth'
    config_file = 'ddod_r50_fpn_1x_apple.py'
    checkpoint_file_mask = 'weights/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth'
    config_file_mask = 'mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")
    # 初始化检测器
    dete_model = init_detector(config_file, checkpoint_file, device=device)
    mask_model = init_detector(config_file_mask, checkpoint_file_mask, device=device)
    # 检测阈值
    score_thr = 0.3
    score_thr_mask = 0.3
    # 图像尺寸
    width = 720
    height = 1280
    # 分类器设置
    num_classes = 2
    img_size = 224

    # ---图像变换---
    data_transform = transforms.Compose(
        [transforms.Resize(img_size),
         # transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
         ]
    )
    # ------------

    # ---lockscreen detect init---
    cl_model = create_model(num_classes=num_classes).to(device)
    model_weight_path = "weights/convnext_tiny_1k_224_ema_lock_epoch_30.pth"
    cl_model.load_state_dict(torch.load(model_weight_path, map_location=device))
    cl_model.eval()
    json_path = 'class_indices_lock.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    with open(json_path, "r") as rf:
        class_indict = json.load(rf)
    # ---------------------------

    # ---sense detect init---
    num_classes_sense = 9
    sense_model = create_model(num_classes=num_classes_sense).to(device)
    model_weight_path_sense = "weights/convnext_tiny_1k_224_ema_sense_epoch_30.pth"
    sense_model.load_state_dict(torch.load(model_weight_path_sense, map_location=device))
    sense_model.eval()
    json_path_sense = 'class_indices_sense.json'
    assert os.path.exists(json_path_sense), "file: '{}' dose not exist.".format(json_path_sense)
    with open(json_path_sense, "r") as rf:
        class_sense = json.load(rf)
    # ----------------------

    app.debug = True
    app.run(host='0.0.0.0', port=5000, threaded=False)
