import base64
import time

import requests
import json
import os
import threading

def req(file_name):
    read_file_url = read_file_path + file_name
    print(read_file_url)
    with open(read_file_url, 'rb') as f:
        base64_data = base64.b64encode(f.read()).decode()
    data = {'img_base64': base64_data, 'debug_mode': 0}
    req_data = json.dumps(data)
    r = requests.post(url, data=req_data)
    j = json.loads(r.text)
    print(j["overlap"])
    # print(j)
    # for i, index in enumerate(j['results']):
    #     if i['type'] == 'phone_front' or i['type'] == 'phone_back':
    #         print(i['img_base64'])
    #         img = base64.b64decode(i['img_base64'])
    #         with open(save_file_path + file_name + "_" + str(index), 'wb') as wf:
    #             wf.write(img)
    return


url = 'http://localhost:5000/iphone_api'

repeat = 100
read_file_path = "./not/"
save_file_path = "phone_api_out/"
# 输入文件格式
pic_format = ["png", "PNG", "jpg", "JPG", "jpeg"]
# 获取输入文件目录下所有文件
file_list = [f for f in os.listdir(read_file_path) if f.split(".")[-1] in pic_format]
print(file_list)

for i in range(repeat):
    t = threading.Thread(target=req, args=(file_list[i], ))
    t.start()
    time.sleep(1)

