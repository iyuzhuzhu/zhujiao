import requests
import json
import numpy as np


# from general_functions.functions import single_hysteresis_alarm
# from general_functions import functions
#
# 示例使用
def cross_entropy(p, q):
    p, q = np.array(p), np.array(q)
    assert p.shape == q.shape, "The two distributions must have the same shape."
    return -np.sum(p * np.log(q + 1e-10))  # 添加一个小值避免log(0)


p = np.array([0.1, 0.2, 0.3])
q = np.array([0.1, 0.2, 0.3])
r = np.array([0.2, 0.4, 0.6])
t = np.array([0.3, 0.5, 0.7])
print(cross_entropy(p, q))
print(cross_entropy(r, q))
print(cross_entropy(t, q))
print(cross_entropy(r+t, q))
# values = [False, False, False]
# result = any(values)
# print(result)  # 输出: True
# a = ['sensor1_x_minor', 'sensor1_x_major', 'sensor1_x_fatal', 'sensor1_y_minor']
# b = ['sensor1_x_1', 'sensor1_x_2', 'sensor1_x_3', 'sensor1_y_1']
# c = functions.remove_lower_alarm(b)
# d = functions.return_final_format_alarms(c)
# print(c)
# print(d)
# a = [1 ,2, 3, 4, 5, 6]
# for i in a:
#     for j in a:
#         if i == j:
#             print("相等", i, j)
# if single_hysteresis_alarm(a, 2.9, 6, 5, 4, True):
#     print("异常")
# is_running = None
# if is_running:
#     print('等同True')
# else:
#     print('等同False')
# re = {
#                 "rms_mean": [1],
#                 "rms_x": [2],
#                 "rms_y": [3],
#                 "rms_z": [4],
#             }
# for r in re:
#     print(r)
# 需要传递的数据
# data = {
#     "bail_mill_name": "bm3",
#     "shot": "2220000",
#     "is_running": False,
#     "data_time": "2024-08-24 08:45"
# }
# shots = np.arange(100, 105)
# print(shots)
#
# data = {"bail_mill_name": "bm1", "shot": "1110000", "is_running": None, "data_time": "2024-08-23 03:09", "results": [{"senor": "sensor1", "alarm": None, "rms_mean": 0.013464680591177195, "rms_x": 0.017936836754939994, "rms_y": 0.014681936742392864, "rms_z": 0.007775268276198724}, {"senor": "sensor2", "alarm": None, "rms_mean": 0.0, "rms_x": 0.0, "rms_y": 0.0, "rms_z": 0.0}, {"senor": "sensor3", "alarm": None, "rms_mean": 0.0, "rms_x": 0.0, "rms_y": 0.0, "rms_z": 0.0}, {"senor": "sensor4", "alarm": None, "rms_mean": 0.0, "rms_x": 0.0, "rms_y": 0.0, "rms_z": 0.0}, {"senor": "sensor5", "alarm": None, "rms_mean": 0.015473393350410869, "rms_x": 0.007772049454963081, "rms_y": 0.010270217144423853, "rms_z": 0.02837791345184567}, {"senor": "sensor6", "alarm": None, "rms_mean": 0.0, "rms_x": 0.0, "rms_y": 0.0, "rms_z": 0.0}]}
#
# # 发送 POST 请求到 Flask 应用的 API 路由
# url = "http://localhost:5000/api/update_data"
# headers = {'Content-Type': 'application/json'}  # 请求头，指定数据格式为 JSON
#
# response = requests.post(url, headers=headers, data=json.dumps(data))
#
# # 输出服务器的响应
# print(response.status_code)  # 状态码
# print(response.json())       # 响应的JSON内容
#
# # 目标URL（Flask 应用程序的 API 端点）
# url = "http://localhost:5000/api/get_data"
#
# # 发送 GET 请求到 Flask 应用
# response = requests.get(url)
#
# # 检查请求是否成功
# if response.status_code == 200:
#     # 将响应内容解析为JSON格式
#     data = response.json()
#     print("Data received from server:", data)
#     print(type(data))
# else:
#     print("Failed to retrieve data. Status code:", response.status_code)
