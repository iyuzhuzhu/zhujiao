from pymongo import MongoClient
from general_functions.database_data import DatabaseFinder, split_string
import numpy as np

# 数据库连接信息
client = MongoClient("mongodb://localhost:27017/")
db = client["bm"]
collection = db["bm1_ai"]
input = "sensors.sensor1.r_ai"

data = DatabaseFinder(input, 1110400, 5000, db, collection_name='bm1_ai').data
print(np.max(data))

from pymongo import MongoClient, DESCENDING

# 连接数据库
# 查最大 r_ai 的文档，并提取 shot, r_ai, z_ai
result = collection.find_one(
    {},
    {
        "_id": 0,
        "shot": 1,
        "sensors.sensor1.r_ai": 1,
        "sensors.sensor1.z_ai": 1
    },
    sort=[("sensors.sensor1.r_ai", DESCENDING)]
)

# 格式化输出
if result:
    r_ai = result["sensors"]["sensor1"]["r_ai"]
    z_ai = result["sensors"]["sensor1"]["z_ai"]
    print(f"最大 r_ai 值为: {r_ai}")
    print(f"对应 shot: {result['shot']}")
    print(f"对应的 z_ai 值为: {z_ai}")
else:
    print("未找到符合条件的记录。")

# 查询所有文档中的 sensors.sensor1.r_ai 字段
