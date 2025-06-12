from alarmSystem.Data.db.collectionDB import CollectionDB
from pymongo import MongoClient


def split_string(string, separate_identifier):
    string_list = string.split(separate_identifier)
    return string_list


def connect_mongodb(address):
    client = MongoClient(address)
    return client


def connect_database(client, database_name):
    db = client[database_name]
    return db


def connect_mongodb_database(address, database_name):
    client = connect_mongodb(address)
    db = connect_database(client, database_name)
    return client, db


def connect_config_database(config):
    client, db = connect_mongodb_database(config['db']['connection'], config['db']['db_name'])
    return client, db


def connect_config_collection(config):
    client, db = connect_config_database(config)
    collection = db[config['db']['collection']]
    return client, db, collection


def split_input_collection(input, separate_identifier='>'):
    """
    将数据库输入进行切割
    :param input: 数据库查询输入 如bm1_rms>sensors.sensor1.r_rms
    :param separate_identifier: 划分input的collection的辨识符 如>
    :return: 集合名称，集合内要取出的数据对应的关键字
    """
    input_collection = split_string(input, separate_identifier)
    collection = input_collection[0]
    input = input_collection[1]
    return collection, input


def get_is_running_shot(db, collection_name, shot_num, max_shot: int, min_shot=-1):
    """
    返回对应collection中shot介于min_sot和max_shot中的shot_num数目的正在运行的shot列表
    """
    collection_db = CollectionDB(db, collection_name)
    documentation_list = collection_db.find_latest_n_records(shot_num, max_shot=max_shot, min_shot=min_shot)
    # print(documentation_list)
    is_running_shot = []
    for documentation in documentation_list:
        is_running_shot.append(documentation['shot'])
    return is_running_shot


class DatabaseFinder(CollectionDB):
    def __init__(self, input, shot: int, shot_num, db, collection_name, separate_identifier='.'):
        """
        返回数据库的一定数目的记录
        :param input: 查询字符  如sensors.sensor1.r_rms
        :param shot: 查询炮号
        :param shot_num: 需要查询的记录数目
        :param db: 已经连接到的数据库
        :param collection_name: 查询的集合名称
        :param separate_identifier: 输入的查询关键字间的划分符号 如sensors.sensor1.r_rms的划分符号为.
        """
        super().__init__(db, collection_name)
        self.input_list = split_string(input, separate_identifier)
        self.shot = shot
        self.shot_num = shot_num
        self.data = self.get_data()
        # print(self.data)

    def find_documentation_list(self):
        documentation_list = super().find_latest_n_records(self.shot_num, max_shot=self.shot)
        # print(documentation_list)
        return documentation_list

    def from_documentation_get_data(self, documentation):
        data = documentation
        for key in self.input_list:
            data = data[key]
        return data

    def get_data(self):
        data_list = []
        documentation_list = self.find_documentation_list()
        for documentation in documentation_list:
            data = self.from_documentation_get_data(documentation)
            data_list.append(data)
        return data_list


def get_data(shot, input, shot_num, db, input_separate_identifier):
    """
    输入如collection>sensors.sensor1.r_rms，取出对应数据
    :param input: 输入字符串 如collection>sensors.sensor1.r_rms
    :param shot_num: 需要取出的数据炮数
    :param db: 已连接的数据库
    :return:
    """
    collection_name, data_key = split_input_collection(input)
    # collection_name = functions.replace_ball_mill_name(collection_name, self.name)
    data = DatabaseFinder(data_key, shot, shot_num, db, collection_name, input_separate_identifier).data
    return data


if __name__ == '__main__':
    db_config = {
        "connection": "mongodb://localhost:27017/",
        "db_name": "bm",
        "collection": "bm1_rms"
    }
    # 输入
    input_query = "sensors.sensor1.r_rms"
    client = MongoClient(db_config["connection"])
    db = client[db_config["db_name"]]
    database_finder = DatabaseFinder(input_query, 1108215, 5, db, "bm1_rms")
    print(database_finder.data)
