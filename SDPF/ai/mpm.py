from general_functions import functions
from general_functions.models import BasicModel
import torch
import numpy as np
import os
from ai.Train import preprocessing_training_data, create_train_model
from ai.Inference import predict_single
from pathlib import Path
from ai.plot import plot_rec
from alarmSystem.Data.db.collectionDB import CollectionDB


import numpy as np
import os
from general_functions.data_processor import multiaxis_gaf_image


def sensors_to_gaf(
    sensor_dict,
    save_dir,
    axis_first=False,
    method="summation",
    image_format="npy"
):
    """
    Parameters
    ----------
    sensor_dict : dict
        {
            "sensor1": {
                "channel0": np.array,
                "channel1": np.array,
                "channel2": np.array
            },
            ...
        }
    save_dir : str
        保存路径
    axis_first : bool
        False -> (T, C)
        True  -> (C, T)
    """

    os.makedirs(save_dir, exist_ok=True)

    for sensor_name, channels in sensor_dict.items():

        # 1️⃣ 按 channel 名字排序，防止顺序错乱
        channel_keys = sorted(channels.keys())  # ['channel0','channel1','channel2']

        # 2️⃣ 每个 channel 去掉最后一个点
        channel_data = []
        for ch in channel_keys:
            arr = channels[ch][:-1]   # 核心要求：去掉最后一个数据
            channel_data.append(arr)

        # 3️⃣ 堆叠为 (T, C)
        # shape: (4096, 3)
        data = np.stack(channel_data, axis=1)

        if axis_first:
            data = data.T  # (C, T)

        # 4️⃣ 生成 GAF（多通道）
        gaf_img = multiaxis_gaf_image(
            data=data,
            save_path=save_dir,
            image_name=sensor_name,
            axis_first=axis_first,
            method=method,
            image_format=image_format
        )

        # 5️⃣ 明确保存（防止函数内部保存不符合你预期）
        save_path = os.path.join(save_dir, f"{sensor_name}.npy")
        np.save(save_path, gaf_img)

        print(f"✅ Saved GAF for {sensor_name}: {gaf_img.shape} -> {save_path}")


class Ai(BasicModel):
    def __init__(self, name, config_path, shot, model_name='ai'):
        super().__init__(name, config_path, shot, model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.train_models()
        sample_data, sensors_data = functions.get_sensors_data(self.data_source, self.shot, self.name,
                                                               self.sensors)
        print(sensors_data)

def main():
    # test_shots_calculate()
    # # 输入参数
    # config_path, name, shot = functions.get_input_params('ai')
    config_path = './config.yml'
    name = 'bm1'
    shot = '1110400'
    Ai(name, config_path, shot)
    # test_shots_calculate()

if __name__ == "__main__":
    main()
