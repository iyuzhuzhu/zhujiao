import os
import numpy as np
from pyts.image import GramianAngularField
import imageio.v2 as imageio


def multiaxis_gaf_image(
    data,
    save_path,
    image_name,
    axis_first=False,
    method="summation",
    image_format="npy",
):
    """
    将多轴时间序列转换为多通道 GAF 图像并保存。

    Parameters
    ----------
    data : np.ndarray
        时间序列数据，shape 为 (T, C) 或 (C, T), C = 轴向数量（通常是 3）
    save_path : str
        图像保存目录
    image_name : str
        文件名（不含后缀）
    axis_first : bool
        True  -> data shape (C, T)
        False -> data shape (T, C)
    method : str
        GAF 方法: "summation" (GASF) 或 "difference" (GADF)
    image_format : str
        "npy"  -> 保存为 numpy
        "png"  -> 保存为多通道 png（不推荐做训练）
    """

    os.makedirs(save_path, exist_ok=True)

    # 确保数据是 (C, T)
    if not axis_first:
        data = data.T

    num_axes, seq_len = data.shape

    gaf = GramianAngularField(
        method=method,
        image_size=seq_len
    )

    gaf_images = []

    for axis_data in data:
        axis_data = axis_data.reshape(1, -1)
        gaf_img = gaf.fit_transform(axis_data)[0]
        gaf_images.append(gaf_img)

    # (C, N, N)
    multi_channel_img = np.stack(gaf_images, axis=0)

    # 保存
    full_path = os.path.join(save_path, image_name)

    if image_format == "npy":
        np.save(full_path + ".npy", multi_channel_img)

    elif image_format == "png":
        # 归一化到 0–255（仅用于可视化）
        img = (multi_channel_img - multi_channel_img.min()) / \
              (multi_channel_img.max() - multi_channel_img.min() + 1e-8)
        img = (img * 255).astype(np.uint8)

        # 转成 (H, W, C)
        img = np.transpose(img, (1, 2, 0))
        imageio.imwrite(full_path + ".png", img)

    else:
        raise ValueError("Unsupported image_format")

    return multi_channel_img

if __name__ == "__main__":
    from general_functions.models import BasicModel
    from general_functions import functions
    # # 输入参数
    # config_path, name, shot = functions.get_input_params('rms')
    config_path = './config.yml'
    name = 'bm1'
    config = functions.read_config(config_path)
    # 得到bail_mill中的bail_name
    shots = np.arange(1108200, 1110400)
    # shots = np.arange(1012849, 1110500)
    class mpm(BasicModel):
        def __init__(self, name, config_path, shot, model_name='ai'):
            super().__init__(name, config_path, shot, model_name)
            
    for shot in shots:
        shot = str(shot)
        mpm(name, config_path, shot)
        print(shot)

        
            # self.train_models()
            
            # self.predict_single_axis_loss()