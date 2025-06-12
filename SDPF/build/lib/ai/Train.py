import os.path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from ai.ae_model import LstmAutoencoder
from collections import Counter


EPOCH = 150
LR = 0.0004
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_SIZE = 0.1
BATCH_SIZE = 50


def is_tensor(obj):
    return isinstance(obj, torch.Tensor)


def transform_to_tensor(data):
    if is_tensor(data):
        return data
    else:
        target_length, _ = get_most_common_length(data, dim=1)
        data = filter_by_length(data, target_length, dim=0)
        return torch.Tensor(data)


def get_most_common_length(arrays_list, dim=1):
    """
    找出列表中NumPy一维数组出现次数最多的长度。

    参数:
        arrays_list (list of np.ndarray): 包含NumPy一维数组的列表。

    返回:
        tuple: 一个包含两个元素的元组，第一个元素是出现次数最多的长度，
               第二个元素是这个长度出现的次数。
    """
    # 提取每个数组的长度
    lengths = [arr.shape[0] for arr in arrays_list if arr.ndim == dim]
    # 如果列表为空或没有符合条件的一维数组，则返回None
    if not lengths:
        return None, 0
    # 使用Counter来计算每个长度出现的次数
    length_counts = Counter(lengths)
    # 找出出现次数最多的长度
    most_common_length, count = length_counts.most_common(1)[0]
    return most_common_length, count


def filter_by_length(data, target_length, dim=0):
    """
    过滤数据，仅保留指定维度上长度等于目标值的样本。

    Args:
        data (list): 原始数据集，每个元素为一个样本（如numpy数组或PyTorch张量）。
        target_length (int): 目标维度长度（默认为4096）。
        dim (int): 需要检查长度的维度（默认为1，即第2个维度）。

    Returns:
        list: 过滤后的数据集。
    """
    filtered_data = []
    for sample in data:
        # 检查样本在目标维度的长度
        if sample.shape[dim] == target_length:
            filtered_data.append(sample)
        else:
            print(f"剔除样本：形状为{sample.shape}，目标维度{dim}长度应为{target_length}")
    return filtered_data


def split_train_val_test(data, test_size, val_test_size, random):
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random)
    val_data, test_data = train_test_split(test_data, test_size=val_test_size, random_state=random)
    return train_data, val_data, test_data


def split_data_batch(data, batch_size):
    train_loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)
    # for train_batch in train_loader:
    #     print(train_batch[0].shape)
    #     break
    return train_loader


def preprocessing_training_data(training_data, test_size=TEST_SIZE, val_test_size=0.5, random=RANDOM_SEED,
                                batch_size=BATCH_SIZE):
    """
    对数据转化为Tensor，进行预处理，划分训练集，测试集，试验集，以及将训练集划分batch，参数为train模块的全局变量设置
    """
    training_data = transform_to_tensor(training_data)
    train_data, val_data, test_data = split_train_val_test(training_data, test_size, val_test_size, random)
    train_data = split_data_batch(train_data, batch_size)
    return train_data, val_data, test_data


def create_train_model(train_data, val_data, model_path, folder_path=None, epochs=EPOCH, device=DEVICE, lr=LR,
                       is_plot_loss=True, loss_picture_name='history.png'):
    model = LstmAutoencoder()
    model = model.to(device)
    history = train_model(model, train_data, val_data, model_path, epochs, device, lr)
    if is_plot_loss:
        try:
            plot_history(history, folder_path, loss_picture_name)
        except Exception as e:
            print(e)
    return history


def train_model(model, train_dataset, val_dataset, path, n_epochs, device, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='sum').to(device)
    #     criterion = nn.L1Loss(reduction='sum').to(device)
    history = dict(train=[], val=[])
    for epoch in range(1, n_epochs + 1):
        model = model.train()
        train_losses = []
        for step, (seq_true,) in enumerate(train_dataset):
            optimizer.zero_grad()
            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)
            loss = criterion(seq_pred, seq_true)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        history['train'].append(np.mean(train_losses))  # 记录训练epoch损失
        if epoch % 20 == 0:
            print("第{}个train_epoch，loss：{}".format(epoch, history['train'][-1]))
        # os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model, path)
        val_losses = []
        model = model.eval()
        with torch.no_grad():
            for seq_true in val_dataset:
                seq_true = seq_true.reshape(1, -1)
                seq_true = seq_true.to(device)
                seq_pred = model(seq_true)
                loss = criterion(seq_pred, seq_true)
                val_losses.append(loss.item())
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['val'].append(val_loss)
    return history


def plot_history(history, save_folder_path, fig_name):
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    history['train'] = np.array(history['train'])
    plt.plot(history['train'] / 50)
    # ax.plot(history['val'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    # plt.legend(['train', 'test'])
    plt.title('Loss over training epochs')
    plt.subplot(122)
    plt.plot(history['val'])
    # ax.plot(history['val'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    # plt.legend(['train', 'test'])
    plt.title('Loss over val epochs')
    save_path = os.path.join(save_folder_path, fig_name)
    plt.savefig(save_path)
    plt.close()


def main():
    pass


if __name__ == '__main__':
    pass
