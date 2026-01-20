# 振动与GAF深度自编码器异常检测方案

整体实现流程参考ai/main.py文件（任务运行输入参考）
整体类可以继承general_functions/models.BasicModel对象
本次项目尽可能采用已有的函数，提高代码复用率


配置文件格式如下：
```yml
data_source: D:/资源汇总/新球磨项目/Data/Daq/$bm$/$shot_2$00/$sensor$/$shot$.hdf5 # 数据地址
Inference_path: D:/资源汇总/新球磨项目/Data/Inference/$bm$/$shot_2$00/$shot$ # 输出地址
threshold_config_path: D:/资源汇总/新球磨项目/ballmill_project/code/SDPF/configs/threshold_configs/$bm$/ai_threshold_config.yml
plot_config_path: D:/资源汇总/新球磨项目/ballmill_project/code/SDPF/configs/plot_configs/ai_plot_config/ai_plot_config.yml
model_path: D:/bm_model/Train/$bm$/$sensor$ # 输出地址
model_name: $channel$.pth
#plot_config_path: E:/资源汇总/新球磨项目/configs/plot_configs/basic_plot_config/basic_plot_config.yml
#plot_desc_path: E:/资源汇总/新球磨项目/configs/plot_configs/basic_plot_config/basic_plot_desc.yml
#plot_desc_file: plot_desc.yml # 复制的画图配置文件的名称
is_training: False # 是否自动训练与更新警报阈值
training_shots: 5000

db: # 数据库的连接信息
  connection: mongodb://localhost:27017/
  db_name: bm
  collection: $bm$_ai
  is_running_collection: $bm$_rms

channels:
  - channel0: r
  - channel1: r
  - channel2: z

sensors:
  - sensor1     # the prefix for this sensor, all fields of this sensor will have this prefix
#  - sensor2     # the prefix for this sensor, all fields of this sensor will have this prefix
#  - sensor3     # the prefix for this sensor, all fields of this sensor will have this prefix
#  - sensor4     # the prefix for this sensor, all fields of this sensor will have this prefix
#  - sensor5     # the prefix for this sensor, all fields of this sensor will have this prefix
#  - sensor6     # the prefix for this sensor, all fields of this sensor will have this prefix
```

实现以下内容
- 数据读取：读取振动hdf5文件获得原始数据
- 数据预处理：将振动数据转化为GAF，与振动原始数据一起作为自编码器输入
- 自编码器模型搭建：整体上先对原始数据和GAF分别进行特征提取，随后整合特征，将联合特征作为解码器解码重构原始数据和GAF
- 模型训练：训练根据配置文件和输入的炮号，将炮号前的运行中的振动数据作为输入，损失则用GAF和原始数据的重构损失
- 阈值确定：根据历史运行的loss设置阈值（这里仅将振动原始数据作为阈值评判标准）
- 结果留存：将振动数据损失以原先格式上传至mongodb

## 数据读取

数据读取以下程序完成:
```python
sample_data, sensors_data = functions.get_sensors_data(self.data_source, self.shot, self.name, self.sensors)
# 返回数据格式如下
{'sensor1': {'channel0': array([-7.32421875e-03, -2.19726562e-02, -1.95312500e-02, ...,
        9.03320312e-03, -3.41796875e-03,  3.83750000e+01], shape=(4097,)), 'channel1': array([-9.52148438e-03,  9.27734375e-03,  7.81250000e-03, ...,
       -2.00195312e-02, -3.41796875e-03,  3.83750000e+01], shape=(4097,)), 'channel2': array([ 2.29492188e-02,  6.83593750e-03, -6.59179688e-03, ...,
        1.24511719e-02,  1.44042969e-02,  3.83750000e+01], shape=(4097,))}}
```

## 数据预处理

通过python包完成将时序数据转化为GAF的函数，读取数据后将所有传感器振动采集数据与GAF联合输入神经网络

## 模型搭建

模型针对于多传感器数据采用LSTM编码器，GAF采用CNN编码器，将特征提取后的特征联合，重构时序数据与GAF图，损失包括时序数据mse和GAF
（为了便于对比仅用传感器数据训练的模型，希望可以单独查看时序数据重构损失）




