from general_functions import functions
from general_functions.models import BasicModel
import torch
import numpy as np
import os
import yaml
from ai.Train import create_train_model
from ai.Inference import predict_single
from pathlib import Path
from ai.plot import plot_rec
from alarmSystem.Data.db.collectionDB import CollectionDB
from torch.utils.data import DataLoader, TensorDataset


class Ai(BasicModel):
    def __init__(self, name, config_path, shot, model_name='ai'):
        super().__init__(name, config_path, shot, model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.train_models()
        self.ai()
        # self.predict_single_axis_loss()

    def ai(self):
        if self.config['is_training']:
            self.train_models()
        rms_record = functions.get_rms_record(self.config['db']['is_running_collection'], self.name, int(self.shot),
                                              self.db)
        if rms_record and rms_record.get('is_running', False):
            sample_data, sensors_data = functions.get_sensors_data(self.data_source, self.shot, self.name, self.sensors)
            rec_data = self.predict_single_shot(sensors_data)
            self.plot_model(sensors_data, rec_data, sample_data)
            self.single_shot_summary(rec_data, rms_record)

    def get_training_data(self):
        is_running_collection = functions.replace_ball_mill_name(self.config['db']['is_running_collection'], self.name)
        # print(is_running_collection, self.shot)
        running_shots_data = functions.get_is_running_shots_data(self.db, is_running_collection,
                                                                 self.config['training_shots'],
                                                                 int(self.shot), self.data_source, self.name,
                                                                 self.sensors,
                                                                 self.channels)
        return running_shots_data

    def get_save_model_path(self, sensor):
        """
        得到model的完整保存路径（包括模型的文件名称）
        """
        model_name = f"{sensor}_lstm_model.pth"
        model_path = self.config['model_path']
        model_path = functions.replace_ball_mill_name(model_path, self.name)
        model_path = functions.replace_sensor(model_path, sensor)
        functions.create_folder(model_path)
        folder_path = model_path
        model_path = os.path.join(model_path, model_name)
        return model_path, folder_path

    def save_normalization_params(self, sensor, norm_params):
        norm_config_path = self.config.get('normalization_config_path')
        if not norm_config_path:
            # Fallback or create default path if not in config
            norm_config_path = os.path.join(os.path.dirname(self.config_path), 'normalization_config.yml')
            
        norm_config_path = functions.replace_ball_mill_name(norm_config_path, self.name)
        
        if os.path.exists(norm_config_path):
            with open(norm_config_path, 'r', encoding='utf-8') as f:
                config = yaml.load(f, Loader=yaml.FullLoader) or {}
        else:
            config = {}
            
        if 'sensors_normalization' not in config:
            config['sensors_normalization'] = {}
            
        config['sensors_normalization'][sensor] = norm_params
        
        os.makedirs(os.path.dirname(norm_config_path), exist_ok=True)
        
        with open(norm_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)

    def load_normalization_params(self, sensor):
        norm_config_path = self.config.get('normalization_config_path')
        if not norm_config_path:
             norm_config_path = os.path.join(os.path.dirname(self.config_path), 'normalization_config.yml')
            
        norm_config_path = functions.replace_ball_mill_name(norm_config_path, self.name)
        
        if not os.path.exists(norm_config_path):
            return None
            
        with open(norm_config_path, 'r', encoding='utf-8') as f:
            config = yaml.load(f, Loader=yaml.FullLoader) or {}
            
        return config.get('sensors_normalization', {}).get(sensor)

    def train_models(self):
        """
        训练针对于球磨机的每个sensor的所有channel统一训练
        """
        running_shots_data = self.get_training_data()
        
        for sensor in self.sensors:
            sensor_data = running_shots_data[sensor]
            channels = list(sensor_data.keys())
            num_shots = len(sensor_data[channels[0]])
            
            # Filter data by length
            target_len = 4096 # Default length
            sensor_data_list = []
            for i in range(num_shots):
                shot_dict = {}
                is_valid = True
                for ch in channels:
                    data = sensor_data[ch][i]
                    if len(data) < target_len:
                        is_valid = False
                        break
                    shot_dict[ch] = data[:target_len]
                if is_valid:
                    sensor_data_list.append(shot_dict)
            
            print(f"Sensor {sensor}: {len(sensor_data_list)}/{num_shots} shots used")
            if not sensor_data_list:
                continue

            # Normalization
            norm_params = {}
            for ch in channels:
                all_data = np.concatenate([s[ch] for s in sensor_data_list])
                mean = np.mean(all_data)
                std = np.std(all_data)
                if std == 0: std = 1.0
                norm_params[ch] = {'mean': float(mean), 'std': float(std)}
                for s in sensor_data_list:
                    s[ch] = (s[ch] - mean) / std
            
            self.save_normalization_params(sensor, norm_params)
            
            # Prepare Tensor: (Batch, Seq_Len, Num_Channels)
            seq_list = []
            for shot_data in sensor_data_list:
                ch_arrays = [shot_data[ch] for ch in channels]
                seq = np.stack(ch_arrays, axis=1) # (Seq_Len, Num_Channels)
                seq_list.append(seq)
            seq_tensor = torch.tensor(np.array(seq_list), dtype=torch.float32)
            
            # Dataset
            dataset = TensorDataset(seq_tensor)
            train_size = int(0.9 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            
            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=128, num_workers=4, pin_memory=True)
            
            model_path, folder_path = self.get_save_model_path(sensor)
            
            # Train
            create_train_model(train_loader, val_loader, model_path, folder_path, 
                               seq_len=target_len, num_channels=len(channels))

    def load_model(self, sensor):
        """
        加载对应sensor的模型
        """
        model_path, folder_path = self.get_save_model_path(sensor)
        
        # 安全处理: 如果是新版PyTorch (2.6+), 需要处理 weights_only=True 的默认行为
        # 这里我们显式设置 weights_only=False，因为我们加载的是我们自己训练的模型
        # 且包含了自定义的模型类结构 (CnnAutoencoder)
        try:
            return torch.load(model_path, map_location=self.device, weights_only=False)
        except TypeError:
            # 兼容旧版本 PyTorch (不支持 weights_only 参数)
             return torch.load(model_path, map_location=self.device)

    def predict_single_shot(self, sensors_data):
        rec_data = {}
        for sensor in self.sensors:
            rec_data[sensor] = {}
            sensor_data = sensors_data[sensor]
            channels = list(sensor_data.keys())
            
            # Load normalization params
            norm_params = self.load_normalization_params(sensor)
            
            # Prepare input
            ch_arrays_norm = []
            for ch in channels:
                data = sensor_data[ch]
                if norm_params and ch in norm_params:
                    mean = norm_params[ch]['mean']
                    std = norm_params[ch]['std']
                    data = (data - mean) / std
                ch_arrays_norm.append(data)
            
            # Stack: (1, Seq_Len, Num_Channels)
            seq = np.stack(ch_arrays_norm, axis=1)
            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Load model
            try:
                model = self.load_model(sensor)
                model.eval()
            except FileNotFoundError:
                print(f"Model for {sensor} not found. Skipping.")
                continue
            
            # Predict
            with torch.no_grad():
                rec_seq = model(seq_tensor)
                rec_seq_np = rec_seq.cpu().numpy().squeeze(0) # (Seq_Len, Num_Channels)
                
                for i, ch in enumerate(channels):
                    rec_data[sensor][ch] = {}
                    rec_ch = rec_seq_np[:, i]
                    orig_ch = seq[:, i] # Normalized input
                    
                    # Denormalize for loss calculation (optional, but usually we want loss in original scale)
                    if norm_params and ch in norm_params:
                        mean = norm_params[ch]['mean']
                        std = norm_params[ch]['std']
                        rec_ch = rec_ch * std + mean
                        orig_ch = orig_ch * std + mean
                    
                    loss = np.mean((rec_ch - orig_ch)**2)
                    rec_data[sensor][ch]['loss'] = float(loss)
                    rec_data[sensor][ch]['pre'] = rec_ch
                    
        return rec_data

    def plot_model(self, sensors_data, rec_data, sample_data, drop_last=True):
        """
        绘制振动波形图像，并复制plot_desc文件
        """
        # sample_data, sensors_data = functions.get_sensors_data(self.data_source, self.shot, self.name, self.sensors)
        plot_config_data = self.get_plot_config()
        save_plot_folder = plot_config_data['save_plot_folder']
        output_path = functions.create_output_folder(self.config['Inference_path'], self.shot, self.name)
        output_folder = os.path.join(output_path, save_plot_folder)
        functions.create_folder(output_folder)
        # print(rec_data)
        # plots.copy_and_rename_file(self.config['plot_desc_path'], output_folder,
        #                            self.config['plot_desc_file'])  # 复制plot_desc
        for plot_data in plot_config_data['plots']:
            sensor = plot_data['name']
            channel = plot_data['plot_channel']
            if sensors_data[sensor] is not None:
                time, true_data = self.get_plot_x_y(sample_data['SampleRate'], sensors_data[sensor][channel], drop_last)
                rec_channel = rec_data[sensor][channel]['pre']
                plot_rec(time, true_data, rec_channel, plot_data, output_folder)

    @staticmethod
    def get_plot_x_y(fs, channel_data, drop_last=True):
        """
        得到需要绘制的振动波形图像的x,y
        """
        time, vibration = None, None
        if channel_data is not None:
            if drop_last:
                vibration = channel_data[:-1]
            else:
                vibration = channel_data
            time = np.arange(0, len(vibration), 1) / fs  # 得到时间参数
        return time, vibration

    def get_plot_config(self):
        """
        读取绘图配置文件
        """
        plot_config_path = self.config['plot_config_path']
        plot_config_data = functions.read_config(plot_config_path)
        return plot_config_data

    @staticmethod
    def get_single_sensor_result(single_sensor_rms=None, err=False):
        """
        返回单个sensor计算rms的结果，如果sensor计算rms报错则返回各个结果为None的字典
        :param single_sensor_rms: 正常情况下，没有计算报错得到的sensor_rms数据
        :param err: 有没有发生计算故障 False为未发生故障
        :return: 单个sensor的result
        """
        sensor_result = {
            "axis_1_ai": None,
            "axis_2_ai": None,
            "axis_3_ai": None,
            "r_ai": None,
            "z_ai": None,
            "r_ai_alarm": 0,  # 无警报，H=1,HH=2
            "z_ai_alarm": 0,  # 1级警报
        }
        if not err:
            sensor_result.update(single_sensor_rms)
        return sensor_result

    @staticmethod
    def default_single_sensor_template(single_sensor_data, model_name):
        """
        预设置所有的报警等级为0
        :param model_name: 模型名称
        :param single_sensor_data:单个传感器的数据分析结果
        :return:
        """
        single_sensor_data['r_' + model_name + "_alarm"] = 0
        single_sensor_data['z_' + model_name + "_alarm"] = 0
        return single_sensor_data

    def calculate_single_sensor_ai(self, single_sensor_rec_data):
        """
        :param single_sensor_rec_data: 传感器的重构数据，包括重构波形和损失
        :return: 单个传感器的(channel0 1 2)与轴向和径向的rms字典和温度
        :return: rms数据字典
        """
        single_sensor_data = {}
        data = single_sensor_rec_data
        channel_num, r_ai, r_num, z_ai, z_num = 0, 0, 0, 0, 0
        for channel in data.keys():
            # axis = functions.channel_to_axis(self.config['channels'], channel)
            rec_loss = float(data[channel]['loss'])
            axis = 'axis' + '_' + str(channel_num+1) + '_' + self.model_name  # axis_1_ai
            single_sensor_data[axis] = rec_loss
            if self.config['channels'][channel_num]['channel' + str(channel_num)] == 'r':
                r_ai += rec_loss
                r_num += 1
            elif self.config['channels'][channel_num]['channel' + str(channel_num)] == 'z':
                z_ai = rec_loss
                z_num += 1
            channel_num += 1
        single_sensor_data['r_' + self.model_name] = r_ai / r_num
        single_sensor_data['z_' + self.model_name] = z_ai / z_num
        single_sensor_data = self.default_single_sensor_template(single_sensor_data, self.model_name)
        # print(single_sensor_data)
        return single_sensor_data

    def calculate_single_sensors_ai(self, rec_data):
        """
        得到所有传感器的rms值和温度
        :return:
        """
        sensors_ai = {}
        # sample_data, data = functions.get_single_sensor_data(self.data_source, self.shot, self.name, sensor)
        for sensor in self.sensors:
            try:
                # sensor_data = rec_data[sensor]
                sensors_ai[sensor] = self.calculate_single_sensor_ai(rec_data[sensor])
            except Exception as e:
                # print(e)
                sensors_ai[sensor] = self.get_single_sensor_result(err=True)
        return sensors_ai

    def single_shot_sensors_summary(self, rec_data, is_running: bool):
        """
        将模型单独炮的数据分析结果汇总
        :return:
        """
        sensors = {}
        sensors_ai = self.calculate_single_sensors_ai(rec_data)
        for sensor, single_sensor in sensors_ai.items():
            # if is_running:
            #     sensors[sensor] = self.get_alarm(sensor, single_sensor)
            # 防止出现传感器出现故障，导致数据没有被采集的故障报错导致程序中断运行
            try:
                if is_running:
                    sensors[sensor] = self.get_alarm(sensor, single_sensor)
            except Exception as e:
                # print(e)
                sensors[sensor] = self.get_single_sensor_result(err=True)
        return sensors

    def single_shot_summary(self, rec_data, rms_record):
        """
        汇总当前summary信息
        :return:
        """
        single_shot_summary = rms_record
        sensors = self.single_shot_sensors_summary(rec_data,  rms_record['is_running'])
        single_shot_summary['sensors'] = sensors
        # self.date_time = functions.get_sample_time(sample_data)
        # single_shot_summary = functions.single_shot_summary(self.name, self.shot, sensors, self.date_time
        #                                                     , is_running)
        # functions.save_single_summary_mongodb(single_shot_summary, address, collection_name, self.shot,
        #                                       database_name=self.config['db']['db_name'])
        # print(single_shot_summary)
        functions.save_summary_mongodb(single_shot_summary, self.db, self.collection_name, self.shot)
        return single_shot_summary


def test_shots_calculate():
    # # 输入参数
    # config_path, name, shot = functions.get_input_params('rms')
    config_path = './config.yml'
    name = 'bm1'
    config = functions.read_config(config_path)
    # 得到bail_mill中的bail_name
    shots = np.arange(1108200, 1110400)
    # shots = np.arange(1012849, 1110500)
    for shot in shots:
        shot = str(shot)
        Ai(name, config_path, shot)
        print(shot)


def main():
    # test_shots_calculate()
    # # 输入参数
    # config_path, name, shot = functions.get_input_params('ai')
    config_path = './config.yml'
    name = 'bm1'
    shot = '1110400'
    Ai(name, config_path, shot)


if __name__ == "__main__":
    main()
    # test_shots_calculate()