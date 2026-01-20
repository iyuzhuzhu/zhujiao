from general_functions import functions
from general_functions.models import BasicModel
import torch
import numpy as np
import os
import yaml
from ai.Train_gaf import prepare_data_per_sensor, train_model, DEVICE, IMAGE_SIZE
from ai.ae_model_gaf import MultiModalAutoencoder
from ai.gaf_transform import time_series_to_gaf
from torch.utils.data import DataLoader, TensorDataset
from ai.plot import plot_rec

class AiGaf(BasicModel):
    def __init__(self, name, config_path, shot, model_name='ai_gaf'):
        super().__init__(name, config_path, shot, model_name)
        self.device = DEVICE
        self.ai()

    def ai(self):
        # print(self.config)
        if self.config.get('is_training', False):
            self.train_models()
        
        # Check if running
        rms_record = functions.get_rms_record(self.config['db']['is_running_collection'], self.name, int(self.shot), self.db)
        
        if rms_record and rms_record.get('is_running', False):
            sample_data, sensors_data = functions.get_sensors_data(self.data_source, self.shot, self.name, self.sensors)
            # sensors_data: {sensor: {channel: array}}
            rec_data = self.predict_single_shot(sensors_data)
            self.plot_model(sensors_data, rec_data, sample_data)
            self.single_shot_summary(rec_data, rms_record)

    def get_training_data(self):
        is_running_collection = functions.replace_ball_mill_name(self.config['db']['is_running_collection'], self.name)
        
        # Get shot list first
        # Use int(self.shot) - 1 to ensure training data is strictly before the current shot
        shot_list = functions.get_is_running_shot(self.db, is_running_collection, 
                                                  self.config['training_shots'], 
                                                  int(self.shot) - 1)
        
        # Get data
        running_shots_data = functions.get_shots_raw_data(self.data_source, shot_list, 
                                                          self.name, self.sensors, 
                                                          self.channels)
                                                          
        return running_shots_data, shot_list

    def get_gaf_path(self, sensor, shot):
        gaf_source = self.config.get('gaf_source')
        if not gaf_source:
            return None
        # Replace placeholders
        path = functions.replace_bm_shot_path(gaf_source, self.name, shot)
        path = functions.replace_sensor(path, sensor)
        return path

    def get_or_create_gaf(self, sensor, shot, ch_arrays, image_size=64):
        """
        Get GAF image from disk or create it.
        ch_arrays: list of 1D arrays (original data, not normalized)
        """
        gaf_path = self.get_gaf_path(sensor, shot)
        
        # Try load
        if gaf_path and os.path.exists(gaf_path):
            try:
                img = np.load(gaf_path)
                return img
            except Exception as e:
                print(f"Error loading GAF from {gaf_path}: {e}")
        
        # Create
        gafs = []
        for ch_arr in ch_arrays:
            gaf = time_series_to_gaf(ch_arr, image_size=image_size)
            gafs.append(gaf)
        img = np.stack(gafs, axis=0) # (Ch, H, W)
        
        # Save
        if gaf_path:
            try:
                functions.create_folder(os.path.dirname(gaf_path))
                np.save(gaf_path, img)
            except Exception as e:
                print(f"Error saving GAF to {gaf_path}: {e}")
                
        return img

    def get_save_model_path(self, sensor):
        # Model per sensor
        model_name = f"{sensor}_gaf_model.pth"
        model_path = self.config['model_path']
        model_path = functions.replace_ball_mill_name(model_path, self.name)
        model_path = functions.replace_sensor(model_path, sensor)
        functions.create_folder(model_path)
        full_path = os.path.join(model_path, model_name)
        return full_path

    def save_normalization_params(self, sensor, norm_params):
        norm_config_path = self.config.get('normalization_config_path')
        if not norm_config_path:
            print("normalization_config_path not found in config")
            return
            
        norm_config_path = functions.replace_ball_mill_name(norm_config_path, self.name)
        
        # Read existing or create new
        if os.path.exists(norm_config_path):
            with open(norm_config_path, 'r', encoding='utf-8') as f:
                config = yaml.load(f, Loader=yaml.FullLoader) or {}
        else:
            config = {}
            
        if 'sensors_normalization' not in config:
            config['sensors_normalization'] = {}
            
        config['sensors_normalization'][sensor] = norm_params
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(norm_config_path), exist_ok=True)
        
        with open(norm_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)

    def load_normalization_params(self, sensor):
        norm_config_path = self.config.get('normalization_config_path')
        if not norm_config_path:
            return None
            
        norm_config_path = functions.replace_ball_mill_name(norm_config_path, self.name)
        
        if not os.path.exists(norm_config_path):
            return None
            
        with open(norm_config_path, 'r', encoding='utf-8') as f:
            config = yaml.load(f, Loader=yaml.FullLoader) or {}
            
        return config.get('sensors_normalization', {}).get(sensor)

    def train_models(self):
        running_shots_data, shot_list = self.get_training_data()
        # running_shots_data: {sensor: {channel: [shot1, shot2, ...]}}
        
        for sensor in self.sensors:
            # Reorganize data
            # We need list of dicts: [{'ch1': s1, 'ch2': s1}, {'ch1': s2, 'ch2': s2}, ...]
            sensor_data = running_shots_data[sensor]
            channels = list(sensor_data.keys())
            num_shots = len(sensor_data[channels[0]])
            
            # Filter data by specified length
            target_len = self.config.get('seq_len', 4096)
            sensor_data_list = []
            valid_indices = []
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
                    valid_indices.append(i)
            
            print(f"Sensor {sensor}: {len(sensor_data_list)}/{num_shots} shots used (Length >= {target_len})")
            
            if not sensor_data_list:
                print(f"No valid data for sensor {sensor}")
                continue

            # --- Prepare GAF (Before Normalization) ---
            img_list = []
            for idx, list_idx in enumerate(valid_indices):
                shot = shot_list[list_idx]
                # Get raw data for GAF
                ch_arrays = [sensor_data_list[idx][ch] for ch in channels]
                img = self.get_or_create_gaf(sensor, shot, ch_arrays, image_size=IMAGE_SIZE)
                img_list.append(img)
            img_tensor = torch.tensor(np.array(img_list), dtype=torch.float32)
            # ------------------------------------------

            # --- Normalization ---
            norm_params = {}
            for ch in channels:
                # Gather all data for this channel to compute mean/std
                all_data = np.concatenate([s[ch] for s in sensor_data_list])
                mean = np.mean(all_data)
                std = np.std(all_data)
                if std == 0: std = 1.0
                norm_params[ch] = {'mean': float(mean), 'std': float(std)}
                
                # Apply normalization
                for s in sensor_data_list:
                    s[ch] = (s[ch] - mean) / std
            
            self.save_normalization_params(sensor, norm_params)
            # ---------------------
            
            # Prepare seq tensor
            seq_list = []
            for shot_data in sensor_data_list:
                ch_arrays = [shot_data[ch] for ch in channels]
                seq = np.stack(ch_arrays, axis=1)
                seq_list.append(seq)
            seq_tensor = torch.tensor(np.array(seq_list), dtype=torch.float32)
            
            # Split
            dataset = TensorDataset(seq_tensor, img_tensor)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32)
            
            # Initialize model
            num_channels = len(channels)
            seq_len = seq_tensor.shape[1]
            model = MultiModalAutoencoder(seq_len=seq_len, num_channels=num_channels, image_size=IMAGE_SIZE)
            
            # Train
            model_path = self.get_save_model_path(sensor)
            print(f"Training model for {sensor}...")
            train_model(train_loader, val_loader, model, model_path=model_path)
            
            # Update thresholds
            self.update_thresholds(sensor, model, train_loader, norm_params, channels)

    def update_thresholds(self, sensor, model, data_loader, norm_params=None, channels=None):
        """
        Calculate thresholds based on training data and update config.
        """
        model.eval()
        losses = []
        
        # Prepare std tensor for denormalization/scaling
        std_tensor = None
        if norm_params and channels:
            stds = [norm_params[ch]['std'] for ch in channels]
            std_tensor = torch.tensor(stds, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            for seq, img in data_loader:
                seq, img = seq.to(self.device), img.to(self.device)
                rec_seq, rec_img = model(seq, img)
                
                diff = (rec_seq - seq)
                
                # Scale back to original units if normalization was used
                if std_tensor is not None:
                    diff = diff * std_tensor
                
                diff = diff ** 2 # (Batch, Seq, Ch)
                mse = torch.mean(diff, dim=1) # (Batch, Ch)
                losses.append(mse.cpu().numpy())
                
        losses = np.concatenate(losses, axis=0) # (N, Ch)
        
        # Calculate thresholds for r and z directions
        thresholds = {}
        
        # Map channels to directions
        r_indices = []
        z_indices = []
        
        for i, ch_config in enumerate(self.config['channels']):
            direction = list(ch_config.values())[0]
            if direction == 'r':
                r_indices.append(i)
            elif direction == 'z':
                z_indices.append(i)
                
        # Calculate R statistics
        if r_indices:
            r_losses = losses[:, r_indices] # (N, num_r)
            r_mean_loss = np.mean(r_losses, axis=1) # (N,) Average across channels for each sample
            
            mean_r = np.mean(r_mean_loss)
            std_r = np.std(r_mean_loss)
            
            thresholds['of_h_r'] = float(mean_r + 3 * std_r)
            thresholds['of_hh_r'] = float(mean_r + 5 * std_r)
        else:
            thresholds['of_h_r'] = 0.0
            thresholds['of_hh_r'] = 0.0

        # Calculate Z statistics
        if z_indices:
            z_losses = losses[:, z_indices]
            z_mean_loss = np.mean(z_losses, axis=1)
            
            mean_z = np.mean(z_mean_loss)
            std_z = np.std(z_mean_loss)
            
            thresholds['of_h_z'] = float(mean_z + 3 * std_z)
            thresholds['of_hh_z'] = float(mean_z + 5 * std_z)
        
        # Update config file
        self.save_thresholds(sensor, thresholds)

    def save_thresholds(self, sensor, thresholds):
        threshold_config_path = self.config['threshold_config_path']
        threshold_config_path = functions.replace_ball_mill_name(threshold_config_path, self.name)
        
        # Read existing config
        if os.path.exists(threshold_config_path):
            with open(threshold_config_path, 'r', encoding='utf-8') as f:
                config = yaml.load(f, Loader=yaml.FullLoader) or {}
        else:
            config = {}
            
        if 'sensors_threshold' not in config:
            config['sensors_threshold'] = {}
            
        if sensor not in config['sensors_threshold']:
            config['sensors_threshold'][sensor] = {}
            
        # Update values
        for key, value in thresholds.items():
            config['sensors_threshold'][sensor][key] = value
            
        # Save config
        with open(threshold_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)
            
    def load_model(self, sensor, num_channels, seq_len):
        model_path = self.get_save_model_path(sensor)
        model = MultiModalAutoencoder(seq_len=seq_len, num_channels=num_channels, image_size=IMAGE_SIZE)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def predict_single_shot(self, sensors_data):
        # sensors_data: {sensor: {channel: array}}
        rec_data = {}
        
        for sensor in self.sensors:
            rec_data[sensor] = {}
            sensor_data = sensors_data[sensor]
            channels = list(sensor_data.keys())
            
            # Load normalization params
            norm_params = self.load_normalization_params(sensor)
            
            # Prepare input
            ch_arrays_raw = [] # For GAF
            ch_arrays_norm = [] # For Seq
            
            for ch in channels:
                data = sensor_data[ch]
                ch_arrays_raw.append(data)
                
                if norm_params and ch in norm_params:
                    mean = norm_params[ch]['mean']
                    std = norm_params[ch]['std']
                    data = (data - mean) / std
                ch_arrays_norm.append(data)
            
            # Seq (Normalized)
            seq = np.stack(ch_arrays_norm, axis=1)
            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # GAF (Raw -> Load/Create)
            img = self.get_or_create_gaf(sensor, self.shot, ch_arrays_raw, image_size=IMAGE_SIZE)
            img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Load model
            try:
                model = self.load_model(sensor, len(channels), seq.shape[0])
            except FileNotFoundError:
                print(f"Model for {sensor} not found. Skipping.")
                continue
                
            # Predict
            with torch.no_grad():
                rec_seq, rec_img = model(seq_tensor, img_tensor)
                
                # Store results
                rec_seq_np = rec_seq.cpu().numpy().squeeze(0) # (Seq, Ch)
                seq_np = seq # Normalized input
                
                for i, ch in enumerate(channels):
                    rec_data[sensor][ch] = {}
                    
                    rec_ch = rec_seq_np[:, i]
                    orig_ch = seq_np[:, i]
                    
                    # Denormalize for loss calculation and plotting
                    if norm_params and ch in norm_params:
                        mean = norm_params[ch]['mean']
                        std = norm_params[ch]['std']
                        rec_ch = rec_ch * std + mean
                        orig_ch = orig_ch * std + mean
                    
                    # Per channel loss (in original scale)
                    ch_loss = np.mean((rec_ch - orig_ch)**2)
                    rec_data[sensor][ch]['loss'] = float(ch_loss)
                    rec_data[sensor][ch]['pre'] = rec_ch
                    
        return rec_data

    def plot_model(self, sensors_data, rec_data, sample_data, drop_last=True):
        """
        绘制振动波形图像，并复制plot_desc文件
        """
        plot_config_data = self.get_plot_config()
        save_plot_folder = plot_config_data['save_plot_folder']
        output_path = functions.create_output_folder(self.config['Inference_path'], self.shot, self.name)
        output_folder = os.path.join(output_path, save_plot_folder)
        functions.create_folder(output_folder)
        
        for plot_data in plot_config_data['plots']:
            sensor = plot_data['name']
            channel = plot_data['plot_channel']
            if sensors_data[sensor] is not None:
                time, true_data = self.get_plot_x_y(sample_data['SampleRate'], sensors_data[sensor][channel], drop_last)
                rec_channel = rec_data[sensor][channel]['pre']
                
                # Ensure rec_channel matches true_data length
                if len(rec_channel) > len(true_data):
                    rec_channel = rec_channel[:len(true_data)]
                
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
            rec_loss = float(data[channel]['loss'])
            axis = 'axis' + '_' + str(channel_num+1) + '_' + self.model_name  # axis_1_ai
            single_sensor_data[axis] = rec_loss
            if self.config['channels'][channel_num]['channel' + str(channel_num)] == 'r':
                r_ai += rec_loss
                r_num += 1
            elif self.config['channels'][channel_num]['channel' + str(channel_num)] == 'z':
                z_ai += rec_loss
                z_num += 1
            channel_num += 1
        single_sensor_data['r_' + self.model_name] = r_ai / r_num if r_num > 0 else 0
        single_sensor_data['z_' + self.model_name] = z_ai / z_num if z_num > 0 else 0
        single_sensor_data = self.default_single_sensor_template(single_sensor_data, self.model_name)
        return single_sensor_data

    def calculate_single_sensors_ai(self, rec_data):
        """
        得到所有传感器的rms值和温度
        """
        sensors_ai = {}
        for sensor in self.sensors:
            try:
                sensors_ai[sensor] = self.calculate_single_sensor_ai(rec_data[sensor])
            except Exception as e:
                print(f"Error calculating AI for {sensor}: {e}")
                sensors_ai[sensor] = self.get_single_sensor_result(err=True)
        return sensors_ai

    def single_shot_sensors_summary(self, rec_data, is_running: bool):
        """
        将模型单独炮的数据分析结果汇总
        """
        sensors = {}
        sensors_ai = self.calculate_single_sensors_ai(rec_data)
        for sensor, single_sensor in sensors_ai.items():
            try:
                if is_running:
                    sensors[sensor] = self.get_alarm(sensor, single_sensor)
            except Exception as e:
                print(f"Error getting alarm for {sensor}: {e}")
                sensors[sensor] = self.get_single_sensor_result(err=True)
        return sensors

    def single_shot_summary(self, rec_data, rms_record):
        """
        汇总当前summary信息
        """
        single_shot_summary = rms_record
        sensors = self.single_shot_sensors_summary(rec_data,  rms_record['is_running'])
        single_shot_summary['sensors'] = sensors
        functions.save_summary_mongodb(single_shot_summary, self.db, self.collection_name, self.shot)
        return single_shot_summary

def main():
    config_path = './config.yml'
    name = 'bm1'
    shot = '1110400'
    AiGaf(name, config_path, shot)

if __name__ == "__main__":
    main()
