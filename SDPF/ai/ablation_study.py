import torch
import torch.nn as nn
import numpy as np
import os
import yaml
from torch.utils.data import DataLoader, TensorDataset
from general_functions import functions
from general_functions.database_data import connect_mongodb_database

from general_functions.models import BasicModel
from ai.main_gaf import AiGaf
from ai.ae_model_gaf import Cnn1dEncoder, Cnn1dDecoder
import matplotlib.pyplot as plt

Epoch = 100
LR = 1e-4
# --- 1. 定义仅包含序列分支的模型 ---
class SeqOnlyAutoencoder(nn.Module):
    def __init__(self, seq_len, num_channels, lstm_hidden=256, joint_dim=128, dropout_p=0.2):
        super(SeqOnlyAutoencoder, self).__init__()
        
        # Encoder
        self.seq_encoder = Cnn1dEncoder(in_channels=num_channels, latent_dim=lstm_hidden, seq_len=seq_len)
        
        self.dropout = nn.Dropout(p=dropout_p)
        
        # Bottleneck
        self.bottleneck = nn.Linear(lstm_hidden, joint_dim)
        self.expand = nn.Linear(joint_dim, lstm_hidden)
        
        # Decoder
        self.seq_decoder = Cnn1dDecoder(latent_dim=lstm_hidden, out_channels=num_channels, seq_len=seq_len)

    def forward(self, x_seq):
        # x_seq: (Batch, Seq_Len, Num_Channels)
        
        # Permute for Conv1d: (Batch, Num_Channels, Seq_Len)
        x_seq = x_seq.permute(0, 2, 1)
        
        h_seq = self.seq_encoder(x_seq)
        h_seq = self.dropout(h_seq)
        
        # Bottleneck
        h_latent = torch.nn.functional.leaky_relu(self.bottleneck(h_seq), negative_slope=0.1)
        h_latent = self.dropout(h_latent)
        
        h_expand = self.expand(h_latent)
        
        rec_seq = self.seq_decoder(h_expand)
        
        # Permute back: (Batch, Seq_Len, Num_Channels)
        rec_seq = rec_seq.permute(0, 2, 1)
        
        return rec_seq

# --- 2. 训练与测试类 ---
class AblationStudy(AiGaf):
    def __init__(self, name, config_path, shot, model_name='ai_ablation'):
        # 不调用 super().__init__ 以避免自动执行 ai()
        self.date_time, self.is_running = None, None
        self.name, self.shot, self.model_name, self.config_path = name, shot, model_name, config_path
        self.config, self.ruamel_yaml = functions.load_yaml(self.config_path)
        print(f"Loaded config from {self.config_path}: {self.config.keys()}")
        self.data_source, self.output_path, self.sensors = (self.config['data_source'], self.config['Inference_path'],
                                                            self.config['sensors'])
        self.collection_name = functions.replace_ball_mill_name(self.config['db']['collection'], self.name)
        self.client, self.db = connect_mongodb_database(self.config['db']['connection'], self.config['db']['db_name'])
        self.channels = self.get_sensor_channels()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_ablation_model_path(self, sensor):
        model_name = f"{sensor}_seq_only_model.pth"
        model_path = self.config['model_path']
        model_path = functions.replace_ball_mill_name(model_path, self.name)
        model_path = functions.replace_sensor(model_path, sensor)
        functions.create_folder(model_path)
        full_path = os.path.join(model_path, model_name)
        return full_path

    def train_seq_only(self):
        print("=== Starting Ablation Training (Sequence Only) ===")
        running_shots_data, _ = self.get_training_data()
        
        for sensor in self.sensors:
            sensor_data = running_shots_data[sensor]
            channels = list(sensor_data.keys())
            num_shots = len(sensor_data[channels[0]])
            
            # Filter data
            target_len = self.config.get('seq_len', 4096)
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
            
            # Prepare Tensor
            seq_list = []
            for shot_data in sensor_data_list:
                ch_arrays = [shot_data[ch] for ch in channels]
                seq = np.stack(ch_arrays, axis=1)
                seq_list.append(seq)
            seq_tensor = torch.tensor(np.array(seq_list), dtype=torch.float32)
            
            # Dataset
            dataset = TensorDataset(seq_tensor)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32)
            
            # Model
            num_channels = len(channels)
            model = SeqOnlyAutoencoder(seq_len=target_len, num_channels=num_channels)
            model.to(self.device)
            
            # Training Loop
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
            criterion = nn.MSELoss()
            best_val_loss = float('inf')
            epochs = 50
            patience = 15
            counter = 0
            
            print(f"Training Seq-Only Model for {sensor}...")
            for epoch in range(epochs):
                model.train()
                train_loss = 0
                for (seq,) in train_loader:
                    seq = seq.to(self.device)
                    optimizer.zero_grad()
                    rec_seq = model(seq)
                    loss = criterion(rec_seq, seq)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                train_loss /= len(train_loader)
                
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for (seq,) in val_loader:
                        seq = seq.to(self.device)
                        rec_seq = model(seq)
                        loss = criterion(rec_seq, seq)
                        val_loss += loss.item()
                val_loss /= len(val_loader)
                
                # Update scheduler
                scheduler.step(val_loss)
                
                print(f"Epoch {epoch+1}/{epochs}, Train: {train_loss:.4f}, Val: {val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), self.get_ablation_model_path(sensor))
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        print(f"Early stopping triggered at epoch {epoch+1}")
                        break
            
            self.norm_params = norm_params

    def compare_models(self, test_shots):
        print("\n=== Starting Comparison (GAF vs Seq-Only vs LSTM) ===")
        
        # Load Models
        sensor = self.sensors[0] # Assume 1 sensor for now
        target_len = self.config.get('seq_len', 4096)
        
        # 1. Load GAF Model
        gaf_model_path = self.get_save_model_path(sensor)
        gaf_model = self.load_model(sensor, len(self.channels), target_len) # Uses main_gaf's load_model
        
        # 2. Load Seq-Only Model
        seq_model_path = self.get_ablation_model_path(sensor)
        seq_model = SeqOnlyAutoencoder(seq_len=target_len, num_channels=len(self.channels))
        seq_model.load_state_dict(torch.load(seq_model_path, map_location=self.device))
        seq_model.to(self.device)
        seq_model.eval()
        
        # Load Norm Params
        norm_params = self.load_normalization_params(sensor)
        
        results = {'shot': [], 'gaf_loss': [], 'seq_loss': []}
        
        for shot in test_shots:
            # Get Data
            try:
                sample_data, sensors_data = functions.get_sensors_data(self.data_source, str(shot), self.name, self.sensors)
            except:
                print(f"Shot {shot} data not found.")
                continue
                
            if sensors_data[sensor] is None: continue
            
            # Prepare Input
            channels = list(sensors_data[sensor].keys())
            ch_arrays_raw = []
            ch_arrays_norm = []
            
            for ch in channels:
                data = sensors_data[sensor][ch]
                if len(data) > target_len: data = data[:target_len]
                elif len(data) < target_len: 
                     pad_len = target_len - len(data)
                     data = np.pad(data, (0, pad_len), mode='constant')
                
                ch_arrays_raw.append(data)
                
                if norm_params and ch in norm_params:
                    mean = norm_params[ch]['mean']
                    std = norm_params[ch]['std']
                    data = (data - mean) / std
                ch_arrays_norm.append(data)
            
            # Tensors
            seq = np.stack(ch_arrays_norm, axis=1)
            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # GAF Input
            from ai.gaf_transform import time_series_to_gaf
            gafs = []
            for ch_arr in ch_arrays_raw:
                gaf = time_series_to_gaf(ch_arr, image_size=64)
                gafs.append(gaf)
            img = np.stack(gafs, axis=0)
            img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                # GAF Model
                rec_seq_gaf, _ = gaf_model(seq_tensor, img_tensor)
                loss_gaf = torch.mean((rec_seq_gaf - seq_tensor)**2).item()
                
                # Seq Only Model
                rec_seq_only = seq_model(seq_tensor)
                loss_seq_only = torch.mean((rec_seq_only - seq_tensor)**2).item()
            
            print(f"Shot {shot}: GAF={loss_gaf:.4f}, Seq={loss_seq_only:.4f}")
            results['shot'].append(str(shot))
            results['gaf_loss'].append(loss_gaf)
            results['seq_loss'].append(loss_seq_only)
            
        # Plot Comparison
        plt.figure(figsize=(10, 6))
        plt.plot(results['shot'], results['gaf_loss'], label='With GAF (Multi-modal)', marker='o')
        plt.plot(results['shot'], results['seq_loss'], label='Without GAF (Seq Only)', marker='x')
        plt.xlabel('Shot')
        plt.ylabel('MSE Loss (Normalized)')
        plt.title('Ablation Study: Reconstruction Loss Comparison')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('ablation_study_result.png')
        print("Comparison plot saved to ablation_study_result.png")

def main():
    config_path = 'ai/config.yml'
    name = 'bm1'
    shot = '1110400' # Current shot
    
    study = AblationStudy(name, config_path, shot)
    
    # 1. Train the Seq-Only Model
    study.train_seq_only()
    
    # 3. Compare on subsequent shots
    test_shots = range(1110401, 1110451) # Test 50 shots
    study.compare_models(test_shots)

if __name__ == "__main__":
    main()
