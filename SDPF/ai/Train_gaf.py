import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from ai.gaf_transform import time_series_to_gaf
from ai.ae_model_gaf import MultiModalAutoencoder
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4
IMAGE_SIZE = 64

def prepare_data_per_sensor(sensor_data_list, channels, image_size=IMAGE_SIZE):
    """
    Args:
        sensor_data_list: List of dictionaries, each dict is one shot's data for a sensor.
                          e.g. [{'channel0': arr, 'channel1': arr}, ...]
        channels: List of channel names.
    Returns:
        seq_tensor: (N, Seq_Len, Num_Channels)
        img_tensor: (N, Num_Channels, H, W)
    """
    seq_list = []
    img_list = []
    
    for shot_data in sensor_data_list:
        # Stack channels for sequence
        # shot_data[ch] is (4096,)
        # We want (4096, Num_Channels)
        ch_arrays = [shot_data[ch] for ch in channels]
        seq = np.stack(ch_arrays, axis=1) # (4096, Num_Channels)
        
        # Create GAF for each channel
        gafs = []
        for ch_arr in ch_arrays:
            gaf = time_series_to_gaf(ch_arr, image_size=image_size)
            gafs.append(gaf)
        img = np.stack(gafs, axis=0) # (Num_Channels, H, W)
        
        seq_list.append(seq)
        img_list.append(img)
        
    seq_tensor = torch.tensor(np.array(seq_list), dtype=torch.float32)
    img_tensor = torch.tensor(np.array(img_list), dtype=torch.float32)
    
    return seq_tensor, img_tensor

def train_model(train_loader, val_loader, model, epochs=EPOCHS, lr=LR, device=DEVICE, model_path=None):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion_seq = nn.MSELoss()
    criterion_img = nn.MSELoss() # Or L1Loss
    
    best_val_loss = float('inf')
    
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for seq, img in train_loader:
            seq, img = seq.to(device), img.to(device)
            
            optimizer.zero_grad()
            rec_seq, rec_img = model(seq, img)
            
            loss_seq = criterion_seq(rec_seq, seq)
            loss_img = criterion_img(rec_img, img)
            # Increase weight for sequence loss to force model to learn waveform
            loss = 50.0 * loss_seq + loss_img 
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_loss_seq = 0
        val_loss_img = 0
        with torch.no_grad():
            for seq, img in val_loader:
                seq, img = seq.to(device), img.to(device)
                rec_seq, rec_img = model(seq, img)
                l_seq = criterion_seq(rec_seq, seq)
                l_img = criterion_img(rec_img, img)
                loss = 100.0 * l_seq + l_img
                val_loss += loss.item()
                val_loss_seq += l_seq.item()
                val_loss_img += l_img.item()
        val_loss /= len(val_loader)
        val_loss_seq /= len(val_loader)
        val_loss_img /= len(val_loader)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} (Seq: {val_loss_seq:.4f}, Img: {val_loss_img:.4f})")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if model_path:
                torch.save(model.state_dict(), model_path)
                
    return model

