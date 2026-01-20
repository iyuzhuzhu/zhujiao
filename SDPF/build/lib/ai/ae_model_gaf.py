import torch
import torch.nn as nn

class Cnn1dEncoder(nn.Module):
    def __init__(self, in_channels, latent_dim, seq_len):
        super(Cnn1dEncoder, self).__init__()
        # Assuming seq_len is 4096
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=3, stride=2, padding=1), # 2048
            nn.LeakyReLU(0.1),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1), # 1024
            nn.LeakyReLU(0.1),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1), # 512
            nn.LeakyReLU(0.1),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1), # 256
            nn.LeakyReLU(0.1),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1), # 128
            nn.LeakyReLU(0.1),
            nn.Flatten()
        )
        # Calculate flatten size dynamically or hardcode for 4096
        # 4096 -> 2048 -> 1024 -> 512 -> 256 -> 128
        # Output channels: 256
        # Flatten size: 256 * 128 = 32768
        self.flatten_size = 256 * (seq_len // 32) 
        self.fc = nn.Linear(self.flatten_size, latent_dim)

    def forward(self, x):
        x = self.net(x)
        x = self.fc(x)
        return x

class Cnn1dDecoder(nn.Module):
    def __init__(self, latent_dim, out_channels, seq_len):
        super(Cnn1dDecoder, self).__init__()
        self.seq_len = seq_len
        self.flatten_size = 256 * (seq_len // 32)
        self.fc = nn.Linear(latent_dim, self.flatten_size)
        
        self.net = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # 256
            nn.LeakyReLU(0.1),
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # 512
            nn.LeakyReLU(0.1),
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # 1024
            nn.LeakyReLU(0.1),
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # 2048
            nn.LeakyReLU(0.1),
            nn.ConvTranspose1d(16, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1), # 4096
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 256, self.seq_len // 32)
        x = self.net(x)
        return x

class CnnEncoder(nn.Module):
    def __init__(self, in_channels, latent_dim, image_size=64):
        super(CnnEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1), # 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 8x8
            nn.ReLU(),
            nn.Flatten()
        )
        # Calculate flatten size
        self.flatten_size = 128 * (image_size // 8) * (image_size // 8)
        self.fc = nn.Linear(self.flatten_size, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

class CnnDecoder(nn.Module):
    def __init__(self, latent_dim, out_channels, image_size=64):
        super(CnnDecoder, self).__init__()
        self.image_size = image_size
        self.flatten_size = 128 * (image_size // 8) * (image_size // 8)
        self.fc = nn.Linear(latent_dim, self.flatten_size)
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1), # 64x64
            # Removed Sigmoid to allow negative values (GAF is [-1, 1])
            # Or use Tanh if you want to enforce range
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 128, self.image_size // 8, self.image_size // 8)
        x = self.deconv(x)
        return x

class MultiModalAutoencoder(nn.Module):
    def __init__(self, seq_len, num_channels, image_size=64, lstm_hidden=256, cnn_latent=64, joint_dim=128):
        super(MultiModalAutoencoder, self).__init__()
        
        # Using 1D CNN instead of LSTM for sequence
        self.seq_encoder = Cnn1dEncoder(in_channels=num_channels, latent_dim=lstm_hidden, seq_len=seq_len)
        self.cnn_encoder = CnnEncoder(in_channels=num_channels, latent_dim=cnn_latent, image_size=image_size)
        
        # Fusion
        self.fusion_fc = nn.Linear(lstm_hidden + cnn_latent, joint_dim)
        
        # Splitting for decoding
        self.split_fc_seq = nn.Linear(joint_dim, lstm_hidden)
        self.split_fc_cnn = nn.Linear(joint_dim, cnn_latent)
        
        self.seq_decoder = Cnn1dDecoder(latent_dim=lstm_hidden, out_channels=num_channels, seq_len=seq_len)
        self.cnn_decoder = CnnDecoder(latent_dim=cnn_latent, out_channels=num_channels, image_size=image_size)

    def forward(self, x_seq, x_img):
        # x_seq: (Batch, Seq_Len, Num_Channels)
        # x_img: (Batch, Num_Channels, H, W)
        
        # Permute for Conv1d: (Batch, Num_Channels, Seq_Len)
        x_seq = x_seq.permute(0, 2, 1)
        
        h_seq = self.seq_encoder(x_seq)
        h_cnn = self.cnn_encoder(x_img)
        
        # Concatenate
        h_joint = torch.cat((h_seq, h_cnn), dim=1)
        # Use LeakyReLU to avoid dying ReLU
        h_joint = torch.nn.functional.leaky_relu(self.fusion_fc(h_joint), negative_slope=0.1)
        
        # Decode
        h_seq_dec = self.split_fc_seq(h_joint)
        h_cnn_dec = self.split_fc_cnn(h_joint)
        
        rec_seq = self.seq_decoder(h_seq_dec)
        rec_img = self.cnn_decoder(h_cnn_dec)
        
        # Permute back: (Batch, Seq_Len, Num_Channels)
        rec_seq = rec_seq.permute(0, 2, 1)
        
        return rec_seq, rec_img
