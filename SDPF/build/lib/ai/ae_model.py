import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CnnEncoder(nn.Module):
    def __init__(self, seq_len, num_channels, latent_dim=256, hidden_dim=None):
        super(CnnEncoder, self).__init__()
        self.seq_len = seq_len
        self.num_channels = num_channels
        self.latent_dim = latent_dim

        # CNN Layers to reduce sequence length
        # Input: (Batch, Ch, Seq)
        self.cnn = nn.Sequential(
            nn.Conv1d(num_channels, 16, kernel_size=3, stride=2, padding=1), # L/2
            nn.LeakyReLU(0.1),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1), # L/4
            nn.LeakyReLU(0.1),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1), # L/8
            nn.LeakyReLU(0.1),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1), # L/16
            nn.LeakyReLU(0.1),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1), # L/32
            nn.LeakyReLU(0.1),
        )
        
        self.flatten_size = 256 * (seq_len // 32)
        self.dropout = nn.Dropout(p=0.3)
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)

    def forward(self, x):
        # x: (Batch, Seq_Len, Num_Channels) -> Permute to (Batch, Ch, Seq_Len)
        x = x.permute(0, 2, 1)
        
        # CNN
        x = self.cnn(x) # (Batch, 256, Seq_Len/32)
        x = self.dropout(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC
        latent = self.fc_mu(x)
        return latent

class CnnDecoder(nn.Module):
    def __init__(self, seq_len, num_channels, latent_dim=256, hidden_dim=None):
        super(CnnDecoder, self).__init__()
        self.seq_len = seq_len
        self.num_channels = num_channels
        self.latent_dim = latent_dim
        self.reduced_len = seq_len // 32
        self.flatten_size = 256 * self.reduced_len

        self.fc_expand = nn.Linear(latent_dim, self.flatten_size)
        self.dropout = nn.Dropout(p=0.3)
        
        self.cnn_transpose = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # L/16
            nn.LeakyReLU(0.1),
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # L/8
            nn.LeakyReLU(0.1),
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # L/4
            nn.LeakyReLU(0.1),
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # L/2
            nn.LeakyReLU(0.1),
            nn.ConvTranspose1d(16, num_channels, kernel_size=3, stride=2, padding=1, output_padding=1), # L
        )

    def forward(self, x):
        # x: (Batch, Latent_Dim)
        hidden = self.fc_expand(x) # (Batch, Flatten_Size)
        hidden = self.dropout(hidden)
        
        # Reshape: (Batch, 256, Reduced_Len)
        hidden = hidden.view(-1, 256, self.reduced_len)
        
        # CNN Transpose
        out = self.cnn_transpose(hidden) # (Batch, Num_Channels, Seq_Len)
        
        # Permute back: (Batch, Seq_Len, Num_Channels)
        reconstruction = out.permute(0, 2, 1)
        return reconstruction

class CnnAutoencoder(nn.Module):
    def __init__(self, seq_len=4096, num_channels=3, latent_dim=256, hidden_dim=512, device=DEVICE):
        super(CnnAutoencoder, self).__init__()
        self.encoder = CnnEncoder(seq_len, num_channels, latent_dim, hidden_dim).to(device)
        self.decoder = CnnDecoder(seq_len, num_channels, latent_dim, hidden_dim).to(device)

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction

# Alias for backward compatibility
LstmAutoencoder = CnnAutoencoder

# Alias for backward compatibility if needed, or just replace usage
# LstmAutoencoder = CnnLstmAutoencoder
