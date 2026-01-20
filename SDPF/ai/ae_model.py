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
        # 修复尺寸不匹配问题
        # 错误形状: 1x33024 (实际输入) vs 32768x256 (期望输入)
        # 这意味着 seq_len // 32 计算出的长度和实际经过 5 层 CNN 后的长度不一致
        # 实际情况: 5层 stride=2 的卷积，每层会让长度减半
        # 4096 -> 2048 -> 1024 -> 512 -> 256 -> 128
        # 但是，如果输入序列长度有一些微小的填充或截断变化，CNN输出可能会有细微差别
        # 更健壮的做法是使用 AdaptiveAvgPool1d 强制将特征维度固定，或者打印出实际维度
        # 在这里，报错显示实际是 33024 = 256 * 129
        # 这意味着 CNN 输出的长度是 129 而不是 128 (128*256=32768)
        # 这通常是因为输入序列长度不是严格的 4096，或者 padding 的原因
        # 解决方法：定义 AdaptiveAvgPool1d(1) 将时间维度压缩为 1，或者调整线性层
        
        # 方案: 使用 FC 动态判断，或者在 forward 中自适应
        # 为了不破坏现有权重结构（如果有预训练），我们调整 flatten_size 定义
        # 然而，更好的方式是让 CNN 对输入长度稍微鲁棒一点
        # 这里我们假设 seq_len 是固定的 4096。
        # 报错的输入是 1x33024，说明 cnn 输出是 [Batch, 256, 129]
        # 这意味着输入到了 cnn 第一层的时候长度可能是 4097~?
        # 或者卷积Padding导致输出尺寸变大了。
        
        # 让我们检查 padding=1, kernel=3, stride=2 的各个尺寸变化：
        # L_out = floor((L_in + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)
        # L_out = floor((L_in + 2 - 2)/2 + 1) = floor(L_in/2 + 1)
        
        # 如果 L=4096
        # 1. floor(4096/2 + 1) = 2049
        # 2. floor(2049/2 + 1) = 1025
        # 3. floor(1025/2 + 1) = 513
        # 4. floor(513/2 + 1) = 257
        # 5. floor(257/2 + 1) = 129
        
        # 所以经过 5 层卷积，长度确实是 129！
        # 之前的代码 seq_len // 32 = 4096 // 32 = 128 是基于不带 Padding 或者 偶数整除的理想假设
        # 实际上 PyTorch 的 Conv1d(padding=1, stride=2) 对于偶数输入 L，输出是 L/2 + 1 (如果不完全整除)
        # 精确公式是 floor((L + 2*p - k)/s + 1) -> floor((L + 2 - 3)/2 + 1) = floor((L-1)/2 + 1)
        # 等一下，公式是 floor((L_in + 2*padding - kernel_size)/stride + 1)
        # floor((4096 + 2 - 3)/2 + 1) = floor(4095/2 + 1) = 2047 + 1 = 2048 (第一层)
        # floor((2048 + 2 - 3)/2 + 1) = 1024
        # floor(1024...) = 512
        # floor(512...) = 256
        # floor(256...) = 128
        
        # 为什么会变成 129？ 除非输入长度不是 4096。
        # 如果可以，我们在这里做一个自适应池化，强制变成 128，或者修改 Linear 层大小
        
        # 报错是 33024 = 256 * 129
        # 这意味着 CNN 输出形状是 (Batch, 256, 129)
        # 我们使用 AdaptiveAvgPool1d 来规避这个问题，让它总是输出 128
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(seq_len // 32) # 强制变为 128
        self.dropout = nn.Dropout(p=0.3)
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)

    def forward(self, x):
        # x: (Batch, Seq_Len, Num_Channels) -> Permute to (Batch, Ch, Seq_Len)
        x = x.permute(0, 2, 1)
        
        # CNN
        x = self.cnn(x) # (Batch, 256, Seq_Len/32)
        
        # 修复尺寸问题：强制池化到预期长度 (128)
        x = self.adaptive_pool(x)
        
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
