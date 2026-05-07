import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio



class CNN_TimeFreq(nn.Module):
    def __init__(self, num_classes=27, in_channels=12, n_fft=128, hop_length=64,
            win_length=128, ch1=32, ch2=64, ch3=128, dropout=0.5, use_batchnorm=True, use_fcnn=False,
            window_size2D=(4, 4), **kwargs): 
        """
        Integrated a time-frequency transformation (STFT spectrogram) using torchaudiow with a 2D convolutional network.
        
        :param num_classes: Number of output classes (default 27)
        :param in_channels: Number of ECG leads (default 12)
        :param n_fft: Number of points used in the Fourier Transform (frequency resolution)
        :param hop_length: Window sliding stride (how much the window slide each step)
        :param win_length: Window size in samples
        :param window_size: Kernel size for the 2D sliding window classifier (freq, time) 
                            Only used if use_fcnn=True
        """
        super().__init__()

        # Spectrogram module
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            power=2.0 # returns power spectrogram
        ) # (batch_size, n_leads, freq, time)

        padding = 0

        # 2D Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, ch1, kernel_size=3, padding=padding)
        self.conv2 = nn.Conv2d(ch1, ch2, kernel_size=3, padding=padding)
        self.conv3 = nn.Conv2d(ch2, ch3, kernel_size=3, padding=padding)

        # Batch Norm layers
        # nn.Identity() -> no operation
        self.bn1 = nn.BatchNorm2d(ch1) if use_batchnorm else nn.Identity()
        self.bn2 = nn.BatchNorm2d(ch2) if use_batchnorm else nn.Identity()
        self.bn3 = nn.BatchNorm2d(ch3) if use_batchnorm else nn.Identity()

        # 2D Max Pooling layers
        self.pool = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)

        self.dropout = nn.Dropout(dropout)

        self.use_fcnn = use_fcnn
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        if use_fcnn:
            self.classifier = nn.Conv2d(
                in_channels=ch3,
                out_channels=num_classes,
                kernel_size=window_size2D,
                stride=1,
                padding=0
            )
        else:
            self.fc = nn.Linear(ch3, num_classes)

    def forward(self, x, **kwargs):
        """
        Args:
            x: (batch_size, n_leads, n_samples)
            
        Returns:
            (batch_size, num_classes)
        """

        # Apply spectrogram lead wise
        batch_size, n_leads, n_samples = x.shape
        x = x.reshape(batch_size * n_leads, n_samples) # flatten
        x = self.spectrogram(x) # (batch*n_leads, freq, time)  [0, 69055] 

        x = torch.clamp(x, min=1e-10)       # prevent log(0) = -inf

        x = torch.log(x)                     # compress to [-23, 11] 
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
        x = (x - mean) / std                 # normalize to mean=0, std=1

        x = x.reshape(batch_size, n_leads, x.shape[-2], x.shape[-1]) # (batch_size, n_leads, freq, time)

        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.01) # non-zero gradient 0.01 for negative inputs
        x = self.pool(x)

        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.01)
        x = self.pool(x)

        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.01)

        x = self.dropout(x)

        if self.use_fcnn:
            # FCNN: Sliding window classifier before Global pooling
            x = self.classifier(x) # (batch_size, num_classes, freq_windows, time_windows)
            x = self.global_pool(x) # (batch_size, num_classes, 1, 1)
            x = x.view(x.size(0), -1) # (batch_size, num_classes)
        else:
            # Standard CNN: Global pooling before MLP layer
            x = self.global_pool(x) # (batch_size, ch3, 1, 1)
            x = x.view(x.size(0), -1) # (batch_size, ch3)
            x = self.fc(x) # (batch_size, num_classes)

        return x
