import torch.nn as nn
import torch.nn.functional as F



class CNN(nn.Module):
    def __init__(self, num_classes: int = 27, in_channels: int = 12, ch1: int = 32, ch2: int = 64, ch3: int = 128,
                 kernel_size: int = 3, dropout: float = 0.5, use_batchnorm: bool = True, use_fcnn: bool = False, 
                 window_size1D: int = 4000, **kwargs):
        """
        Combined CNN architecture for ECG classification with FCNN flag
        
        Args:
            num_classes: Number of output classes (default 27)
            in_channels: Number of ECG leads (default 12)
            ch1, ch2, ch3: Number of channels in conv layers
            use_batchnorm: Boolean flag to activate or deactivate batch norm
            use_fcnn: If True, uses FCNN architecture (sliding window + global pooling)
                     If False, uses standard CNN architecture (global pooling + FC layer)
            window_size: Kernel size for the sliding window classifier (only used if use_fcnn=True)
        """
        super().__init__()

        self.use_fcnn = use_fcnn
        padding = 0

        # Convolutional layers (shared by both architectures)
        self.conv1 = nn.Conv1d(in_channels, ch1, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(ch1, ch2, kernel_size, padding=padding)
        self.conv3 = nn.Conv1d(ch2, ch3, kernel_size, padding=padding)

        # Batch Norm layers (shared by both architectures)
        # nn.Identity() -> no operation
        self.bn1 = nn.BatchNorm1d(ch1) if use_batchnorm else nn.Identity()
        self.bn2 = nn.BatchNorm1d(ch2) if use_batchnorm else nn.Identity()
        self.bn3 = nn.BatchNorm1d(ch3) if use_batchnorm else nn.Identity()

        # Pooling layers (shared by both architectures)
        self.pool1 = nn.MaxPool1d(2)
        self.pool2 = nn.MaxPool1d(2)

        # Dropout layer (shared by both architectures)
        self.dropout = nn.Dropout(dropout)

        # Global Average Pooling (shared by both architectures)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        if use_fcnn:
            # FCNN: Sliding-window classifier
            self.classifier = nn.Conv1d(
                in_channels=ch3,
                out_channels=num_classes,
                kernel_size=window_size1D,
                stride=1,
                padding=0
            )
        else:
            # Standard CNN: MLP layer
            self.fc = nn.Linear(ch3, num_classes)

    def forward(self, x, **kwargs):
        """
        Args:
            x: (batch_size, in_channels, n_samples)
            
        Returns:
            (batch_size, num_classes)
        """
        # Layer 1
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01) # non-zero gradient 0.01 for negative inputs
        x = self.pool1(x)

        # Layer 2
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.01)
        x = self.pool2(x)

        # Layer 3
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.01)
        x = self.dropout(x)

        if self.use_fcnn:
            # FCNN: Sliding window classifier before Global pooling
            x = self.classifier(x)  #(batch_size, num_classes, T_windows)
            x = self.global_pool(x)  #(batch_size, num_classes, 1)
            x = x.squeeze(-1)  #(batch_size, num_classes)
        else:
            # Standard CNN: Global pooling before MLP layer
            x = self.global_pool(x)  #(batch_size, ch3, 1)
            x = x.squeeze(-1)  #(batch_size, ch3)
            x = self.fc(x)  #(batch_size, num_classes)

        return x
