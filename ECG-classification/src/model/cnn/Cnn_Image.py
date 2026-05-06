import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Image(nn.Module):
    def __init__(self, num_classes=27, input_h=512, input_w=512, mode='square'):
        super(CNN_Image, self).__init__()
        
        self.mode = mode
        self.input_h = input_h
        
        if mode == 'square':
            kernel = (3, 3)
            stride = (1, 1)
            padding = (1, 1)
        elif mode == 'rectangle':
            kernel = (input_h // 2, 3) 
            stride = (2, 1)
            padding = (0, 1)
        else:
            raise ValueError("Mode must be 'square' or 'rectangle'")

        self.conv1 = nn.Conv2d(in_channels=12, out_channels=32, 
                               kernel_size=kernel, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
    
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, batch_mask=None):
        # x shape: [B, S, C, H, W]
        b, s, c, h, w = x.shape
        x = x.reshape(-1, c, h, w)

        if x.dtype == torch.uint8:
            x = x.to(device=x.device, dtype=torch.float32).div_(255.0)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x) # [B*S, 27]
        logits = logits.view(b, s, -1) # [B, S, 27]

        logits = logits.mean(1)
            
        return logits # [B, 27]