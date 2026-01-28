import torch.nn as nn

class KeywordCNNv2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        def conv_block(in_ch, out_ch, kernel=3, stride=1, pool=True):
            layers = [
                nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=kernel//2, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ]
            if pool:
                layers.append(nn.MaxPool2d(2, 2))
            return nn.Sequential(*layers)

        # Feature extractor
        self.layer1 = conv_block(1, 32)
        self.layer2 = conv_block(32, 64)
        self.layer3 = conv_block(64, 128)

        # Depthwise separable conv (for temporal modeling)
        self.dw_conv = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3,1), padding=(1,0), groups=128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Squeeze-and-Excitation (light attention)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 128, 1),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Depthwise temporal conv
        x = self.dw_conv(x)

        # Squeeze-and-Excitation
        se = self.se(x)
        x = x * se

        # Global average pooling
        x = x.mean(dim=[2, 3])
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
class KeywordCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        
        # Global average pooling
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: [batch, 1, n_mels, time]
        x = self.pool(nn.functional.relu(self.bn1(self.conv1(x))))
        x = self.pool(nn.functional.relu(self.bn2(self.conv2(x))))
        x = self.pool(nn.functional.relu(self.bn3(self.conv3(x))))
        
        # Global average pooling
        x = x.mean(dim=[2, 3])  # [batch, channels]
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
class KeywordDSCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.dw1 = nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32)
        self.pw1 = nn.Conv2d(32, 64, kernel_size=1)

        self.dw2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64)
        self.pw2 = nn.Conv2d(64, 64, kernel_size=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = nn.functional.relu(self.pw1(nn.functional.relu(self.dw1(x))))
        x = self.pool(x)

        x = nn.functional.relu(self.pw2(nn.functional.relu(self.dw2(x))))
        x = self.pool(x)

        x = x.mean(dim=[2,3])
        x = self.fc(x)
        return x

class KeywordMLP(nn.Module):
    def __init__(self, num_classes: int, hidden: int = 24):
        super().__init__()
        self.fc1 = nn.Linear(64 * 32, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        # Accept: [B,1,64,32] or [B,64,32]
        if x.ndim == 4:
            x = x.squeeze(1)  # [B,64,32]
        x = x.reshape(x.shape[0], -1)  # [B,2048]

        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)  # logits
        return x