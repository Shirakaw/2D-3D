import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# 数据加载和预处理
class DislocationDataset(Dataset):
    def __init__(self, image_paths, targets, transform=None):
        self.image_paths = image_paths
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        target = self.targets[idx]

        if self.transform:
            img = self.transform(img)

        return img, target

# 创建自定义数据集（请根据你的实际情况修改 image_paths 和 targets）
train_image_paths = [...]  # 训练集图片路径列表
train_targets = [...]  # 训练集目标列表
test_image_paths = [...]  # 测试集图片路径列表
test_targets = [...]  # 测试集目标列表

# 定义图像预处理操作
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 创建数据集和数据加载器
train_dataset = DislocationDataset(train_image_paths, train_targets, transform=transform)
test_dataset = DislocationDataset(test_image_paths, test_targets, transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 设置模型参数
cnn_output_size = 512  # 根据实际情况调整，这里假设ResNet输出的特征大小为512
hidden_size = 128
num_layers = 2
num_classes = ...  # 根据实际情况设置类别数

import torch
import torch.nn as nn
from torchvision.models import resnet18

class FeatureExtractorResNet(nn.Module):
    def __init__(self):
        super(FeatureExtractorResNet, self).__init__()
        resnet = resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        return x

class DislocationPredictionModelResNetLSTM(nn.Module):
    def __init__(self, cnn_output_size, hidden_size, num_layers, num_classes):
        super(DislocationPredictionModelResNetLSTM, self).__init__()
        self.cnn = FeatureExtractorResNet()
        self.lstm = nn.LSTM(cnn_output_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, h):
        batch_size = x.size(0)

        # 使用ResNet提取特征
        cnn_features = self.cnn(x)

        # LSTM序列建模
        lstm_out, h
        lstm_out, h = self.lstm(cnn_features.view(batch_size, 1, -1), h)

        # 全连接层
        out = self.fc(lstm_out.contiguous().view(batch_size, -1))

        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.lstm.num_layers, batch_size, self.lstm.hidden_size).zero_(),
                  weight.new(self.lstm.num_layers, batch_size, self.lstm.hidden_size).zero_())
        return hidden

# 训练模型的函数和测试模型的函数与之前的实现相同，这里不再重复。

# 创建模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DislocationPredictionModelResNetLSTM(cnn_output_size, hidden_size, num_layers, num_classes).to(device)
criterion = nn.MSELoss()  # 可根据实际情况更换损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练和评估模型
num_epochs = 20
for epoch in range(num_epochs):
    train_loss = train_model(model, train_dataloader, criterion, optimizer, device)
    test_loss = test_model(model, test_dataloader, criterion, device)
    print(f"Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

# 保存模型
torch.save(model.state_dict(), "dislocation_prediction_model.pth")
