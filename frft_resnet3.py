import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import os

# 데이터 디렉토리 설정
data_dir = '/Users/m1_4k/그림/hymenoptera_data'

# 데이터 전처리
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                           [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                           [0.229, 0.224, 0.225])
    ])
}

# ImageFolder로 로드
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

dataloaders = {x: DataLoader(image_datasets[x], batch_size=16,
                            shuffle=True, num_workers=0)
               for x in ['train', 'val']}

print(f"Train samples: {len(image_datasets['train'])}")
print(f"Val samples: {len(image_datasets['val'])}")
print(f"Classes: {image_datasets['train'].classes}")

class FrFTLoss(nn.Module):
    def __init__(self, alpha_orders=[0.5, 1.0, 1.5], lambda_weight=0.5):
        super(FrFTLoss, self).__init__()
        self.alpha_orders = alpha_orders
        self.lambda_weight = lambda_weight
        self.ce_loss = nn.CrossEntropyLoss()
    
    def frft_1d(self, x, alpha):
        """1D 분수 푸리에 변환 (개선된 버전)"""
        N = len(x)
        
        if alpha == 0:
            return x
        if alpha == 1:
            return np.fft.fft(x)
        
        # Fractional FT 계산
        result = np.zeros(N, dtype=complex)
        phi = np.pi * alpha / 2
        
        for k in range(N):
            sum_val = 0.0
            for m in range(N):
                # Kernel 계산
                kernel_real = np.cos(phi * (m**2 + k**2) / (N**2))
                kernel_imag = -np.sin(phi * (m**2 + k**2) / (N**2))
                kernel_phase = 2 * np.pi * alpha * m * k / (N**2)
                
                kernel_val = (kernel_real - 1j * kernel_imag) * np.exp(-1j * kernel_phase)
                sum_val += x[m] * kernel_val
            
            result[k] = sum_val / N
        
        return result
    
    def apply_frft_to_batch(self, logits, alpha):
        """배치의 각 샘플에 FrFT 적용"""
        batch_size = logits.shape[0]
        num_classes = logits.shape[1]
        device = logits.device
        
        frft_logits = []
        for i in range(batch_size):
            logit_np = logits[i].detach().cpu().numpy()
            frft_result = self.frft_1d(logit_np, alpha)
            frft_magnitude = np.abs(frft_result)
            frft_logits.append(frft_magnitude)
        
        frft_logits = np.array(frft_logits)
        frft_logits = torch.tensor(frft_logits, dtype=torch.float32, device=device)
        
        return frft_logits
    
    def forward(self, logits, targets):
        """
        logits: (batch_size, num_classes)
        targets: (batch_size,)
        """
        # 원본 도메인 손실
        ce_loss = self.ce_loss(logits, targets)
        
        # FrFT 도메인 손실들
        frft_losses = []
        for alpha in self.alpha_orders:
            frft_logits = self.apply_frft_to_batch(logits, alpha)
            
            # FrFT 공간에서도 크로스엔트로피 계산
            frft_ce = self.ce_loss(frft_logits, targets)
            frft_losses.append(frft_ce)
        
        # 평균 FrFT 손실
        avg_frft_loss = torch.mean(torch.stack(frft_losses))
        
        # 최종 손실 = CE Loss + λ * FrFT Loss
        total_loss = ce_loss + self.lambda_weight * avg_frft_loss
        
        return total_loss
        

# 디바이스 설정
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# ResNet-18 모델 로드
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 2개 클래스 (ants, bees)
model = model.to(device)

# FrFT Loss 함수 및 옵티마이저
frft_loss = FrFTLoss(alpha_orders=[0.5, 1.0, 1.5], lambda_weight=0.5).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# 학습 기록
history = {
    'train_acc': [],
    'val_acc': []
}

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    running_corrects = 0
    total_samples = 0
    
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = frft_loss(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)
    
    epoch_acc = float(running_corrects) / total_samples
    
    return epoch_acc

def val_epoch(model, dataloader, device):
    model.eval()
    running_corrects = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
    
    epoch_acc = float(running_corrects) / total_samples
    
    return epoch_acc

# 학습 시작
num_epochs = 30

print("Starting training...")
for epoch in range(num_epochs):
    train_acc = train_epoch(model, dataloaders['train'], optimizer, device)
    val_acc = val_epoch(model, dataloaders['val'], device)
    
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    
    scheduler.step()
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Acc: {train_acc:.4f}")
    print(f"Val Acc: {val_acc:.4f}")
    print("-" * 50)

print("Training completed!")

print("\n" + "="*60)
print("Starting Standard CrossEntropyLoss Training...")
print("="*60 + "\n")

# 새로운 ResNet-18 모델 (표준 Loss용)
model_standard = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model_standard.fc.in_features
model_standard.fc = nn.Linear(num_ftrs, 2)
model_standard = model_standard.to(device)

# 표준 Loss 및 옵티마이저
standard_loss = nn.CrossEntropyLoss()
optimizer_standard = optim.SGD(model_standard.parameters(), lr=0.001, momentum=0.9)
scheduler_standard = optim.lr_scheduler.StepLR(optimizer_standard, step_size=7, gamma=0.1)

# 표준 Loss 학습 기록
history_standard = {
    'train_acc': [],
    'val_acc': []
}

def train_epoch_standard(model, dataloader, optimizer, device):
    model.train()
    running_corrects = 0
    total_samples = 0
    
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = standard_loss(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)
    
    epoch_acc = float(running_corrects) / total_samples
    
    return epoch_acc

def val_epoch_standard(model, dataloader, device):
    model.eval()
    running_corrects = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
    
    epoch_acc = float(running_corrects) / total_samples
    
    return epoch_acc

# 표준 Loss 학습
for epoch in range(num_epochs):
    train_acc = train_epoch_standard(model_standard, dataloaders['train'], 
                                     optimizer_standard, device)
    val_acc = val_epoch_standard(model_standard, dataloaders['val'], device)
    
    history_standard['train_acc'].append(train_acc)
    history_standard['val_acc'].append(val_acc)
    
    scheduler_standard.step()
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Acc: {train_acc:.4f}")
    print(f"Val Acc: {val_acc:.4f}")
    print("-" * 50)

print("Standard Training completed!")


# Accuracy 그래프만 출력
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Train Accuracy 비교
axes[0].plot(history['train_acc'], label='FrFT Train Acc', marker='o', linewidth=2)
axes[0].plot(history_standard['train_acc'], label='Standard Train Acc', 
             marker='^', linewidth=2, linestyle='--')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Training Accuracy Comparison', fontsize=13)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Val Accuracy 비교
axes[1].plot(history['val_acc'], label='FrFT Val Acc', marker='s', linewidth=2)
axes[1].plot(history_standard['val_acc'], label='Standard Val Acc', 
             marker='d', linewidth=2, linestyle='--')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title('Validation Accuracy Comparison', fontsize=13)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('frft_vs_standard_accuracy.png', dpi=300, bbox_inches='tight')
print("Accuracy graph saved as 'frft_vs_standard_accuracy.png'")
plt.show()

# 성능 요약 출력
print("\n" + "="*60)
print("COMPARISON RESULTS")
print("="*60)
print(f"\nFrFT Loss:")
print(f"  Final Train Acc: {history['train_acc'][-1]:.4f}")
print(f"  Final Val Acc: {history['val_acc'][-1]:.4f}")
print(f"  Best Val Acc: {max(history['val_acc']):.4f}")

print(f"\nStandard Loss:")
print(f"  Final Train Acc: {history_standard['train_acc'][-1]:.4f}")
print(f"  Final Val Acc: {history_standard['val_acc'][-1]:.4f}")
print(f"  Best Val Acc: {max(history_standard['val_acc']):.4f}")

print(f"\nImprovement (FrFT vs Standard):")
acc_improvement = (max(history['val_acc']) - max(history_standard['val_acc'])) * 100
print(f"  Val Accuracy Improvement: {acc_improvement:.2f}%")
print("="*60)