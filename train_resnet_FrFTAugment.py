import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eig
import time
import random


def frft_2d(image, alpha_x, alpha_y):
    """2D Fractional Fourier Transform"""
    H, W = image.shape
    
    # X 방향 FrFT
    result = torch.zeros_like(image)
    for i in range(H):
        result[i, :] = frft_1d(image[i, :], alpha_x)
    
    # Y 방향 FrFT
    temp = torch.zeros_like(result)
    for j in range(W):
        temp[:, j] = frft_1d(result[:, j], alpha_y)
    
    return temp


def frft_1d(x, alpha):
    """1D Fractional Fourier Transform using eigendecomposition"""
    N = len(x)
    
    # DFT 행렬 생성
    n = torch.arange(N, dtype=torch.float32)
    k = n.reshape((N, 1))
    M = torch.exp(-2j * np.pi * k * n / N)
    
    # NumPy로 변환하여 고유값 분해
    M_np = M.numpy()
    eigenvalues, eigenvectors = eig(M_np)
    
    # Fractional power 적용
    eigenvalues_alpha = np.power(eigenvalues, alpha)
    
    # FrFT 행렬 재구성
    M_alpha = eigenvectors @ np.diag(eigenvalues_alpha) @ np.linalg.inv(eigenvectors)
    
    # 입력 신호에 적용
    x_np = x.numpy()
    result = M_alpha @ x_np
    
    return torch.from_numpy(result.real.astype(np.float32))


class FrFTAugmentation:
    """FrFT 기반 데이터 오그멘테이션"""
    def __init__(self, alpha_range=(0.3, 0.7), prob=0.5):
        """
        Args:
            alpha_range: FrFT alpha 파라미터 범위 (min, max)
            prob: 오그멘테이션 적용 확률
        """
        self.alpha_range = alpha_range
        self.prob = prob
    
    def __call__(self, img):
        """
        Args:
            img: PIL Image or Tensor (C, H, W)
        Returns:
            Transformed tensor
        """
        if random.random() > self.prob:
            return img
        
        # Tensor로 변환 (아직 Tensor가 아닌 경우)
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)
        
        # 랜덤 alpha 값 선택
        alpha_x = random.uniform(*self.alpha_range)
        alpha_y = random.uniform(*self.alpha_range)
        
        C, H, W = img.shape
        result = torch.zeros_like(img)
        
        # 각 채널에 FrFT 적용
        for c in range(C):
            result[c] = frft_2d(img[c], alpha_x, alpha_y)
        
        return result


class ResNetClassifier(nn.Module):
    """일반적인 ResNet18 분류기 (FrFT는 데이터 오그멘테이션으로만 사용)"""
    def __init__(self, num_classes=10):
        super(ResNetClassifier, self).__init__()
        
        # ResNet18 백본 사용
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()
        self.resnet.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        return self.resnet(x)


def train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs=30):
    """모델 학습"""
    train_acc_history = []
    test_acc_history = []
    
    for epoch in range(epochs):
        # 학습 모드
        model.train()
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}/{epochs}, Batch: {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')
        
        train_acc = 100. * train_correct / train_total
        train_acc_history.append(train_acc)
        
        # 테스트 모드
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()
        
        test_acc = 100. * test_correct / test_total
        test_acc_history.append(test_acc)
        
        print(f'Epoch {epoch+1}/{epochs}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
    
    return train_acc_history, test_acc_history


def main():
    # 하이퍼파라미터
    batch_size = 4
    learning_rate = 0.1
    epochs = 20
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 데이터 전처리 - FrFT 오그멘테이션 포함
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        FrFTAugmentation(alpha_range=(0.3, 0.7), prob=0.5),  # FrFT 오그멘테이션 추가
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # CIFAR-10 데이터셋
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # 모델 초기화
    model = ResNetClassifier(num_classes=10).to(device)
    print(f'Model initialized: ResNet18 with FrFT Data Augmentation')
    
    # 손실 함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    
    # 학습률 스케줄러
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 학습 시작
    print('\n' + '='*60)
    print('Training Configuration:')
    print(f'  - Model: ResNet18')
    print(f'  - Data Augmentation: FrFT (alpha range: 0.3-0.7, prob: 0.5)')
    print(f'  - Loss Function: CrossEntropyLoss (standard)')
    print(f'  - Epochs: {epochs}')
    print(f'  - Batch Size: {batch_size}')
    print(f'  - Learning Rate: {learning_rate}')
    print('='*60 + '\n')
    print('Starting training...')
    start_time = time.time()
    
    train_acc_history, test_acc_history = train_model(
        model, train_loader, test_loader, criterion, optimizer, device, epochs
    )
    
    end_time = time.time()
    print(f'\nTotal training time: {(end_time - start_time)/60:.2f} minutes')
    
    # Accuracy 그래프만 그리기
    plt.figure(figsize=(10, 6))
    epochs_range = range(1, epochs + 1)
    plt.plot(epochs_range, train_acc_history, 'b-', label='Train Accuracy', linewidth=2)
    plt.plot(epochs_range, test_acc_history, 'r-', label='Test Accuracy', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Training and Test Accuracy', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('accuracy_plot.png', dpi=300, bbox_inches='tight')
    print('\nAccuracy plot saved as accuracy_plot.png')
    
    # 최종 결과 출력
    print(f'\nFinal Train Accuracy: {train_acc_history[-1]:.2f}%')
    print(f'Final Test Accuracy: {test_acc_history[-1]:.2f}%')
    print(f'Best Test Accuracy: {max(test_acc_history):.2f}%')


if __name__ == '__main__':
    main()