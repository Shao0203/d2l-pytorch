import pandas as pd
import os
import matplotlib.pyplot as plt
from torchvision.io import decode_image
from torchvision import datasets, transforms
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch
from torch import nn
import numpy as np


'''1. 张量

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
x_ones = torch.ones_like(x_data)
x_rand = torch.rand_like(x_data, dtype=torch.float)

# print(x_data, '\n', x_np, '\n', x_ones, '\n', x_rand)


shape = (2, 3)
rand_tensor = torch.randn(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
# print(rand_tensor, '\n', ones_tensor, '\n', zeros_tensor)
# print(rand_tensor.numel(), '\n', rand_tensor.shape, '\n', rand_tensor.dtype, '\n', rand_tensor.device)

# print(torch.backends.mps.is_available(), torch.accelerator.is_available(), torch.accelerator.current_accelerator())

tensor = torch.tensor(range(12)).reshape((3, 4))
ten1 = torch.ones(4, 4)
# print(tensor[0])
# print(tensor[:, 0])
# print(tensor[:, -1])
# print(tensor[..., -1])
# print(torch.sum(tensor, dim=0))
# print(tensor.sum(axis=1))
ten1[:, 1] = 0
# t1 = torch.cat([tensor, tensor, tensor], dim=1)
# print(tensor, '\n', tensor.T)
# y1 = ten1 @ ten1.T
# # print(y1)
# y2 = tensor.matmul(tensor.T)
# y2 = torch.matmul(ten1, ten1.T)
# # print(y2)

# z1 = ten1 * ten1
# print(z1)
# z2 = ten1.mul(ten1)
# print(z2)
# z3 = torch.rand_like(ten1)
# torch.mul(ten1, ten1, out=z3)
# print(z3)
# agg = ten1.sum()
# agg_item = agg.item()
# print(agg, type(agg), agg_item, type(agg_item))

# print(ten1)
# ten2 = ten1 + 5
# print(ten2)
# ten3 = ten1.add(5)
# print(ten3)
# ten1.add_(5)
# print(torch.allclose(ten1, ten2) and torch.allclose(ten2, ten3))

# t = torch.ones(5)
# print(f"t: {t}", type(t))
# n = t.numpy()
# print(f"n: {n}", type(n))
# t.add_(1)
# print(f"t: {t}")
# print(f"n: {n}")

n = np.ones(5)
t = torch.from_numpy(n)
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
'''


'''2. 数据集与数据加载器

training_data = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

# labels_map = {
#     0: "T-Shirt",
#     1: "Trouser",
#     2: "Pullover",
#     3: "Dress",
#     4: "Coat",
#     5: "Sandal",
#     6: "Shirt",
#     7: "Sneaker",
#     8: "Bag",
#     9: "Ankle Boot",
# }

# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(training_data), size=(1,)).item()
#     img, label = training_data[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.title(labels_map[label])
#     plt.axis('off')
#     plt.imshow(img.squeeze(), cmap='gray')
# plt.show()


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file='data/labels.csv', img_dir='data/fashion_images', transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)  # read the image and transfer it into tensor
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# dataset = CustomImageDataset()
# dataset[0]  # trigger __getitem__(idx=0)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
# train_features, train_labels = next(iter(train_dataloader))
# print(f'Feature batch shape: {train_features.shape}')
# print(f'Labels batch shape: {train_labels.shape}')
# img = train_features[0]
# label = train_labels[0]
# print(f'Label: {label}')
# plt.imshow(img.squeeze(), cmap='gray')
# plt.show()

# X, y = next(iter(train_dataloader))
# print(X.shape, y.shape)
'''


'''3. 变换 (Transforms)

ds = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float, scale=True)]),
    target_transform=v2.Lambda(lambda y: F.one_hot(torch.tensor(y), num_classes=10).float())
)
'''


'''4. 构建神经网络

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=28*28, out_features=512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
X = torch.rand(1, 28, 28, device=device)
logits = model(X)  # scores
# pred_probab = nn.Softmax(dim=1)(logits)
softmax = nn.Softmax(1)
pred_probab = softmax(logits)
y_pred = pred_probab.argmax(1)

input_image = torch.rand(3, 28, 28)
flatten = nn.Flatten()
flat_image = flatten(input_image)
# print(flat_image.shape)  # (3, 784)

layer1 = nn.Linear(in_features=784, out_features=20)
hidden1 = layer1(flat_image)
# print(layer1.weight.shape)  # (20, 784)
# print(hidden1.shape)  # (3, 20)
# print(f'Before ReLU: {hidden1}\n\n')
hidden1 = nn.ReLU()(hidden1)
# print(f'After ReLU: {hidden1}')

seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10),
    softmax,
)
pred_prob = seq_modules(input_image)
# print(pred_prob.shape)  # (3, 10)

# print(f"Model structure: {model}\n\n")
# for name, param in model.named_parameters():
#     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
# model.parameters()

'''


'''5. 自动微分 (Autograd)

x = torch.ones(5)
y = torch.zeros(3)
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
loss.backward()  # calc gradients and save to w.grad and b.grad
# print(w.grad, '\n', b.grad)
# print(z.requires_grad)
# print(z.grad)
# print(loss)
# print(f"Gradient function for z = {z.grad_fn}")
# print(f"Gradient function for loss = {loss.grad_fn}")
# print(z.grad)
# print(x.requires_grad)  # False
# print(y.requires_grad)  # False
# print(z.requires_grad)  # True
# print(w.requires_grad)  # True
# print(b.requires_grad)  # True
# with torch.no_grad():
#     z = torch.matmul(x, w) + b
# print(z.requires_grad)  # False

# z = torch.matmul(x, w) + b
# print(z.requires_grad)  # True
# z_det = z.detach()
# print(z_det.requires_grad)  # False
# loss.backward()
# print(w.grad, '\n', b.grad)
# # print(z.requires_grad, z_det.requires_grad)


# inp = torch.eye(4, 5, requires_grad=True)
# out = (inp+1).pow(2).t()
# out.backward(torch.ones_like(out), retain_graph=True)
# print(f"First call\n{inp.grad}")
# out.backward(torch.ones_like(out), retain_graph=True)
# print(f"\nSecond call\n{inp.grad}")
# inp.grad.zero_()
# out.backward(torch.ones_like(out), retain_graph=True)
# print(f"\nCall after zeroing gradients\n{inp.grad}")
'''


'''6. 优化模型参数
'''
training_data = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float, scale=True)]),
)

test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float, scale=True)]),
)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f'loss: {loss:>7f} [{current:>5d}/{size:>5d}]')


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f'Test Error: \n Accuracy: {correct:.1%}, Avg loss: {test_loss:>8f} \n')


train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)
model = NeuralNetwork()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters())

for epoch in range(5):
    print(f'Epoch {epoch+1}\n-------------------------------')
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print('Done!')
