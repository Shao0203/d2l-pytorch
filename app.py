import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def load_data():
    ''' Download training / test data (60000, 10000) from open datasets. '''
    training_data = datasets.FashionMNIST(
        root='data',
        train=True,
        download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    )

    test_data = datasets.FashionMNIST(
        root='data',
        train=False,
        download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    )

    return training_data, test_data


def create_dataloader():
    batch_size = 64
    training_data, test_data = load_data()
    train_dataloader = DataLoader(training_data, batch_size=batch_size)  # num_batch=len(train_dataloader)=938
    test_dataloader = DataLoader(test_data, batch_size=batch_size)  # num_batch=len(test_dataloader)=157

    # for X, y in test_dataloader:
    #     print(f'Shape of X [N, C, H, W]: {X.shape}')  # torch.Size([64, 1, 28, 28])
    #     print(f'Shape of y: {y.shape} | data type: {y.dtype}')  # torch.Size([64]) | torch.int64

    return train_dataloader, test_dataloader


def build_model(device='mps'):
    model = NeuralNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    # print(model)
    return model, loss_fn, optimizer


def train(dataloader, model, loss_fn, optimizer, device='mps'):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f'loss: {loss:>7f} [{current:>5d}/{size:>5d}]')


def test(dataloader, model, loss_fn, device='mps'):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f'Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n')


def fit():
    # device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
    train_dataloader, test_dataloader = create_dataloader()
    model, loss_fn, optimizer = build_model(device)
    epochs = 5

    for t in range(epochs):
        print(f'Epoch {t+1}\n-------------------------------')
        train(train_dataloader, model, loss_fn, optimizer, device)
        test(test_dataloader, model, loss_fn, device)
    print('Done!')

    torch.save(model.state_dict(), 'model.pth')
    print('Saved PyTorch Model State to model.pth')


def load_model_pred(device='mps'):
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load('model.pth', weights_only=True))
    training_data, test_data = load_data()

    classes = [
        'T-shirt/top',
        'Trouser',
        'Pullover',
        'Dress',
        'Coat',
        'Sandal',
        'Shirt',
        'Sneaker',
        'Bag',
        'Ankle boot',
    ]
    model.eval()
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: <{predicted}>, Actual: <{actual}>')


# fit()
load_model_pred()
