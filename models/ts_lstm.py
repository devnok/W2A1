import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

device = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)


class LSTMModel(nn.Module):
    def __init__(
        self, num_classes: int, input_size: int, hidden_size: int, num_layers: int, seq_length: int
    ):
        super(LSTMModel, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc_1 = nn.Linear(hidden_size, 128)  # fully connected 1
        self.fc = nn.Linear(128, num_classes)  # fully connected last layer

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device))
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device))

        out, (hn, cn) = self.lstm(x, (h0, c0))

        hn = hn.view(-1, self.hidden_size)  # reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out)  # first Dense
        out = self.relu(out)  # relu
        out = self.fc(out)  # Final Output

        return out


class ForeCastLSTM:
    def split_dataset(self, X, y, split_ratio: float = 0.8):
        dataset_size = len(X)
        train_size = int(dataset_size * split_ratio)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        return X_train, y_train, X_test, y_test

    def reshape_dataset(self, X_train, y_train, X_test, y_test):
        X_train_tensors = Variable(torch.Tensor(X_train))
        X_test_tensors = Variable(torch.Tensor(X_test))

        y_train_tensors = Variable(torch.Tensor(y_train))
        y_test_tensors = Variable(torch.Tensor(y_test))

        X_train_tensors_f = torch.reshape(
            X_train_tensors, (X_train_tensors.shape[0], 1, X_train_tensors.shape[1])
        )
        X_test_tensors_f = torch.reshape(
            X_test_tensors, (X_test_tensors.shape[0], 1, X_test_tensors.shape[1])
        )

        return X_train_tensors_f, y_train_tensors, X_test_tensors_f, y_test_tensors

    def fit_model(
        self,
        X,
        y,
        learning_rate: float = 0.01,
        num_epochs: int = 200,
        batch_size: int = 64,
        num_classes: int = 1,
        input_size: int = 1,
        hidden_size: int = 128,
        seq_length: int = 1,
        num_layers: int = 1,
    ):
        X_train, y_train, X_test, y_test = self.split_dataset(X, y)

        (
            X_train_tensors_f,
            y_train_tensors,
            X_test_tensors_f,
            y_test_tensors,
        ) = self.reshape_dataset(X_train, y_train, X_test, y_test)

        model = LSTMModel(
            num_classes, input_size, hidden_size, num_layers, X_train_tensors_f.shape[1]
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            outputs = model.forward(X_train_tensors_f.to(device))
            optimizer.zero_grad()

            # obtain the loss function
            loss = criterion(outputs, y_train_tensors.to(device))

            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
