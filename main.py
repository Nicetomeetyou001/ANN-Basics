import torch
import torch.nn as nn
import torch.optim as optim


class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, mode='regression'):
        super(SimpleMLP, self).__init__()
        self.mode = mode
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.output_layer(x)
        return x


def train_model(model, inputs, targets, epochs=100, learning_rate=0.01):
    if model.mode == 'regression':
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    history = {'loss': []}

    for epoch in range(epochs):
        predictions = model(inputs)
        loss = criterion(predictions, targets)

        if torch.isnan(loss):
            print("Error: Loss is NaN. Stopping.")
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history['loss'].append(loss.item())

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    return history


if __name__ == "__main__":
    X_reg = torch.randn(10, 5)
    y_reg = torch.randn(10, 1)

    model_reg = SimpleMLP(input_size=5, hidden_size=10, output_size=1, mode='regression')
    history_reg = train_model(model_reg, X_reg, y_reg, epochs=50)

    print(f"Final Regression Loss: {history_reg['loss'][-1]:.4f}")

    torch.save(model_reg.state_dict(), 'regression_model.pth')

    X_class = torch.randn(10, 5)
    y_class = torch.randint(0, 3, (10,))

    model_class = SimpleMLP(input_size=5, hidden_size=10, output_size=3, mode='classification')
    history_class = train_model(model_class, X_class, y_class, epochs=50)

    print(f"Final Classification Loss: {history_class['loss'][-1]:.4f}")