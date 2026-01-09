import torch
from main import SimpleMLP


def predict_new_data():
    new_input = torch.tensor([[0.5, -1.2, 0.3, 0.9, -0.4]])

    model = SimpleMLP(input_size=5, hidden_size=10, output_size=1, mode='regression')

    try:
        model.load_state_dict(torch.load('regression_model.pth'))
        model.eval()
    except FileNotFoundError:
        print("Model file not found.")
        return

    with torch.no_grad():
        prediction = model(new_input)

    print(f"Input: {new_input.tolist()}")
    print(f"Predicted Output: {prediction.item():.4f}")


if __name__ == "__main__":
    predict_new_data()