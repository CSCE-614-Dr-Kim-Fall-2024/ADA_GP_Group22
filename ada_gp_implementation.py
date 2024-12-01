import torch
from torch import nn # nn contains all of PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt

# Check PyTorch version
torch.__version__

# Create *known* parameters
weight = 0.7
bias = 0.3

# Create data
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

X[:10], y[:10]

# Create train/test split
train_split = int(0.8 * len(X)) # 80% of data used for training set, 20% for testing 
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

len(X_train), len(y_train), len(X_test), len(y_test)

def plot_predictions(train_data=X_train, 
                     train_labels=y_train, 
                     test_data=X_test, 
                     test_labels=y_test, 
                     predictions=None):
  """
  Plots training data, test data and compares predictions.
  """
  plt.figure(figsize=(10, 7))

  # Plot training data in blue
  plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
  
  # Plot test data in green
  plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

  if predictions is not None:
    # Plot the predictions in red (predictions were made on the test data)
    plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

  # Show the legend
  plt.legend(prop={"size": 14});
plot_predictions();




# Create a DNN model class
input_size = 28 * 28
hidden_sizes = [128, 64]
output_size = 10
class DNNModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DNNModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], output_size)
        )

    def forward(self, x):
        return self.layers(x)
    def train_adagp(model, predictor, dataloader, criterion, optimizer, predictor_optimizer, epochs=10, warmup_epochs=5):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        predictor = predictor.to(device)

        for epoch in range(epochs):
           model.train()
           for i, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                batch_size = inputs.size(0)

                inputs = inputs.view(batch_size, -1)

                if len(inputs.shape) == 4:  # (batch_size, 1, 28, 28) for MNIST
                  inputs = inputs.view(inputs.size(0), -1)  # Flatten to (batch_size, 784)

            # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                if epoch < warmup_epochs:
                # Warm-up phase: Backpropagate true gradients
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Train predictor model
                    with torch.no_grad():
                        predicted_gradients = predictor(inputs.view(batch_size, -1))
                    true_gradients = torch.autograd.grad(loss, model.parameters(), retain_graph=True)
                    predictor_loss = nn.MSELoss()(predicted_gradients, torch.cat([g.view(-1) for g in true_gradients]))
                    predictor_optimizer.zero_grad()
                    predictor_loss.backward()
                    predictor_optimizer.step()
                else:
                # Phase BP
                   if i % 2 == 0:  # Alternate between BP and GP
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    # Train predictor model
                        with torch.no_grad():
                            predicted_gradients = predictor(inputs.view(batch_size, -1))
                        true_gradients = torch.autograd.grad(loss, model.parameters(), retain_graph=True)
                        predictor_loss = nn.MSELoss()(predicted_gradients, torch.cat([g.view(-1) for g in true_gradients]))
                        predictor_optimizer.zero_grad()
                        predictor_loss.backward()
                        predictor_optimizer.step()
                   else:
                    # Phase GP: Use predicted gradients
                        predicted_gradients = predictor(inputs.view(batch_size, -1))
                        optimizer.zero_grad()
                        for param, grad in zip(model.parameters(), predicted_gradients.split([p.numel() for p in model.parameters()])):
                            param.grad = grad.view(param.size())
                        optimizer.step()

           print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")


class GradientPredictor(nn.Module):
    def __init__(self, input_size, output_size):
        super(GradientPredictor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.layers(x)

# Set manual seed since nn.Parameter are randomly initialized
torch.manual_seed(42)

# Create an instance of the model (this is a subclass of nn.Module that contains nn.Parameter(s))

# Define input_size, hidden_sizes and output_size
input_size = 1  # Example input size, adjust as needed
hidden_sizes = [128, 64] # Example hidden sizes, adjust as needed
output_size = 1  # Example output size, adjust as needed

# Create an instance of the model providing the necessary arguments
model_0 = DNNModel(input_size, hidden_sizes, output_size)

# Check the nn.Parameter(s) within the nn.Module subclass we created
list(model_0.parameters())



# Make predictions with model
with torch.inference_mode(): 
    y_preds = model_0(X_test)

# Note: in older PyTorch code you might also see torch.no_grad()
# with torch.no_grad():
#   y_preds = model_0(X_test)

# Check the predictions
print(f"Number of testing samples: {len(X_test)}") 
print(f"Number of predictions made: {len(y_preds)}")
print(f"Predicted values:\n{y_preds}")
plot_predictions(predictions=y_preds)


y_test - y_preds

# Create the loss function
loss_fn = nn.L1Loss() # MAE loss is same as L1Loss

# Create the optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), # parameters of target model to optimize
                            lr=0.01) # learning rate (how much the optimizer should change parameters at each step, higher=more (less stable), lower=less (might take a long time))

torch.manual_seed(42)

# Set the number of epochs (how many times the model will pass over the training data)
epochs = 100

# Create empty loss lists to track values
train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):
    ### Training

    # Put model in training mode (this is the default state of a model)
    model_0.train()

    # 1. Forward pass on train data using the forward() method inside 
    y_pred = model_0(X_train)
    # print(y_pred)

    # 2. Calculate the loss (how different are our models predictions to the ground truth)
    loss = loss_fn(y_pred, y_train)

    # 3. Zero grad of the optimizer
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Progress the optimizer
    optimizer.step()

    ### Testing

    # Put the model in evaluation mode
    model_0.eval()

    with torch.inference_mode():
      # 1. Forward pass on test data
      test_pred = model_0(X_test)

      # 2. Caculate loss on test data
      test_loss = loss_fn(test_pred, y_test.type(torch.float)) # predictions come in torch.float datatype, so comparisons need to be done with tensors of the same type

      # Print out what's happening
      if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")


# Plot the loss curves
plt.plot(epoch_count, train_loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend();

# Find our model's learned parameters
print("The model learned the following values for weights and bias:")
print(model_0.state_dict())
print("\nAnd the original values for weights and bias are:")
print(f"weights: {weight}, bias: {bias}")

# 1. Set the model in evaluation mode
model_0.eval()

# 2. Setup the inference mode context manager
with torch.inference_mode():
  # 3. Make sure the calculations are done with the model and data on the same device
  # in our case, we haven't setup device-agnostic code yet so our data and model are
  # on the CPU by default.
  # model_0.to(device)
  # X_test = X_test.to(device)
  y_preds = model_0(X_test)
y_preds


plot_predictions(predictions=y_preds)

from pathlib import Path

# 1. Create models directory 
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path 
MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict 
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(), # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH) 

# Check the saved file path
!ls -l models/01_pytorch_workflow_model_0.pth

# Instantiate a new instance of our model (this will be instantiated with random weights)
loaded_model_0 = DNNModel(input_size, hidden_sizes, output_size)

# Load the state_dict of our saved model (this will update the new instance of our model with trained weights)
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

# 1. Put the loaded model into evaluation mode
loaded_model_0.eval()

# 2. Use the inference mode context manager to make predictions
with torch.inference_mode():
    loaded_model_preds = loaded_model_0(X_test) # perform a forward pass on the test data with the loaded model


# Compare previous model predictions with loaded model predictions (these should be the same)
y_preds == loaded_model_preds
