import torch
from torch import nn

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from timeit import default_timer as timer
from tqdm.auto import tqdm
import random

import mlxtend
import torchmetrics
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

from pathlib import Path


print("Version ==================================================")
print(f"Torch version: {torch.__version__}")
print(f"Torch vision version: {torchvision.__version__}")
print("\n\n")




print("Device agnostic code ==========================================")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print("\n\n")




print("Load train and test datasets ====================================")
train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

print(f"Train data size: {len(train_data)}")
print(f"Test data size: {len(test_data)}")
print("\n\n")



print("Create data loaders ==========================================")
# batch size hyperparameter
BATCH_SIZE = 32

# turn dataset into iterable batches
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False)

print(f"Train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
print(f"Test dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")
print("\n\n")



# Tiny VGG architecture #################################
class FashionMNISTModel(nn.Module):
  def __init__(self,
               input_shape: int,
               hidden_units: int,
               output_shape: int):
    super().__init__()
    self.conv_block_1 = nn.Sequential(
      nn.Conv2d(in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2)
    )

    self.conv_block_2 = nn.Sequential(
      nn.Conv2d(in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2)
    )

    self.classifier = nn.Sequential(
      nn.Flatten(),
      nn.Linear(in_features=hidden_units* 7* 7,
                out_features=output_shape)
    )

  def forward(self, x):
    x= self.conv_block_1(x)
    # print(f"Output shape of conv_block_1: {x.shape}")
    x= self.conv_block_2(x)
    # print(f"Output shape of conv_block_2: {x.shape}")
    x= self.classifier(x)
    # print(f"Output shape of classifier: {x.shape}")
    return x



print("Display sample image shape ======================================")
image, label = train_data[0]
print(f"Image shape: {image.shape}")
print("\n\n")




print("Display class names and idx =====================================")
class_names= train_data.classes
class_idx = train_data.class_to_idx
print(f"Class names: {class_names}")
print(f"Class idx: {class_idx}")
print("\n\n")



# create model #############################################
torch.manual_seed(42)

model = FashionMNISTModel(input_shape=1,
                          hidden_units=10,
                          output_shape=len(class_names)).to(device)



# loss function and optimizer #################################
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=0.1)



#  train step functions #############################################
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device = device):

  train_loss, train_acc = 0, 0

  # put model into training mode
  model.train()

  # loop through the training batches
  for batch, (X, y) in enumerate(dataloader):
    # put data on target device
    X, y = X.to(device), y.to(device)

    # forward pass
    y_pred = model(X)

    # calculate loss
    loss = loss_fn(y_pred, y)
    train_loss += loss
    train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

    # optimizer zero grad
    optimizer.zero_grad()

    # loss backward
    loss.backward()

    # optimizer step
    optimizer.step()

  # batch loss = mean(batch samples)
  train_loss /= len(dataloader)
  train_acc /= len(dataloader)
  print(f"Train loss= {train_loss:.5f} | Train acc= {train_acc:.2f}%")




# test step function ###########################################
def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device= device):
  test_loss, test_acc = 0, 0

  # put the model in eval mode
  model.eval()
  with torch.inference_mode():
    for X , y in dataloader:

      # send the data to the target device
      X, y = X.to(device), y.to(device)

      # forward pass
      test_pred = model(X)

      # loss and accuracy
      test_loss += loss_fn(test_pred, y)
      test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))

    # calculate the test loss and acc average per batch
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

  print(f"Test loss= {test_loss:.5f}, Test acc= {test_acc:.2f}%")




# create accuracy function #########################################
def accuracy_fn(y_true, y_pred):
  correct = 0
  for i in range(len(y_pred)):
    if y_pred[i].item() == y_true[i].item():
      correct+= 1
  acc = (correct / len(y_pred)) * 100
  return acc


print("Training loop =====================================")
torch.manual_seed(42)
torch.cuda.manual_seed(42)

train_time_start = timer()

epochs = 1

for epoch in tqdm(range(epochs)):
  print(f"Epoch: {epoch}\n--------")
  train_step(model=model,
             dataloader=train_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             accuracy_fn=accuracy_fn,
             device=device)
  test_step(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            device=device)

train_time_end = timer()

total_train_time = train_time_end - train_time_start
print(f"Total train time: {total_train_time:.2f} seconds.")
print("\n\n")




# model evaluation function ###############################
def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device= device):
  loss, acc = 0, 0
  model.eval()
  with torch.inference_mode():
    for X, y in data_loader:

      # make our data device-agnostic
      X, y= X.to(device), y.to(device)

      # make predictions
      y_pred = model(X)

      loss += loss_fn(y_pred, y)
      acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

    loss /= len(data_loader)
    acc /= len(data_loader)

  return f"Model name: {model.__class__.__name__}\n Model loss: {loss.item():.4f} \n Model accuracy: {acc:.2f}%"



print("Model evaluation ===============================================")
model_results = eval_model(model=model,
                           data_loader=test_dataloader,
                           loss_fn=loss_fn,
                           accuracy_fn=accuracy_fn,
                           device=device)

print(f"Model results:\n {model_results}")
print("\n\n")



# make predictions function ############################################
def make_predictions (model: torch.nn.Module,
                      data: list,
                      device = device):
  y_prob_array = []
  model.to(device)
  model.eval()
  with torch.inference_mode():
    for sample in data:
      # prepare the sample
      sample = torch.unsqueeze(sample, dim=0).to(device)

      # forward pass
      y_logit = model(sample)

      # get predictions probability
      y_prob = torch.softmax(y_logit.squeeze(), dim=0)

      # get y_prob out of gpu for further calculations
      y_prob_array.append(y_prob.cpu())

  # stack the y_prob_array to turn list into a tensor
  return torch.stack(y_prob_array)


print("Plot predictions of 9 samples =======================================")
random.seed(42)

# get 9 random samples
test_samples = []
test_labels = []
for sample, label in random.sample(list(test_data), k=9):
  test_samples.append(sample)
  test_labels.append(label)

# make predictions on 9 samples
y_prob = make_predictions(model=model,
                              data=test_samples)

# convert prediction probabilities to labels
pred_classes = y_prob.argmax(dim=1)

print(f"Pred classes: {pred_classes}")
print(f"Test labels: {test_labels}")

# show 9 samples images
plt.figure(figsize=(9, 9))
nrows= 3
ncols= 3
for i, sample in enumerate(test_samples):
  # create subplot
  plt.subplot(nrows, ncols, i+1)

  # plot the target image
  plt.imshow(sample.squeeze(), cmap="gray")
  plt.axis(False)

  # find the prediction in text form
  pred_label = class_names[pred_classes[i]]

  # get the truth label in text form
  truth_label = class_names[test_labels[i]]

  # create a title for the plot
  title_text= f"Pred: {pred_label} | Truth: {truth_label}"

  # check for equality btw pred and truth and change color of title text
  if pred_label == truth_label:
    plt.title(title_text, fontsize=10, c="g") # green text if pred = truth
  else:
    plt.title(title_text, fontsize=10, c="r") # red otherwise
plt.show()
print("\n\n")




print("Plot confusion matrix ===================================")
y_preds = []
model.eval()
with torch.inference_mode():
  for X, y in tqdm(test_dataloader):
    # send X and y to target device
    X, y = X.to(device), y.to(device)

    # forward pass
    y_logit = model(X)

    # turn from logits to probabilities to labels
    y_pred = torch.softmax(y_logit.squeeze(), dim=0).argmax(dim=1)

    # put prediction on CPU for evaluation
    y_preds.append(y_pred.cpu())

# concatenate list of predictions into a tensor
y_pred_tensor = torch.cat(y_preds)
print(f"Predictions: {y_pred_tensor} | Number of samples: {len(y_pred_tensor)}")


confmat = ConfusionMatrix(task="multiclass", num_classes=len(class_names))
confmat_tensor= confmat(preds=y_pred_tensor,
                        target=test_data.targets)

# plot confusion matrix
fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(),
    class_names=class_names,
    figsize=(10, 7)
)
print("\n\n")




print("Saving model ===============================================")
# create model dictory path
MODEL_PATH= Path("checkpoints")
MODEL_PATH.mkdir(parents=True,
                 exist_ok=True)

# create model save
MODEL_NAME= "computer_vision_model.pt"
MODEL_SAVE_PATH= MODEL_PATH/ MODEL_NAME

# save model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(),
           f=MODEL_SAVE_PATH)
print("\n\n")



print("Loading and evaluating saved model ============================================")
torch.manual_seed(42)

loaded_model= FashionMNISTModel(input_shape=1,
                                hidden_units=10,
                                output_shape=len(class_names))

loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

loaded_model.to(device)

torch.manual_seed(42)

loaded_model_results= eval_model(
    model=loaded_model,
    data_loader=test_dataloader,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn
)
print(f"Loaded model results: \n{loaded_model_results}")
print("\n\n")
