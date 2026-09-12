import os

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
from PIL import Image


# version ===========================================
print(f"Torch version: {torch.__version__}")
print(f"Torch vision version: {torchvision.__version__}")
print("\n\n")



# device configuration ====================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print("\n\n")


# walk through dataset directory ==========================
IMAGE_PATH= Path("arm_dataset/")
def walk_through_dir(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories, {len(filenames)} images in {dirpath}")

walk_through_dir(dir_path=IMAGE_PATH)

TRAIN_DIR = "arm_dataset/train"
TEST_DIR = "arm_dataset/test"
print("\n\n")


# write transform for image ============================
data_transform = transforms.Compose([
    # Resize the images to 64x64
    transforms.Resize(size=(244, 244)),
    # Turn the image into a torch.Tensor
    transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
])

# create datasets ====================================
train_data = datasets.ImageFolder(root=TRAIN_DIR, # target folder of images
                                  transform=data_transform, # transforms to perform on data (images)
                                  target_transform=None) # transforms to perform on labels (if necessary)

test_data = datasets.ImageFolder(root=TEST_DIR,
                                 transform=data_transform)

print(f"Train data:\n{train_data}")
print(f"Test data:\n{test_data}")
print("\n\n")


# get class names as a list ================
class_names = train_data.classes
print(f"Class names: {class_names}")

class_dict = train_data.class_to_idx
print(f"Class dict: {class_dict}")
print("\n\n")


# train and test data loaders ======================
BATCH_SIZE =4
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE, # how many samples per batch?
                              shuffle=True) # shuffle the data?

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False) # don't usually need to shuffle testing data

print(f"Train data loader: {train_dataloader}")
print(f"Test data loader: {test_dataloader}")
print("\n\n")


# print batch sample shape =======================
X_batch, y_batch = next(iter(train_dataloader))

print(f"Batch image shape: {X_batch.shape}")
print(f"Batch label shape: {y_batch.shape}")
print("\n\n")


# Tiny VGG architecture #################################
class TinyVGG(nn.Module):
  def __init__(self,
               input_shape: int = 3,
               hidden_units: int =20,
               output_shape: int= len(class_names)):
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
      nn.Linear(in_features=hidden_units* 61* 61,
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


# residual block ===========================
class BasicBlock(nn.Module):
  def __init__(self, in_channels, out_channels, stride=1):
    super().__init__()

    self.conv1 = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(inplace=True)

    self.conv2 = nn.Conv2d(
        in_channels=out_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )
    self.bn2 = nn.BatchNorm2d(out_channels)

    self.shortcut = nn.Sequential()

    if stride != 1 or in_channels != out_channels:
        self.shortcut = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )

  def forward(self, x):

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    out += self.shortcut(x)
    out = self.relu(out)

    return out

# ResNet-18 architecture ========================
class ResNet18(nn.Module):
  def __init__(self,
             color_channels: int= 3,
             num_classes: int= 3):
    super().__init__()

    self.in_channels = 16

    self.conv1 = nn.Conv2d(
        in_channels=color_channels,
        out_channels=16,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )

    self.bn1 = nn.BatchNorm2d(16)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(
        kernel_size=3,
        stride=2,
        padding=1
    )

    self.layer1 = self._make_layer(
        block=BasicBlock,
        out_channels=16,
        num_blocks=2,
        stride=1
    )

    self.layer2 = self._make_layer(
        block=BasicBlock,
        out_channels= 32,
        num_blocks=2,
        stride=2
    )

    self.layer3 = self._make_layer(
        block=BasicBlock,
        out_channels= 64 ,
        num_blocks=2,
        stride=2
    )

    self.layer4 = self._make_layer(
        block=BasicBlock,
        out_channels= 128,
        num_blocks=2,
        stride=2
    )

    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    self.classifier = nn.Linear(
        in_features= 128 ,
        out_features=num_classes
    )

  def _make_layer(self, block, out_channels, num_blocks, stride):
    layers = []

    layers.append(
        block( self.in_channels, out_channels, stride)
    )

    self.in_channels = out_channels

    for b in range(1, num_blocks):
      layers.append(
          block(self.in_channels,out_channels)
      )

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)

    x = self.classifier(x)

    return x

# create model ==========================
torch.manual_seed(42)

model = ResNet18(color_channels=3, num_classes=len(class_names))

model.to(device)

# loss function and optimizer ==================
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=0.01,
                             momentum=0.9,
                            weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=30,gamma=0.1)

# accuracy function =================================
def accuracy_fn(y_true, y_pred):
  correct = 0
  for i in range(len(y_pred)):
    if y_pred[i].item() == y_true[i].item():
      correct+= 1
  acc = (correct / len(y_pred)) * 100
  return acc


#  train step functions ======================
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
  for X_batch, y_batch in dataloader:
    # put data on target device
    X_batch, y_batch= X_batch.to(device), y_batch.to(device)

    # forward pass
    y_pred = model(X_batch)

    # calculate loss
    loss = loss_fn(y_pred, y_batch)
    train_loss += loss.item()
    train_acc += accuracy_fn(y_true=y_batch, y_pred=y_pred.argmax(dim=1))

    # optimizer zero grad
    optimizer.zero_grad()

    # loss backward
    loss.backward()

    # optimizer step
    optimizer.step()

  # batch loss = mean(batch samples)
  train_loss /= len(dataloader)
  train_acc /= len(dataloader)
  return train_loss, train_acc


# test step function  ========================
def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device= device):
  test_loss, test_acc = 0, 0

  # put the model in eval mode
  model.eval()
  with torch.inference_mode():
    for X_batch , y_batch in dataloader:

      # send the data to the target device
      X_batch, y_batch = X_batch.to(device), y_batch.to(device)

      # forward pass
      y_pred = model(X_batch)

      # loss and accuracy
      loss = loss_fn(y_pred, y_batch)
      test_loss += loss.item()
      test_acc += accuracy_fn(y_true=y_batch, y_pred=y_pred.argmax(dim=1))

    # calculate the test loss and acc average per batch
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
  return test_loss, test_acc


# training loop ====================================
torch.manual_seed(42)
torch.cuda.manual_seed(42)

train_time_start = timer()

epochs = 25

train_loss_values, test_loss_values = [], []
train_acc_values, test_acc_values = [], []

for epoch in range(epochs):
  print(f"Epoch: {epoch}\n--------")
  epoch_train_loss, epoch_train_acc = train_step(model=model,
             dataloader=train_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             accuracy_fn=accuracy_fn,
             device=device)
  print(f"Train loss= {epoch_train_loss:.5f} | Train acc= {epoch_train_acc:.2f}%")
  epoch_test_loss, epoch_test_acc = test_step(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            device=device)
  print(f"Train loss= {epoch_test_loss:.5f} | Train acc= {epoch_test_acc:.2f}%")

  # append values for plotting
  train_loss_values.append(epoch_train_loss)
  train_acc_values.append(epoch_train_acc)

  test_loss_values.append(epoch_test_loss)
  test_acc_values.append(epoch_test_acc)

  scheduler.step()

train_time_end = timer()

total_train_time = train_time_end - train_time_start
print(f"Total train time: {total_train_time:.2f} seconds.")
print("\n\n")


# plot loss and accuracy function ======================
def plot_results(train_loss_values,
                      test_loss_values,
                      train_acc_values,
                      test_acc_values,
                      epochs):
  plt.figure(figsize = (20, 6))
  # loss
  plt.subplot(1, 2, 2)
  plt.plot(train_loss_values, label='Train loss')
  plt.plot(test_loss_values, label='Test loss')
  plt.title("Loss")
  plt.xticks(range(0, epochs+1, 1))
  plt.legend()
  plt.grid()
  # accuracy
  plt.subplot(1, 2, 1)
  plt.plot(train_acc_values, label='Train Accuracy')
  plt.plot(test_acc_values, label='Test Accuracy')
  plt.title("Accuracy")
  plt.xticks(range(0, epochs+1, 1))
  plt.legend()
  plt.grid()

  plt.show()

plot_results(train_loss_values,
                  test_loss_values,
                  train_acc_values,
                  test_acc_values,
                  epochs= epochs)

# model evaluation function =====================
def eval_model(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device= device,
               show_confmat= True):
  loss, acc = 0, 0
  y_preds, y_true = [], []

  model.eval()
  with torch.inference_mode():
    for X_batch, y_batch in dataloader:

      # make our data device-agnostic
      X_batch, y_batch= X_batch.to(device), y_batch.to(device)

      # make predictions
      y_logit = model(X_batch)

      loss += loss_fn(y_logit, y_batch).item()
      acc += accuracy_fn(y_true=y_batch, y_pred=y_logit.argmax(dim=1))

      # print(f"Shape logit: {y_logit.shape}, logit: {y_logit}")
      y_prob = y_logit.softmax(dim=1)
      # print(f"Shape prob: {y_prob.shape}, prob: {y_prob}")
      y_pred = y_prob.argmax(dim=1)

      y_preds.append(y_pred.cpu())
      y_true.append(y_batch.cpu())

    loss /= len(dataloader)
    acc /= len(dataloader)

  # concatenate list of predictions into a tensor
  y_pred_tensor = torch.cat(y_preds)
  y_targets_tensor = torch.cat(y_true)

  confmat = ConfusionMatrix(task="multiclass", num_classes=len(class_names))
  confmat_tensor = confmat(preds=y_pred_tensor,
                           target=y_targets_tensor)

  # plot confusion matrix
  if show_confmat:
    fig, ax = plot_confusion_matrix(
      conf_mat=confmat_tensor.numpy(),
      class_names=class_names,
      figsize=(10, 7)
    )
    plt.show()

  return model.__class__.__name__, loss, acc


# model evaluation ===============================================
model_name, model_loss, model_acc = eval_model(model=model,
                           dataloader=test_dataloader,
                           loss_fn=loss_fn,
                           accuracy_fn=accuracy_fn,
                           device=device,
                           show_confmat=True)

print(f"Model results:\n Model name: {model.__class__.__name__}\n Model loss: {model_loss:.4f} \n Model accuracy: {model_acc:.2f}%")
print("\n\n")


# plot predictions of some samples =======================================
random.seed(97)

# get random samples
test_samples = []
test_labels = []
for sample, label in random.sample(list(iter(test_data)), k=9):
  test_samples.append(sample)
  test_labels.append(label)

# make predictions

y_probs = []

model.eval()
with torch.inference_mode():
  for sample in test_samples:
    # prepare the sample
    sample = torch.unsqueeze(sample, dim=0).to(device)

    # forward pass
    y_logit = model(sample)

    # get predictions probability
    y_prob = torch.softmax(y_logit.squeeze(), dim=0)

    # get y_prob out of gpu for further calculations
    y_probs.append(y_prob.cpu())

# stack the y_probs to turn list into a tensor
y_probs =torch.stack(y_probs)

# convert prediction probabilities to labels
pred_classes = y_probs.argmax(dim=1)

print(f"Pred classes: {pred_classes}")
print(f"Test labels: {test_labels}")

# show samples images
plt.figure(figsize=(9, 9))
nrows= 3
ncols= 3
for i, sample in enumerate(test_samples):
  # create subplot
  plt.subplot(nrows, ncols, i+1)

  # plot the target image
  plt.imshow(sample.squeeze().permute(1, 2, 0), cmap="gray")
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





# saving model =============================================
# create model dictory path
MODEL_PATH= Path("checkpoints")
MODEL_PATH.mkdir(parents=True,
                 exist_ok=True)

# create model save
MODEL_NAME= "arm_vgg.pt"
MODEL_SAVE_PATH= MODEL_PATH/ MODEL_NAME

# save model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(),
           f=MODEL_SAVE_PATH)
print("\n\n")




# loading and evaluating saved model =========================================
torch.manual_seed(42)

loaded_model = ResNet18(color_channels=3,
                          num_classes=len(class_names))

loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

loaded_model.to(device)

torch.manual_seed(42)

model_name, model_loss, model_acc = eval_model(
    model=loaded_model,
    dataloader=test_dataloader,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn,
    show_confmat=False
)
print(f"Model results:\n Model name: {model.__class__.__name__}\n Model loss: {model_loss:.4f} \n Model accuracy: {model_acc:.2f}%")
print("\n\n")

