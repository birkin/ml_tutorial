import logging, time

import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms, models


## setup  -----------------------------------------------------------

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s [%(module)s-%(funcName)s()::%(lineno)d] %(message)s',
    datefmt='%d/%b/%Y %H:%M:%S' )
log = logging.getLogger(__name__)
log.debug( 'logging ready' )

## hyper parameters
epochs = 10
batch_size = 32
learning_rate = 0.0001

## for GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
log.debug( f'device: {device}' )


## Prepare the dataset ----------------------------------------------

transform_train = transforms.Compose(
    [   transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        transforms.ColorJitter(brightness=1, contrast=1, saturation=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)
log.debug( f'transform_train, ``{transform_train}``') 
"""
log statement yields...
transform_train, ``Compose(
    Resize(size=(224, 224), interpolation=bilinear, max_size=None, antialias=warn)
    RandomHorizontalFlip(p=0.5)
    RandomAffine(degrees=[0.0, 0.0], scale=(0.8, 1.2), shear=[-10.0, 10.0])
    ColorJitter(brightness=(0.0, 2.0), contrast=(0.0, 2.0), saturation=(0.0, 2.0), hue=None)
    ToTensor()
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
)``
"""
transform = transforms.Compose(
    [   transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)
log.debug( f'transform, ``{transform}``' )
"""
Yields...
transform, ``Compose(
    Resize(size=(224, 224), interpolation=bilinear, max_size=None, antialias=warn)
    ToTensor()
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
)``
"""

train_dataset = datasets.ImageFolder('../flower_photos/train', transform=transform_train)
val_dataset = datasets.ImageFolder('../flower_photos/test', transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle=False)

log.debug( f'train_dataset, ``{train_dataset}``' )
log.debug( f'val_dataset, ``{val_dataset}``' )


## view some images -------------------------------------------------

def im_convert( tensor ):
    """ To visualize some sample images. """
    image = tensor.cpu().clone().detach().numpy()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
    image = image.clip(0, 1)
    return image

classes = ('daisy', 'dandelion', 'roses', 'sunflowers', 'tulips')

dataiter = iter(train_loader)
# images, labels = dataiter.next()  # ```AttributeError: '_SingleProcessDataLoaderIter' object has no attribute 'next'```
images, labels = next( dataiter )

fig = plt.figure(figsize=(25, 4))

for idx in np.arange(20):
    log.debug( f'idx, ``{idx}``' )
    ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
    plt.imshow(im_convert(images[idx]))
    ax.set_title(classes[labels[idx].item()])
    
plt.show()  ## needed, on mac, to see images


## Instantiate the model --------------------------------------------

model = models.alexnet(pretrained=True)  ## "Downloading: "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth" to /path/to/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth"
log.debug( f'model, ``{model}``' )
"""
UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, 
    please use 'weights' instead.
UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. 
    The current behavior is equivalent to passing `weights=AlexNet_Weights.IMAGENET1K_V1`. 
    You can also use `weights=AlexNet_Weights.DEFAULT` to get the most up-to-date weights.
"""

log.debug( f'classifier[6] input features, ``{ model.classifier[6].in_features}``' )
log.debug( f'classifier[6] output features, ``{ model.classifier[6].out_features}``' )

## Prevent the feature-extraction part of the network from being updated during training
for param in model.parameters():
    param.requires_grad = False
"""
- In this context, the code snippet is used to "freeze" the feature extraction part 
    of a pre-trained neural network. 
- By setting param.requires_grad to False, the gradients for those parameters 
    will not be computed during backpropagation, and thus, their values will not be updated 
    during the optimization process. 
- This is typically done when you want to use the pre-trained feature extraction part as-is 
    and only train the remaining layers, e.g., the classifier part of the network, 
    for a new task (transfer learning).
(Thanks, chatGPT!)
"""

## Replace the last layer of the classifier with our current 5-class classifier
n_inputs = model.classifier[6].in_features
last_layer = nn.Linear(n_inputs, len(classes))
model.classifier[6] = last_layer
model.to(device)
log.debug( f'after update, classifier[6] input features, ``{ model.classifier[6].in_features}``' )
log.debug( f'after update, classifier[6] output features, ``{ model.classifier[6].out_features}``' )


## Define the loss function and optimizer ----------------------------

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)


"""
chatGPT prompt...

Describe, to a 7th-grader, an overview of the concepts: 
- "loss function", 
- "optimizer" (in this machine-learning context), 
- "cross-entropy loss"
- "stochastic gradient descent technique"
- "first and second-order moment estimation"

Feel free to use up to ten paragraphs.

---

- Loss Function:
    Imagine you're playing a game where you need to throw a ball into a basket. The "loss function" is like a scorekeeper that tells you how far off your throw was from the target. In machine learning, a loss function measures the difference between the model's predictions and the actual answers (like the correct labels for images). The goal is to minimize this difference so that the model makes better predictions.
- Optimizer:
    An "optimizer" is like a coach who helps you improve your throws in the game. In machine learning, an optimizer is an algorithm that helps the model learn by making small adjustments to the model's parameters. These adjustments help to reduce the difference between the model's predictions and the actual answers, ultimately making the model more accurate.
- Cross-Entropy Loss:
    "Cross-entropy loss" is a specific type of loss function used for problems where the model has to choose between different categories (like identifying which animal is in a picture). It measures how well the model's predicted probabilities match the actual probabilities of the correct categories. A lower cross-entropy loss means the model is doing a better job at predicting the right categories.
- Stochastic Gradient Descent Technique:
    Imagine you're trying to find the shortest path down a hill while blindfolded. One way to do this is to feel the slope of the ground beneath your feet and take small steps in the direction that goes downhill. "Stochastic Gradient Descent" (SGD) is a similar technique in machine learning that helps the optimizer find the best way to adjust the model's parameters. It does this by using information from the loss function to figure out which direction will minimize the difference between the model's predictions and the actual answers.
- First and Second-Order Moment Estimation:
    In our hill example, "first-order moment estimation" is like feeling the slope of the ground to determine which way is downhill. It tells you the average direction that the model's parameters should be adjusted. "Second-order moment estimation" is like feeling the bumps and dips in the ground, which helps you understand how quickly the slope is changing. Combining these two pieces of information can help the optimizer make better adjustments to the model's parameters.
- To sum up, when training a machine learning model, we need a way to measure how well the model is doing (loss function) and a method to help it improve (optimizer). In this case, the cross-entropy loss function measures the difference between the model's predictions and the correct answers for classification problems, and the Adam optimizer uses the stochastic gradient descent technique, along with first and second-order moment estimation, to make smart adjustments to the model's parameters. All of these concepts work together to help the model learn and make better predictions.
"""


## Training the network ---------------------------------------------

# keep track of loss and accuracy
running_loss_history = []
running_corrects_history = []
val_running_loss_history = []
val_running_corrects_history = []
class_total = list(0. for i in range(5))

for e in range(epochs):
  
  running_loss = 0.0
  running_corrects = 0.0
  val_running_loss = 0.0
  val_running_corrects = 0.0
  
  for inputs, labels in train_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    _, preds = torch.max(outputs, 1)
    running_loss += loss.item()
    running_corrects += torch.sum(preds == labels.data)

  else:
    with torch.no_grad():
      for val_inputs, val_labels in val_loader:
        val_inputs = val_inputs.to(device)
        val_labels = val_labels.to(device)
        val_outputs = model(val_inputs)
        val_loss = criterion(val_outputs, val_labels)
        
        _, val_preds = torch.max(val_outputs, 1)
        val_running_loss += val_loss.item()
        val_running_corrects += torch.sum(val_preds == val_labels.data)
      
    epoch_loss = running_loss/len(train_loader.dataset)
    epoch_acc = running_corrects.float()/ len(train_loader.dataset)
    running_loss_history.append(epoch_loss)
    running_corrects_history.append(epoch_acc)
    
    val_epoch_loss = val_running_loss/len(val_loader.dataset)
    val_epoch_acc = val_running_corrects.float()/ len(val_loader.dataset)
    val_running_loss_history.append(val_epoch_loss)
    val_running_corrects_history.append(val_epoch_acc)
    print('epoch :', (e+1))
    print('training loss: {:.4f}, acc {:.4f} '.format(epoch_loss, epoch_acc.item()))
    print('validation loss: {:.4f}, validation acc {:.4f} '.format(val_epoch_loss, val_epoch_acc.item()))

## end of "Training the network"