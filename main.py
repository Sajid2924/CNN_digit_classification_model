from torchvision import datasets
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset , DataLoader
import torch.nn as nn
import torch.optim as optim
torch.manual_seed(42)

# taking data from mnist dataset
df = datasets.MNIST(root='data', train=True, download=True, transform=None)

# converting data to numpy for using train_test_split
images = []
labels = []

for img, label in df:
    images.append(np.array(img))
    labels.append(label)

# 60000, 28, 28
x = np.array(images)
# 60000
y = np.array(labels)

# splitting the data for test/train
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=42)

class CustomDataset(Dataset):
    def __init__(self , features , labels):
        features = features[:, None, :, :]                                    # mnist (N,28,28) ---> CNN (N,1(BlackWhite),28,28)
        features = features/255.0
        self.features = torch.tensor(features , dtype=torch.float32)          # numpy to tensor
        self.labels = torch.tensor(labels, dtype= torch.long)                 # numpy to tensor
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self,index):
        return self.features[index] , self.labels[index]
    
train_dataset = CustomDataset(x_train , y_train)
test_dataset = CustomDataset(x_test , y_test)

# data loading in batches
train_loader = DataLoader(train_dataset , batch_size=32 , shuffle=True)
test_loader = DataLoader(test_dataset , batch_size= 32 , shuffle=False)

# model class
class MyNN(nn.Module):
    def __init__(self,input=1):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input,32,kernel_size=3,padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(32,64,kernel_size=3,padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7 , 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128,64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64,10)
        )
    
    def forward(self,x):
        x=self.features(x)
        x=self.classifier(x)
        return x

epochs = 20
lr = 0.01
model = MyNN()
criterion = nn.CrossEntropyLoss()                                        #loss function
optimizer = optim.SGD(model.parameters(),lr , weight_decay=1e-4)         #optimizer i.e. weight and bias updator

# training loop
for epoch in range(epochs):
    total_loss = 0
    for batch_features , batch_labels in train_loader:
        outputs = model(batch_features)
        loss = criterion(outputs , batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print("epoch :", epoch+1 , "--- loss :" , total_loss)

avg_loss = total_loss / len(train_loader)
print("avg loss :", avg_loss)


# evaluation
model.eval()
total = 0
correct = 0

with torch.no_grad():
    for batch_features , batch_labels in test_loader:
        outputs=model(batch_features)
        _,predicted = torch.max(outputs , 1)

        total += batch_labels.shape[0]
        correct += (predicted == batch_labels).sum().item()

accuracy = correct/total
print("accuracy :", accuracy*100 , "%")