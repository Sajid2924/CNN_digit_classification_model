"""
Hyperparameter tuning of CNN using Optuna.
Explores architecture depth, learning rate, optimizer, and regularization.
"""

from torchvision import datasets
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset , DataLoader
import torch.nn as nn
import torch.optim as optim
import optuna
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

# model class
class MyNN(nn.Module):
    def __init__(self, input, num_hidden ,num_neuron , num_filters , dropout_rate , num_pairs):
        super().__init__()

        layers1 = []

        for i in range(num_pairs):
            layers1.append(nn.Conv2d(input , num_filters , kernel_size=3 , padding='same'))
            layers1.append(nn.ReLU())
            layers1.append(nn.MaxPool2d(kernel_size=2 , stride=2))
            input = num_filters
            num_filters = num_filters*2
        
        self.features = nn.Sequential(*layers1)

        num_filters = num_filters//2
        h=28
        for k in range(num_pairs): h=h//2
        inp = num_filters*h*h
        
        layers2 = []
        layers2.append(nn.Flatten())

        for i in range(num_hidden):
            layers2.append(nn.Linear(inp , num_neuron ))
            layers2.append(nn.BatchNorm1d(num_neuron))
            layers2.append(nn.ReLU())
            layers2.append(nn.Dropout(dropout_rate))
            inp=num_neuron

        layers2.append(nn.Linear(num_neuron,10))

        self.classifier = nn.Sequential(*layers2)
    
    def forward(self,x):
        x=self.features(x)
        x=self.classifier(x)
        return x



def objective(trial):
    num_hidden = trial.suggest_int('num_hidden' , 1,5)
    num_neuron = trial.suggest_int('num_neuron', 8, 128, step=8)
    epochs = trial.suggest_int('epochs' , 10 , 30 , step=5)
    lr = trial.suggest_float('lr', 1e-5 , 1e-1 , log=True)
    dropout_rate = trial.suggest_float('dropout_rate' , 0.1 , 0.5 , step=0.1)
    batch_size = trial.suggest_categorical('batch_size' , [16,32,64,128])
    optimizer_name = trial.suggest_categorical('optimizer_name' , ['ADAM' , 'SGD' , 'RMSprop'])
    weight_decay = trial.suggest_float('weight_decay' , 1e-5 , 1e-2 , log=True)
    num_filters = trial.suggest_categorical('num_filters' , [8,16,32,64,128])
    num_pairs = trial.suggest_int('num_pairs' , 1,4)


    train_loader = DataLoader(train_dataset , batch_size=batch_size , shuffle=True)
    test_loader = DataLoader(test_dataset , batch_size= batch_size , shuffle=False)


    model = MyNN(1 , num_hidden ,num_neuron , num_filters , dropout_rate , num_pairs)
    #loss function;
    criterion = nn.CrossEntropyLoss()

    #optimizer i.e. weight and bias updator
    if optimizer_name=='ADAM':
        optimizer = optim.Adam(model.parameters(),lr=lr , weight_decay=weight_decay)
    elif optimizer_name=='SGD':
        optimizer = optim.SGD(model.parameters(),lr = lr , weight_decay=weight_decay)
    else:
        optimizer = optim.RMSprop(model.parameters(),lr =lr , weight_decay=weight_decay)


    # training loop
    for epoch in range(epochs):
        model.train()
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
    return accuracy


study = optuna.create_study(direction = 'maximize')
study.optimize(objective , n_trials=10)

print("accuracy :" , study.best_value)
print("best accuracy achieved for parameters : " , study.best_params)

#--------------OUTPUT-------------------

# accuracy : 0.9925
# best accuracy achieved for parameters :  {
#     'num_hidden': 1, 
#     'num_neuron': 112, 
#     'epochs': 15, 
#     'lr': 7.504712151350896e-05, 
#     'dropout_rate': 0.2, 
#     'batch_size': 32, 
#     'optimizer_name': 'ADAM', 
#     'weight_decay': 1.2697984884583028e-05, 
#     'num_filters': 32, 
#     'num_pairs': 3
#     }
