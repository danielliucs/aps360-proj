# imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def train(net, train_set, val_set, batch_size=32, learning_rate=0.01, num_epochs=30):

    # reproducible results
    torch.manual_seed(1000)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    # loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr = learning_rate)

    # training accuracy/loss, validation accuracy/loss
    train_acc = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    val_acc = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)

    # training network
    for epoch in range(num_epochs):

        total_train_loss = 0.0
        total_train_acc = 0.0
        total_epoch = 0

        for i, data in enumerate(train_loader):

            # getting data
            inputs, labels = data
            inputs = torch.squeeze(inputs)
            
            # calculating outputs and loss
            outputs = net(inputs)
            loss = criterion(outputs, labels.float())

            # gradient descent
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # keeping track of training accuracy and loss
            corr = (outputs > 0).squeeze().long() == labels
            total_train_acc += int(corr.sum())
            total_train_loss += loss.item()
            total_epoch += len(labels)

        # accuracies and losses
        train_acc[epoch] = float(total_train_acc) / total_epoch
        train_loss[epoch] = float(total_train_loss) / (i+1)
        val_acc[epoch], val_loss[epoch] = evaluate(net, val_loader, criterion)

        # printing accuracies and losses
        print(("Epoch {}: ").format(epoch + 1))
        print(("Train acc: {}, Train loss: {}").format(
                   train_acc[epoch],
                   train_loss[epoch]))
        print(("Validation acc: {}, Validation loss: {}").format(
                   val_acc[epoch],
                   val_loss[epoch]))
        
        # saving a model checkpoint
        model_path = get_model_name(net.name, batch_size, learning_rate, epoch)
        torch.save(net.state_dict(), model_path)
    
    # writing into csv files for plotting
    epochs = np.arange(1, num_epochs + 1)
    np.savetxt("{}_train_acc.csv".format(model_path), train_acc)
    np.savetxt("{}_train_loss.csv".format(model_path), train_loss)
    np.savetxt("{}_val_acc.csv".format(model_path), val_acc)
    np.savetxt("{}_val_loss.csv".format(model_path), val_loss)

def evaluate(net, loader, criterion):

    with torch.no_grad():

        total_loss = 0.0
        total_acc = 0.0
        total_epoch = 0
    
        for i, data in enumerate(loader):
        
            # getting data
            inputs, labels = data
            inputs = torch.squeeze(inputs)
    
            # calculating outputs and loss
            outputs = net(inputs)
            loss = criterion(outputs, labels.float()) 
            # keeping track of accuracy and loss
            corr = (outputs > 0).squeeze().long() == labels
            total_acc += int(corr.sum())
            total_loss += loss.item()
            total_epoch += len(labels)
    
        # accuracies and losses
        acc = float(total_acc) / total_epoch
        loss = float(total_loss) / (i + 1)
    
        return acc, loss

def get_model_name(name, batch_size, learning_rate, epoch):

    path = "out/model_{0}_bs{1}_lr{2}_epoch{3}".format(name, batch_size, learning_rate, epoch)
    return path

def plot_training_curve(path):
    
    train_acc = np.loadtxt("{}_train_acc.csv".format(path))
    val_acc = np.loadtxt("{}_val_acc.csv".format(path))
    train_loss = np.loadtxt("{}_train_loss.csv".format(path))
    val_loss = np.loadtxt("{}_val_loss.csv".format(path))
    plt.title("Train vs Validation Accuracy")
    n = len(train_acc) # number of epochs
    plt.plot(range(1,n+1), train_acc, label="Train")
    plt.plot(range(1,n+1), val_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()
    plt.title("Train vs Validation Loss")
    plt.plot(range(1,n+1), train_loss, label="Train")
    plt.plot(range(1,n+1), val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()
