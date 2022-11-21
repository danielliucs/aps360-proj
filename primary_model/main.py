# imports
from model import hybrid_CNN_RNN
from train import train
from train import evaluate
from train import get_model_name
from train import plot_training_curve
import torch
import torchvision




def main():
    # datasets
    train_set = torchvision.datasets.DatasetFolder("C:/Users/User/Desktop/aps360-proj/dataset/train_frames", loader=torch.load, extensions=('.tensorsaved'))
    val_set = torchvision.datasets.DatasetFolder("C:/Users/User/Desktop/aps360-proj/dataset/val_frames", loader=torch.load, extensions=('.tensorsaved'))

   # declaration of network
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        network = hybrid_CNN_RNN(120).cuda()
    else:
        network = hybrid_CNN_RNN(120)

    #Training model
    train(network, train_set, val_set, use_cuda, batch_size=1, learning_rate=0.0001, num_epochs=30)


if __name__ == '__main__':
    main()