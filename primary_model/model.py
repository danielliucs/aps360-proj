import torch
import torch.nn as nn
import torchvision
import torchvision.models

# hybrid CNN RNN model for video classification
class hybrid_CNN_RNN(nn.Module):
    def __init__(self, hidden_size):
        super(hybrid_CNN_RNN, self).__init__()

        self.name = "hybridCNNRNN"

        # using pretrained AlexNet CNN for image feature extraction
        alexnet = torchvision.models.alexnet(pretrained=True)
        self.CNN = alexnet.features

        # using RNN to process CNN output frame by frame
        self.hidden_size = hidden_size
        self.RNN = nn.GRU(9216, hidden_size, batch_first=True)

        # using fully-connected layer for final binary classification
        self.FC = nn.Linear(hidden_size, 1)

    def forward(self, x):

        # CNN-RNN Loop
        i = 0
        f_map = self.CNN(x[i])
        # print(torch.flatten(f_map).unsqueeze(0).shape)
        out, h_n = self.RNN(torch.flatten(f_map).unsqueeze(0))
        for i in range(1, x.size(1)):
            f_map = self.CNN(x[i])
            out, h_n = self.RNN(torch.flatten(f_map).unsqueeze(0), h_n)

        # FC
        out = self.FC(out[0])

        return out
