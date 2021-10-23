import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.codename = 'mlp'

        #explanation of how to build the architecture

        # Define the network layers in order.
        # Input is 28 * 28.
        # Output is 10 values (one per class).
        # Multiple linear layers each followed by a ReLU non-linearity (apart from the last).
        #raise  NotImplementedError()

        
        #self.layers = nn.Sequential(
        #    nn.Linear(28 * 28, 10)
        #)
        
        
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 32), #input -> hidden
            nn.ReLU(), #hidden
            nn.Linear(32, 10) # hidden -> output
        )

    def forward(self, batch):
        # Flatten the batch for MLP.
        b = batch.size(0)
        batch = batch.view(b, -1)
        # Process batch using the layers.
        x = self.layers(batch)
        return x


class ConvClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.codename = 'conv'

        # Define the network layers in order.
        # Input is 28x28, with one channel.
        # Multiple Conv2d and MaxPool2d layers each followed by a ReLU non-linearity (apart from the last).
        # Needs to end with AdaptiveMaxPool2d(1) to reduce everything to a 1x1 image.
        #raise NotImplementedError()
        self.layers = nn.Sequential(
            
            nn.Conv2d(1, 8, (3, 3)),        # input layer - 1st hidden layer
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),

            nn.Conv2d(8, 16, (3, 3)),       # 1st hidden - 2nd hidden
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),

            nn.Conv2d(16, 32, (3, 3)),      # 2nd hidden - 3rd hidden
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1),        # output 32-D signal for 1 st linear layer
            
        )

        # Linear classification layer.
        # Output is 10 values (one per class).
        self.classifier = nn.Sequential(
            nn.Linear(32, 10)               # 1 st linear layer - output layer
        )

    def forward(self, batch):
        # Add channel dimension for conv.
        b = batch.size(0)
        batch = batch.unsqueeze(1)
        # Process batch using the layers.
        x = self.layers(batch)
        x = self.classifier(x.view(b, -1))
        return x
