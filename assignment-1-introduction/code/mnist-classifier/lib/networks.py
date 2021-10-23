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
        
        """
        parameters:
        
        input - hidden:   784 * 32  = 25088
        input bias :      32        = 32

        hidden - output:  32 * 10   = 320
        hidden bias:      10        = 10

        total learnable parameters: 25450
        We don't cound the ReLU as it has no params
        """

        #for p in self.parameters():
        #    print(p)
        #print("total params:     ",sum(p.numel() for p in self.parameters()))
        #print("learnable params: ",sum(p.numel() for p in self.parameters() if p.requires_grad))
        from torchsummary import summary
        print(summary(self, (28, 28)))

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
        #from torchsummary import summary
        #print(summary(self, (28, 28)))

        """
        ----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 26, 26]              80
              ReLU-2            [-1, 8, 26, 26]               0
         MaxPool2d-3            [-1, 8, 13, 13]               0
            Conv2d-4           [-1, 16, 11, 11]           1,168
              ReLU-5           [-1, 16, 11, 11]               0
         MaxPool2d-6             [-1, 16, 5, 5]               0
            Conv2d-7             [-1, 32, 3, 3]           4,640
              ReLU-8             [-1, 32, 3, 3]               0
 AdaptiveMaxPool2d-9             [-1, 32, 1, 1]               0
           Linear-10                   [-1, 10]             330
================================================================
Total params: 6,218
Trainable params: 6,218
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.13
Params size (MB): 0.02
Estimated Total Size (MB): 0.16
        """

    def forward(self, batch):
        # Add channel dimension for conv.
        b = batch.size(0)
        batch = batch.unsqueeze(1)
        # Process batch using the layers.
        x = self.layers(batch)
        x = self.classifier(x.view(b, -1))
        return x
