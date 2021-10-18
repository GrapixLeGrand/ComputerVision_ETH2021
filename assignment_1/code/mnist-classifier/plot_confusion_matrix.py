import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam

from tqdm import tqdm

from lib.dataset import MNISTDataset
from lib.networks import MLPClassifier, ConvClassifier


BATCH_SIZE = 8
NUM_WORKERS = 4


if __name__ == '__main__': 
    # Create the validation dataset and dataloader.
    valid_dataset = MNISTDataset(split='test')
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    # Create the network.
    net = MLPClassifier()
    #net = ConvClassifier()

    # Load best checkpoint.
    net.load_state_dict(torch.load(f'best-{net.codename}.pth')['net'])
    # Put the network in evaluation mode.
    net.eval()

    # Create the optimizer.
    optimizer = Adam(net.parameters())

    # Based on run_validation_epoch, write code for computing the 10x10 confusion matrix.
    confusion_matrix = np.zeros([10, 10])

    for batch in valid_dataloader:
        
        output = net(batch['input'])
        expected = batch['annotation']

        # for each batch element, for each of the output we need to retain
        # the predicated class, the one with the higher value among the 
        # predicted classes. We also need to retain the class that is the ground
        # truth
        
        for prediction, ground_truth in zip(output, expected):
            predicated_class = torch.argmax(prediction)
            #print("predicted : ", predicated_class, " and is : ", ground_truth)
            i = predicated_class
            j = ground_truth

            confusion_matrix[i][j] += 1



    #raise NotImplementedError()
    #print(run_validation_epoch(net, valid_dataloader))
    
    # Plot the confusion_matrix.
    plt.figure(figsize=[5, 5])
    plt.imshow(confusion_matrix)
    plt.xticks(np.arange(10))
    plt.xlabel('prediction')
    plt.yticks(np.arange(10))
    plt.ylabel('annotation')
    for i in range(10):
        for j in range(10):
            plt.text(i, j, '%d' % (confusion_matrix[i, j]), ha='center', va='center', color='w', fontsize=12.5)
    plt.show()
