import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam

from tqdm import tqdm

from lib.dataset import MNISTDataset
from lib.networks import MLPClassifier, ConvClassifier
from lib.utils import UpdatingMean


BATCH_SIZE = 16
NUM_WORKERS = 4
NUM_EPOCHS = 5

"""
--> Question 3.3 
a) By using a Linear classifier :

output :
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1                   [-1, 10]           7,850
================================================================
Total params: 7,850
Trainable params: 7,850
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.00
Params size (MB): 0.03
Estimated Total Size (MB): 0.03
----------------------------------------------------------------
None
100%|█████████| 3750/3750 [00:07<00:00, 531.32it/s]
[Epoch 01] Loss: 0.4121
[Epoch 01] Acc.: 91.1500%
100%|█████████| 3750/3750 [00:07<00:00, 516.42it/s]
[Epoch 02] Loss: 0.3328
[Epoch 02] Acc.: 91.3400%
100%|█████████| 3750/3750 [00:06<00:00, 545.38it/s]
[Epoch 03] Loss: 0.3213
[Epoch 03] Acc.: 90.6300%
100%|█████████| 3750/3750 [00:06<00:00, 550.57it/s]
[Epoch 04] Loss: 0.3147
[Epoch 04] Acc.: 91.7100%
100%|█████████| 3750/3750 [00:06<00:00, 566.41it/s]
[Epoch 05] Loss: 0.3094
[Epoch 05] Acc.: 91.1100%

https://stats.stackexchange.com/questions/426873/how-does-a-simple-logistic-regression-model-achieve-a-92-classification-accurac

The results were suprsinsingly high but there is a reason.

b)

output :

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1                   [-1, 32]          25,120
              ReLU-2                   [-1, 32]               0
            Linear-3                   [-1, 10]             330
================================================================
Total params: 25,450
Trainable params: 25,450
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.00
Params size (MB): 0.10
Estimated Total Size (MB): 0.10
----------------------------------------------------------------
None
100%|█████████| 3750/3750 [00:08<00:00, 445.65it/s]
[Epoch 01] Loss: 0.4133
[Epoch 01] Acc.: 90.3600%
100%|█████████| 3750/3750 [00:08<00:00, 447.23it/s]
[Epoch 02] Loss: 0.2851
[Epoch 02] Acc.: 91.8300%
100%|█████████| 3750/3750 [00:08<00:00, 442.16it/s]
[Epoch 03] Loss: 0.2405
[Epoch 03] Acc.: 93.0800%
100%|█████████| 3750/3750 [00:08<00:00, 440.43it/s]
[Epoch 04] Loss: 0.2188
[Epoch 04] Acc.: 92.9800%
100%|█████████| 3750/3750 [00:08<00:00, 447.43it/s]
[Epoch 05] Loss: 0.2059
[Epoch 05] Acc.: 94.0900%


--> Question 3.4

100%|█████████| 3750/3750 [00:14<00:00, 260.49it/s]
[Epoch 01] Loss: 0.2886
[Epoch 01] Acc.: 96.6000%
100%|█████████| 3750/3750 [00:14<00:00, 251.96it/s]
[Epoch 02] Loss: 0.0970
[Epoch 02] Acc.: 97.7600%
100%|█████████| 3750/3750 [00:14<00:00, 254.11it/s]
[Epoch 03] Loss: 0.0713
[Epoch 03] Acc.: 97.8400%
100%|█████████| 3750/3750 [00:14<00:00, 259.70it/s]
[Epoch 04] Loss: 0.0608
[Epoch 04] Acc.: 97.8700%
100%|█████████| 3750/3750 [00:14<00:00, 257.88it/s]
[Epoch 05] Loss: 0.0525
[Epoch 05] Acc.: 98.0900%


--> Question 3.5

Computation of the number of parameters of MLP:

https://www.quora.com/How-do-you-calculate-the-number-of-parameters-of-an-MLP-neural-network

see the piece of paper and the file

Computation of # params conv:

see the piece of paper

https://towardsdatascience.com/understanding-and-calculating-the-number-of-parameters-in-convolution-neural-networks-cnns-fc88790d530d


"""


def run_training_epoch(net, optimizer, dataloader):
    loss_aggregator = UpdatingMean()
    # Put the network in training mode.
    net.train()
    # Loop over batches.
    for batch in tqdm(dataloader):
        #raise NotImplementedError()
        # Reset gradients.
        # TODO
        optimizer.zero_grad()

        # Forward pass.
        output = net.forward(batch['input'])

        # Compute the loss - cross entropy.
        # Documentation https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html.
        loss = F.cross_entropy(output, batch['annotation'])

        # Backwards pass.
        # TODO
        loss.backward()
        optimizer.step()

        # Save loss value in the aggregator.
        loss_aggregator.add(loss.item())
    return loss_aggregator.mean()


def compute_accuracy(output, labels):
    return torch.mean((torch.argmax(output, dim=1) == labels).float())


def run_validation_epoch(net, dataloader):
    accuracy_aggregator = UpdatingMean()
    # Put the network in evaluation mode.
    net.eval()
    # Loop over batches.
    for batch in dataloader:
        # Forward pass only.
        output = net(batch['input'])

        # Compute the accuracy using compute_accuracy.
        accuracy = compute_accuracy(output, batch['annotation'])

        # Save accuracy value in the aggregator.
        accuracy_aggregator.add(accuracy.item())
    return accuracy_aggregator.mean()


if __name__ == '__main__':
    # Create the training dataset and dataloader.
    train_dataset = MNISTDataset(split='train')
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True
    )
    
    # Create the validation dataset and dataloader.
    valid_dataset = MNISTDataset(split='test')
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    # Create the network.
    #net = MLPClassifier()
    net = ConvClassifier()

    # Create the optimizer.
    optimizer = Adam(net.parameters())

    # Main training loop.
    best_accuracy = 0
    for epoch_idx in range(NUM_EPOCHS):
        # Training code.
        loss = run_training_epoch(net, optimizer, train_dataloader)
        print('[Epoch %02d] Loss: %.4f' % (epoch_idx + 1, loss))

        # Validation code.
        acc = run_validation_epoch(net, valid_dataloader)
        print('[Epoch %02d] Acc.: %.4f' % (epoch_idx + 1, acc * 100) + '%')

        # Save checkpoint if accuracy is the best so far.
        checkpoint = {
            'epoch_idx': epoch_idx,
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        if acc > best_accuracy:
            best_accuracy = acc
            torch.save(checkpoint, f'best-{net.codename}.pth')
