import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam

from lib.dataset import Simple2DDataset, Simple2DTransformDataset
from lib.networks import LinearClassifier, MLPClassifier
from lib.utils import UpdatingMean


BATCH_SIZE = 8
NUM_WORKERS = 4
NUM_EPOCHS = 10


"""
Question 2.3 output

[Epoch 01] Loss: 1.0745
[Epoch 01] Acc.: 48.0159%
[Epoch 02] Loss: 0.7637
[Epoch 02] Acc.: 48.4127%
[Epoch 03] Loss: 0.6979
[Epoch 03] Acc.: 49.0079%
[Epoch 04] Loss: 0.6932
[Epoch 04] Acc.: 49.0079%
[Epoch 05] Loss: 0.6933
[Epoch 05] Acc.: 48.2143%
[Epoch 06] Loss: 0.6931
[Epoch 06] Acc.: 48.4127%
[Epoch 07] Loss: 0.6930
[Epoch 07] Acc.: 48.2143%
[Epoch 08] Loss: 0.6930
[Epoch 08] Acc.: 48.6111%
[Epoch 09] Loss: 0.6931
[Epoch 09] Acc.: 49.0079%
[Epoch 10] Loss: 0.6932
[Epoch 10] Acc.: 48.6111%

Answer: The classifier is as good as chance and this is to be expected.
The date is not linearly separable, therefore, we can imagine trying to
fit a line in the 2D plane such that given a new point we can predict its class.
The line would originate in the center at (0, 0). We can see that iirrevelently of
"w" we will always split the blues and the red in half leading in 0.5 probabilty
for each class.

Question 2.4

[Epoch 01] Loss: 0.6052
[Epoch 01] Acc.: 77.7778%
[Epoch 02] Loss: 0.3481
[Epoch 02] Acc.: 99.0079%
[Epoch 03] Loss: 0.1183
[Epoch 03] Acc.: 100.0000%
[Epoch 04] Loss: 0.0474
[Epoch 04] Acc.: 100.0000%
[Epoch 05] Loss: 0.0270
[Epoch 05] Acc.: 100.0000%
[Epoch 06] Loss: 0.0180
[Epoch 06] Acc.: 99.8016%
[Epoch 07] Loss: 0.0128
[Epoch 07] Acc.: 99.8016%
[Epoch 08] Loss: 0.0095
[Epoch 08] Acc.: 100.0000%
[Epoch 09] Loss: 0.0079
[Epoch 09] Acc.: 100.0000%
[Epoch 10] Loss: 0.0065
[Epoch 10] Acc.: 99.8016%

Answer: the accuracy is way better because we are using multiple classifier. Instinctively, they
can be organised circularly around the data. This would help mimic the circle shape around the data
and so it increased greatly the accuracy of the prediction.


Queston 2.5

With x² + y^2 = z
[Epoch 01] Loss: 0.9054
[Epoch 01] Acc.: 64.4841%
[Epoch 02] Loss: 0.6072
[Epoch 02] Acc.: 62.6984%
[Epoch 03] Loss: 0.4898
[Epoch 03] Acc.: 57.7381%
[Epoch 04] Loss: 0.4399
[Epoch 04] Acc.: 77.5794%
[Epoch 05] Loss: 0.3995
[Epoch 05] Acc.: 88.4921%
[Epoch 06] Loss: 0.3620
[Epoch 06] Acc.: 96.4286%
[Epoch 07] Loss: 0.3295
[Epoch 07] Acc.: 93.0556%
[Epoch 08] Loss: 0.2991
[Epoch 08] Acc.: 98.2143%
[Epoch 09] Loss: 0.2716
[Epoch 09] Acc.: 97.6190%
[Epoch 10] Loss: 0.2485
[Epoch 10] Acc.: 99.6032%


The data can be linearly separable if mapped on a surface in higher dimension. For example here, we
could map our data to the surface x^2 + y^2 = z and we would therefore use 3-dimentional vectors
(x, y, x^2 + y^2). We need to find a 3D plane that split our data.

"""

def run_training_epoch(net, optimizer, dataloader):
    loss_aggregator = UpdatingMean()
    # Put the network in training mode.
    net.train()
    # Loop over batches.
    for batch in dataloader:
        #raise NotImplementedError()
        
        # Reset gradients.
        # TODO
        optimizer.zero_grad()

        # Forward pass.
        output = net.forward(batch['input'])
    
        # Compute the loss - binary cross entropy.
        # Documentation https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy.html.
        loss = F.binary_cross_entropy(output, batch['annotation'])

        # Backwards pass.
        # TODO
        loss.backward()
        optimizer.step()

        # Save loss value in the aggregator.
        loss_aggregator.add(loss.item())
    return loss_aggregator.mean()


def compute_accuracy(output, labels):
    return torch.mean(((output >= 0.5) == labels).float())


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
    train_dataset = Simple2DDataset(split='train')
    #train_dataset = Simple2DTransformDataset(split='train')
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True
    )
    
    # Create the validation dataset and dataloader.
    valid_dataset = Simple2DDataset(split='valid')
    #valid_dataset = Simple2DTransformDataset(split='valid')
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    # Create the network.
    #net = LinearClassifier(3)
    net = MLPClassifier()

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
