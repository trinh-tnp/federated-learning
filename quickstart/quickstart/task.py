"""quickstart: A Flower / PyTorch app."""
import os
from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn
from datasets import load_dataset
from flwr_datasets.partitioner import NaturalIdPartitioner
from torch.utils.data import DataLoader


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(1, 256),
            nn.Sigmoid(),
            nn.Linear(256, 256),
            nn.Sigmoid(),
            nn.Dropout(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input

        #print(f"###################### Input Shape: {x.shape}")
        #print(f"###################### Layer Shape: {self.layer[0].weight.shape}")
        #print(f"###################### Input Dtype: {x.dtype}")
        #print(f"###################### Layer Dtype: {self.layer[0].weight.dtype}")

        x = self.layer(x)
        return x


def load_data(partition_id: int, num_partitions: int):
    """Load partition HelpDesk data."""

    # Get the path to the simulated data file
    _dir = os.path.dirname(os.path.abspath(__file__))
    data_file_path = os.path.join(os.path.join(_dir, "data"), "IoT")
    data_file = os.path.join(data_file_path, f"activity_{partition_id + 1}.csv")

    partitioner = NaturalIdPartitioner(partition_by="ActivityID")

    # Load edge local dataset
    print(f"###################### Loading data from {data_file}")
    fds = load_dataset("csv", data_files=data_file, split='train')
    print(f"###################### Raw dataset loaded: {fds.batch}")
    partitioner.dataset = fds

    partition = partitioner.load_partition(0)
    #print(f"##################### Loaded partition {partition_id} with {len(partition)} samples")
    #print(f"##################### Partition: {partition}")

    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    def apply_transforms(batch: dict[str, Any]):
        """Apply transforms to the partition from FederatedDataset."""
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)

    trainloader = DataLoader(partition_train_test["train"], shuffle=True)
    testloader = DataLoader(partition_train_test["test"])
    return trainloader, testloader


def train(net, trainloader, epochs, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-03)
    net.train()
    running_loss = 0.0

    for _ in range(epochs):
        for batch in trainloader:
            current_activity_id_ds = batch["ActivityID"].to(torch.float32).to(device)
            next_activity_id_labels = batch["PreActivityID"].to(torch.long).to(device)

            print(f"###################### TRAIN - Input: {current_activity_id_ds}")
            print(f"###################### TRAIN - Label: {next_activity_id_labels}")

            optimizer.zero_grad()

            predicted_activity_id =  torch.argmax(net(current_activity_id_ds), dim=1)
            print(f"###################### TRAIN - Predict: {predicted_activity_id}")

            loss = criterion(net(current_activity_id_ds).reshape((1,-1)), next_activity_id_labels.reshape((-1)))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    print(f"###################### TRAIN - Average Loss: {avg_trainloss}")
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            current_activity_id_ds = batch["ActivityID"].to(torch.float32).to(device)
            next_activity_id_labels = batch["PreActivityID"].to(torch.long).to(device)

            print(f"###################### TEST- Input: {current_activity_id_ds}")
            print(f"###################### TEST- Label: {next_activity_id_labels}")

            outputs = torch.argmax(net(current_activity_id_ds), dim=1)
            print(f"###################### TEST - Outputs: {outputs}")

            loss += criterion(net(current_activity_id_ds).reshape((1,-1)), next_activity_id_labels.reshape((-1))).item()
            correct += (outputs == next_activity_id_labels).item()
    accuracy = correct / len(testloader.dataset)
    print(f"###################### TEST - Accuracy: {accuracy}")
    loss = loss / len(testloader)
    print(f"###################### TEST - Loss: {loss}")
    return loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
