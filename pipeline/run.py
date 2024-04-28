import pickle
import numpy as np
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
import dataset
import rsample_dataset
import csv
import sys
# look into argparse.ArgumentParser

def main():
    N = int(sys.argv[1])
    M = int(sys.argv[2])
    DSET_TYPE = sys.argv[3]
    BATCH_SIZE = int(sys.argv[4])
    NUM_EPOCHS = int(sys.argv[5])
    MODEL_TO_LOAD = None if sys.argv[6] == 'None' else sys.argv[6]
    SAMPLE_SIZE = int(sys.argv[7])
    SAVE = sys.argv[8]
    LR = sys.argv[9]

    # load pickled data, write to index_file for dataset lookup
    def create_index(pickle_file, index_file):
        offsets = []
        with open(pickle_file, 'rb') as file:
            while True:
                offset = file.tell()
                try:
                    pickle.load(file)
                    offsets.append(offset)
                except EOFError:
                    break
        with open(index_file, 'wb') as file:
            pickle.dump(offsets, file)

    class AddGaussianNoise(object):
        """Add Gaussian noise to a tensor."""
        def __init__(self, mean=0.1, std=0.01):
            self.mean = mean
            self.std = std

        def __call__(self, tensor):
            """
            Adds Gaussian noise to a tensor.
            """
            noise = np.random.randn(1781) * self.std + self.mean
            return tensor + noise

    # datafile = f'data/{DSET_TYPE}_{N}_train.pkl'
    datafile = f'data/{DSET_TYPE}_{N}.pkl'
    indexfile = f'indices/{DSET_TYPE}_{N}_train.idx'

    create_index(datafile, indexfile)

    # get new samples to train on (per epoch)
    def new_loaders():
        dataset = rsample_dataset.RSampleDataset(
            pickle_file=datafile,
            index_file=indexfile,
            n_mixture=M,
            num_classes=N,
            transform=AddGaussianNoise(),
            total_samples=SAMPLE_SIZE,
        )

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Create DataLoader for training and validation
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        return train_loader, val_loader

    class SimpleNet(nn.Module):
        def __init__(self, input_size, num_classes):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(input_size, 100)
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(100, num_classes)  # Output logits, not probabilities

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            # x = self.dropout(x)
            x = self.fc2(x)
            return x

    model = SimpleNet(input_size=1781, num_classes=N)

    # Load model weights
    if MODEL_TO_LOAD:
        model.load_state_dict(torch.load(MODEL_TO_LOAD))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR)

    class WeightedBCEWithLogitsLoss(nn.Module):
        def __init__(self, pos_weight):
            super(WeightedBCEWithLogitsLoss, self).__init__()
            # pos_weight should be a tensor of length equal to the number of classes
            # each weight corresponds to the positive class weight for each label
            self.pos_weight = pos_weight

        def forward(self, outputs, targets):
            # Initialize BCEWithLogitsLoss with pos_weight for handling class imbalance
            criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
            return criterion(outputs, targets)

    # Assume higher weight for the positive classes due to sparsity or imbalance
    pos_weight = torch.ones(N) * 10  
    loss_function = WeightedBCEWithLogitsLoss(pos_weight=pos_weight.to(device))  # Ensure weights are on the same device as your model/data

    # Accuracy measure function
    def top_n_accuracy(preds, labels, n=M, correct_n=M):
        """
        Calculate the top-n accuracy for the given predictions and labels.
        """
        top_n_preds = preds.topk(n, dim=1)[1]  # Returns values and indices; [1] to get indices
        
        # Initialize the score
        score = 0.0

        for i in range(labels.size(0)):
            actual_labels = labels[i].bool()
            # select the label positions that are top n
            correct_preds = actual_labels[top_n_preds[i]].float()  
            score += correct_preds.sum().item() / correct_n

        return score


    def train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_function,
            epoch,
            device,
        ):
        model.train()  # Set the model to training mode
        train_loss = 0.0
        accuracy = 0
        halfway_point = len(train_loader) // 2
        is_halfway_recorded = False

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # Forward pass
            outputs = model(data).squeeze(1)
            loss = loss_function(outputs, target)

            # Convert outputs to predicted labels
            predicted_probs = torch.sigmoid(outputs)  # Sigmoid to convert logits to probabilities
            
            # Calculate if top M predictors are accurate
            accuracy += top_n_accuracy(predicted_probs, target, n=M, correct_n=M)

            # Backward and optimize
            optimizer.zero_grad()  # Clear gradients w.r.t. parameters
            loss.backward()  # Backpropagation
            optimizer.step()  # Update parameters

            train_loss += loss.item() * data.size(0)

            if (batch_idx == halfway_point) and not is_halfway_recorded:
                halfway_train_loss = 2 * train_loss / len(train_loader.dataset)
                halfway_accuracy = 2 * accuracy / len(train_loader.dataset)
                print(f'Epoch: {epoch+1}, Halfway: Train Loss: {halfway_train_loss:.4f}, Train Accuracy: {halfway_accuracy:.4f}')
                is_halfway_recorded = True

        # Calculate average loss
        train_loss /= len(train_loader.dataset)
        train_accuracy = accuracy / len(train_loader.dataset)
        return halfway_train_loss, halfway_accuracy, train_loss, train_accuracy

    def eval_one_epoch(model, val_loader, loss_function, device):
        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        accuracy = 0.0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data).squeeze(1)
                loss = loss_function(outputs, target)

                # Convert outputs to predicted labels
                predicted_probs = torch.sigmoid(outputs)
                accuracy += top_n_accuracy(predicted_probs, target, n=M, correct_n=M)
                val_loss += loss.item() * data.size(0)

        # Calculate average loss
        val_loss /= len(val_loader.dataset)
        val_accuracy = accuracy / len(val_loader.dataset)
        return val_loss, val_accuracy



    # Training loop
    def train_model(model, loss_function, optimizer, num_epochs):
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        # pre-training
        train_loader, val_loader = new_loaders()
        train_loss, train_accuracy = eval_one_epoch(model, train_loader, loss_function, device)
        val_loss, val_accuracy = eval_one_epoch(model, val_loader, loss_function, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(f'Epoch: 0 \t'
            f'Training Loss: {train_loss:.4f} \t Training Accuracy: {train_accuracy:.4f} \t'
            f'Validation Loss: {val_loss:.4f} \t Validation Accuracy: {val_accuracy:.4f}')

        # Training Loop
        for epoch in range(num_epochs):
            # generate new data
            train_loader, val_loader = new_loaders()

            # train
            halfway_loss, halfway_accuracy, train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, loss_function, epoch, device)
            train_losses.append(halfway_loss)
            train_losses.append(train_loss)
            train_accuracies.append(halfway_accuracy)
            train_accuracies.append(train_accuracy)

            # val
            val_loss, val_accuracy = eval_one_epoch(model, val_loader, loss_function, device)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            # Print statistics
            print(f'Epoch: {epoch+1}/{num_epochs} \t'
                f'Training Loss: {train_loss:.4f} \t Training Accuracy: {train_accuracy:.4f} \t'
                f'Validation Loss: {val_loss:.4f} \t Validation Accuracy: {val_accuracy:.4f}')
            
            # Save model checkpoint
            if epoch % SAVE == 0 and epoch != 0:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                    'acc': val_accuracy,
                }, f'checkpoints/M{M}_cp{epoch+1}.pth')
        return train_losses, val_losses, train_accuracies, val_accuracies




    train_losses, val_losses, train_accuracies, val_accuracies = train_model(model, loss_function, optimizer, NUM_EPOCHS)

    # Save model & epoch stats
    torch.save(model.state_dict(), f'final_models/{DSET_TYPE}_{N}_M{M}.pt')

    with open(f'epoch_stats/{DSET_TYPE}_{N}_M{M}.csv', 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(train_losses)
        csv_writer.writerow(val_losses)
        csv_writer.writerow(train_accuracies)
        csv_writer.writerow(val_accuracies)


if __name__ == '__main__':
    main()
