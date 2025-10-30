import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import csv
import os
from tqdm import tqdm

# Definiton of NN containing policy and value head
class AdvancedChessPolicyNetwork(nn.Module):
    def __init__(self, input_channels=6, output_size=4672):
        super(AdvancedChessPolicyNetwork, self).__init__()
        
        # First part
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # (Residual Blocks)
        self.residual_block = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512)
        )
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)  # Dropout dla poprawy generalizacji
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        
        # Fully connected layers for Policy Head
        self.policy_fc1 = nn.Linear(512, 512)
        self.policy_fc2 = nn.Linear(512, 256)
        self.policy_fc3 = nn.Linear(256, output_size)

        # Fully connected layers for Value Head
        self.value_fc1 = nn.Linear(512, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # Conv layers
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        
        # Resiudal block
        residual = x
        x = self.residual_block(x)
        x += residual
        x = self.relu(x)
        
        # Dropout
        x = self.dropout(x)
        
        # Global Average Pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)

        # Policy Head
        policy = self.relu(self.policy_fc1(x))
        policy = self.relu(self.policy_fc2(policy))
        policy = self.policy_fc3(policy)

        # Value Head
        value = self.relu(self.value_fc1(x))
        value = self.value_fc2(value)
        value = torch.tanh(value)  # Value between -1 and 1

        return policy, value

# Function for model training
def train_advanced_model(x_filename='X_data.h5', y_filename='y_data.h5', value_filename='position_values.h5', legal_moves_filename='legal_moves.h5', epochs=10, batch_size=32, learning_rate=0.0001):
    # Check if the model exists
    model = AdvancedChessPolicyNetwork(input_channels=6, output_size=4672)
    if os.path.exists("/kaggle/input/testset-1/advanced_policy_network.pth"):
        state_dict = torch.load("/kaggle/input/testset-1/advanced_policy_network.pth", map_location=torch.device('cpu'))
        # Usuń niepasujące klucze
        state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
        model.load_state_dict(state_dict, strict=False)
        print("Model loaded from file.")
    else:
        print("New advanced model created.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Load the data
    x_file = h5py.File(x_filename, 'r')
    y_file = h5py.File(y_filename, 'r')
    value_file = h5py.File(value_filename, 'r')
    legal_moves_file = h5py.File(legal_moves_filename, 'r')

    x_data = x_file['X']
    y_data = y_file['y']
    value_data = value_file['pos_values']
    legal_moves_data = legal_moves_file['lm'] # TODO: edit changed

    # Optimizer and Loss Function setup
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion_policy = nn.CrossEntropyLoss(reduction='none')  # Loss function for Policy Head without masking
    criterion_value = nn.MSELoss()  # Loss function for Value Head
    
    # Lists for traing data
    training_losses = []
    training_accuracies = []
    
    num_samples = len(x_data)
    num_batches = num_samples // batch_size

    # Training process
    for epoch in range(epochs):
        model.train()
        running_policy_loss = 0.0
        running_value_loss = 0.0
        correct = 0
        total = 0
        
        # Tqdm for progress tracking
        for batch_idx in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            
            # Load the batch of data
            X_batch = torch.tensor(x_data[start_idx:end_idx], dtype=torch.float32).to(device)
            y_batch = torch.tensor(y_data[start_idx:end_idx], dtype=torch.long).to(device)
            value_batch = torch.tensor(value_data[start_idx:end_idx], dtype=torch.float32).to(device)

            # Reorganize the tensor size
            X_batch = X_batch.permute(0, 3, 1, 2)  # (batch_size, height, width, channels) -> (batch_size, channels, height, width)

            # Mask technique for legal moves
            batch_masks = []
            for i in range(start_idx, end_idx):
                mask = torch.zeros(4672, dtype=torch.float32)
                legal_moves = legal_moves_data[i]
                mask[legal_moves] = 1.0
                batch_masks.append(mask)
            batch_masks = torch.stack(batch_masks).to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            policy_outputs, value_outputs = model(X_batch)
            
            # Mask neurons responsible for illegal moves
            masked_policy_outputs = policy_outputs + (batch_masks - 1) * 1e5  # -inf dla nielegalnych ruchów
            
            # Loss-Function calculation
            loss_policy = criterion_policy(masked_policy_outputs, y_batch).mean()  # Maskowanie strat dla Policy Head
            loss_value = criterion_value(value_outputs.squeeze(), value_batch)  # Strata dla Value Head
            loss = loss_policy + loss_value
            
            # Backpropagation i optimization
            loss.backward()
            optimizer.step()
            
            # Gathering the data for loss and accuracy
            running_policy_loss += loss_policy.item()
            running_value_loss += loss_value.item()
            _, predicted = torch.max(masked_policy_outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
        
        epoch_policy_loss = running_policy_loss / num_batches
        epoch_value_loss = running_value_loss / num_batches
        epoch_accuracy = 100 * correct / total
        training_losses.append((epoch_policy_loss, epoch_value_loss))
        training_accuracies.append(epoch_accuracy)
        
        print(f"Epoch [{epoch+1}/{epochs}], Policy Loss: {epoch_policy_loss:.4f}, Value Loss: {epoch_value_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
        
        # Save the model after each epoch
        torch.save(model.state_dict(), f"advanced_policy_value_network{epoch}.pth")
    
    # Save the training results in csv file
    with open('training_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Policy Loss', 'Value Loss', 'Accuracy'])
        for epoch in range(epochs):
            writer.writerow([epoch + 1, training_losses[epoch][0], training_losses[epoch][1], training_accuracies[epoch]])
    
    print("Training complete. Results saved to training_results.csv.")

    # Clos HDF5 files
    x_file.close()
    y_file.close()
    value_file.close()
    legal_moves_file.close()

# Start the training
train_advanced_model(x_filename='/kaggle/input/training-set-2019/X_data.h5', y_filename='/kaggle/input/training-set-2019/y_data.h5', value_filename='/kaggle/input/training-set-2019/position_values.h5', legal_moves_filename='/kaggle/input/training-set-2019/legal_moves.h5', epochs=15, batch_size=32, learning_rate=0.0001)