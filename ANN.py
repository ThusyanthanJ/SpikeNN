import os
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision.models import resnet34


# Custom dataset class for loading the data
class VideoDataset(Dataset):
    def __init__(self, folder_path, is_train=True, max_frames=30, transform=None):
        self.folder_path = folder_path
        self.is_train = is_train
        self.max_frames = max_frames
        self.transform = transform

        # Get all class folders (0 swiping left, 1 right, 2 up , 3 down)
        self.classes = os.listdir(folder_path)
        self.video_ids = []  # List to store video ids
        self.labels = []  # List to store labels

        # Split the data: last 50 folders of each class will be test data
        for label, class_name in enumerate(self.classes):
            class_folder = os.path.join(folder_path, class_name)
            # Sort folders lexicographically
            all_folders = sorted(os.listdir(class_folder))

            # Use the last 50 folders as test, the rest as train
            num_test_folders = 50
            if self.is_train:
                folders = all_folders[:-num_test_folders]  # Training folders
            else:
                folders = all_folders[-num_test_folders:]  # Test folders

            for video_id in folders:
                self.video_ids.append(os.path.join(class_name, video_id))
                self.labels.append(label)  # Label is the index of the class

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        video_folder = os.path.join(self.folder_path, video_id)
        if not os.path.exists(video_folder):
            raise FileNotFoundError(f"Folder {video_folder} not found.")

        frames = []
        for img_name in sorted(os.listdir(video_folder)):
            img_path = os.path.join(video_folder, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (64, 64))  # Resize image to 64x64
                frames.append(img)

        if len(frames) > self.max_frames:
            frames = frames[:self.max_frames]

        # Convert to numpy array
        frames = np.array(frames).astype(np.float32) / 255.0  # Normalize

        if self.transform:
            frames = self.transform(frames)

        frames = torch.tensor(frames).permute(
            3, 0, 1, 2)  # Change shape to (C, T, H, W)

        label = self.labels[idx]
        return frames, label


class AdvancedVideoActionRecognitionModel(nn.Module):
    def __init__(self, num_classes=4):
        super(AdvancedVideoActionRecognitionModel, self).__init__()

        # Pre-trained ResNet for spatial feature extraction
        # Load a pre-trained ResNet18 model
        self.resnet = resnet34(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove the final fully connected layer
        self.spatial_feature_dim = 512  # Output dimension of ResNet18 backbone

        # GRU for temporal feature extraction
        self.gru = nn.GRU(input_size=self.spatial_feature_dim,
                          hidden_size=256, num_layers=2, batch_first=True)

        # Fully connected layers for classification
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Input shape: (B, C, T, H, W)
        batch_size, channels, time_steps, height, width = x.shape

        # Extract spatial features for each frame using ResNet
        x = x.permute(0, 2, 1, 3, 4)  # Change to (B, T, C, H, W)
        # Merge batch and time steps: (B*T, C, H, W)
        x = x.reshape(-1, channels, height, width)
        with torch.no_grad():
            # Output shape: (B*T, spatial_feature_dim)
            spatial_features = self.resnet(x)

        # Restore batch and time steps
        spatial_features = spatial_features.view(
            batch_size, time_steps, -1)  # Shape: (B, T, spatial_feature_dim)

        # GRU for temporal feature extraction
        gru_out, _ = self.gru(spatial_features)  # gru_out shape: (B, T, 256)
        x = gru_out[:, -1, :]  # Use the last time step's output

        # Fully connected layers for classification
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Output shape: (B, num_classes)

        return x


def train_and_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Path for the main dataset folder
    # Update with your dataset folder path
    data_folder = 'JESTER DATASET LAST/ANN_4'

    # Load training data
    train_dataset = VideoDataset(data_folder, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Load test data
    test_dataset = VideoDataset(data_folder, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Instantiate the model
    # Number of classes (e.g., 4 classes)
    num_classes = len(train_dataset.classes)
    model = AdvancedVideoActionRecognitionModel(
        num_classes=num_classes).to(device)
    # model = VideoActionRecognitionModel(num_classes=num_classes)
    # model.to(device)
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 50
    train_losses = []
    train_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_predictions / total_predictions * 100
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    # Save the model
    torch.save(model.state_dict(), 'action_recognition_model.pth')

    # Evaluation on test data
    model.eval()
    test_predictions = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            test_predictions.extend(predicted.cpu().numpy())

    # Assuming that you would like to store predictions into a CSV file
    test_df = pd.DataFrame(
        {'video_id': test_dataset.video_ids, 'predicted_label': test_predictions})
    test_df.to_csv('test_predictions.csv', index=False)

    # Plot training accuracy
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training Accuracy')
    plt.show()

    # Plot training loss
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss')
    plt.show()

    # Print final training accuracy and loss
    print(f"Final Training Accuracy: {train_accuracies[-1]:.2f}%")
    print(f"Final Training Loss: {train_losses[-1]:.4f}")

    torch.save(model.state_dict(), 'final_action_recognition_model.pth')


if __name__ == "__main__":
    train_and_evaluate()
