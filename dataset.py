from lib import *

# Defining a custom dataset class that inherits from PyTorch's data.Dataset
class MyDataset(data.Dataset):
    # Constructor method to initialize the class
    def __init__(self, file_list, transform=None, phase="train"):
        # Store the file list in an instance variable
        self.file_list = file_list
        # Store the transform function in an instance variable
        self.transform = transform
        # Store the phase in an instance variable
        self.phase = phase

    # Overriding the default __len__ method to return the length of the file list
    def __len__(self):
        return len(self.file_list)

    # Overriding the default __getitem__ method to return a transformed image and its label
    def __getitem__(self, idx):
        # Get the image path for the index
        img_path = self.file_list[idx]
        # Read the image from the file
        img = Image.open(img_path)
        # Apply the transform function to the image
        img_transformed = self.transform(img, self.phase)

        # Extract the label from the file path
        if self.phase == "train":
            label = img_path[30:34]
        elif self.phase == "val":
            label = img_path[28:32]

        # Map the label to a numerical value (0 for ants and 1 for bees)
        if label == "ants":
            label = 0
        elif label == "bees":
            label = 1

        # Return the transformed image and its label
        return img_transformed, label