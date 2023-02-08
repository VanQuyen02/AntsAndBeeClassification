"""
1. Import required modules from "lib", "image_transform", "config", "utils", and "dataset".
2. Define a main() function to run the program.
3. Create two data path lists for train and validation data using make_datapath_list function.
4. Create two datasets for train and validation data using MyDataset class with respective data path lists and image transformations.
5. Create two data loaders for train and validation datasets using torch.utils.data.DataLoader.
6. Load VGG16 model and replace its last layer to have only 2 output classes.
7. Define Cross Entropy Loss as the criterion for training.
8. Define a Stochastic Gradient Descent (SGD) optimizer with different learning rates for different parts of the network.
9. Train the model using the train_model function, which takes the network, data loaders, criterion, optimizer, and number of epochs as inputs.
10. If the script is run as a main program, the main() function will be called.
The code also has an option to load a pretrained model from a save_path but it's commented out.
"""
from lib import *
from image_transform import ImageTransform
from config import *
from utils import make_datapath_list, train_model, params_to_update, load_model
from dataset import MyDataset

def main():
    train_list = make_datapath_list("train")
    val_list = make_datapath_list("val")

    # dataset
    train_dataset = MyDataset(train_list, transform=ImageTransform(resize, mean, std), phase="train")
    val_dataset = MyDataset(val_list, transform=ImageTransform(resize, mean, std), phase="val")

    # dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=False)
    dataloader_dict = {"train": train_dataloader, "val": val_dataloader}

    # network
    use_pretrained = True
    net = models.vgg16(pretrained=use_pretrained)
    net.classifier[6] = nn.Linear(in_features=4096, out_features=2)

    # loss
    criterior = nn.CrossEntropyLoss()

    # optimizer
    params1, params2, params3 = params_to_update(net)
    optimizer = optim.SGD([
        {'params': params1, 'lr': 1e-4},
        {'params': params2, 'lr': 5e-4},
        {'params': params3, 'lr': 1e-3},
    ], momentum=0.9)

    # training
    train_model(net, dataloader_dict, criterior, optimizer, num_epochs)


if __name__ == "__main__":
    main()

    # network
    # use_pretrained = True
    # net = models.vgg16(pretrained=use_pretrained)
    # net.classifier[6] = nn.Linear(in_features=4096, out_features=2)
    #
    # load_model(net, save_path)