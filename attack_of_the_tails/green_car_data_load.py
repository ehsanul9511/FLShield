import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

data_path="attack_of_the_tails/green cars"

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path)
        image = image.convert('RGB')
        image = image.resize((32,32))
        if self.transform:
            image = self.transform(image)
        sample=(image,1)

        return sample

# Define the transforms
transform = transforms.Compose([transforms.PILToTensor(),   
                                # resize to 32x32
                                #transforms.Resize([32,32,3]),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(32, padding=4),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Create the dataset
dataset = CustomDataset(root_dir=data_path, transform=None)

torch.save(dataset, 'attack_of_the_tails/more_green_cars_dataset.pt')

# Create the data loader
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
