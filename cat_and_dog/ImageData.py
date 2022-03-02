#dataset loader and preprocess
from PIL import Image
from torch.utils.data import Dataset

class CatsDogsDataset(Dataset):
    def __init__(self, img_list, transform=None):
        self.img_list = img_list
        self.transform = transform

    def __getitem__(self, idx):
        img_name =  self.img_list[idx]
        image = Image.open('train/' + img_name)
        if self.transform:
            image = self.transform(image)
        label = 1 if img_name.split('.')[0] == 'dog' else 0

        return image, label

    def __len__(self):
        return len(self.img_list)