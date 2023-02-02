import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms # Import the transforms module from torchvision
from skimage import io

class ISICDataset(Dataset):
    """Custom Dataset for loading ISIC images and annotations.

    Args:
        csv_file (str): Path to the csv file with annotations.
            - Format on column values: 
                [0]: name of the image
                [1..n]: category value -> 1 for True, 0 for False
        root_dir (str): Directory with all the images.
        transform (transforms, optional): Optional transform to be applied on a sample.
        image_file_type (str, optional): File type of the images, defaults to "".
    """

    def __init__(self, csv_file:str, root_dir:str, transform:transforms=None, image_file_type:str="") -> None:
        self.annotations = pd.read_csv(csv_file)
        self.root_dir:str = root_dir
        self.transform = transform
        self.image_file_type = image_file_type

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.annotations)

    def __getitem__(self, index:int) -> tuple:
        """Get the image and label at the given index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            Tuple: (image, label_tensor) where label_tensor is the class label.
        """
        # getting the root directory path and the name of the file from the csv!
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index,0])
        if self.image_file_type:
            img_path = f"{img_path}.{self.image_file_type}"
        
        # reads in the image with scikit-image
        image = io.imread(img_path)
        # reads in correct label -> format: [MEL,NV,BCC,AKIEC,BKL,DF,VASC]
        label_tensor = torch.tensor(self.annotations.iloc[index, 1:])

        if self.transform:
            image = self.transform(image)
        
        return (image, label_tensor)
