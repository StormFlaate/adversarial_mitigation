import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision.io import read_image



class ISICDataset(Dataset):
    """Custom Dataset for loading ISIC images and annotations.

    Args:
        csv_file (str): Path to the csv file with annotations.
            - Format on column values: 
                [0]: name of the image
                [1..n]: category value -> 1 for True, 0 for False
        root_dir (str): Directory with all the images.
        nrows (int, optional): Number of rows to read from the csv file, defaults to None
        (all rows).
        transform (optional): Optional transform to be applied on an image sample.
        target_transform (optional): Optional transform to be applied on the labels.
        image_file_type (str, optional): File type of the images, defaults to "".
        random_seed (int, optional): random seed for repoproducability
    """

    def __init__(
        self, 
        csv_file:str, 
        root_dir:str, 
        nrows:int=None, 
        transform=None,
        target_transform=None,
        image_file_type:str="",
        random_seed:int=42
        ) -> None:
        """
        Initialize the ISICDataset class by reading in the annotations from the csv file,
        storing the root directory for the images, and storing the image transform and
        target transform.
        """
        df = pd.read_csv(csv_file, nrows=nrows)
        # removes the UNK colunn if it exists (2019 dataset)
        if 'UNK' in df.columns:
            df = df.drop('UNK', axis=1)

        # define the types for all columns (except first) as int
        df[df.columns[1:]] = df[df.columns[1:]].astype({col: int for col in df.columns[1:]})

        self.annotations = df
        self.root_dir:str = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.image_file_type = image_file_type
        self.random_seed = random_seed

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.annotations)


    def __getitem__(self, index:int) -> tuple:
        """Get the image and label at the given index.

        Args:
            index (int): Index of the sample to retrieve.
            radom_seed (int): Random seed for specific image, ensures that each image has
            its own transform

        Returns:
            Tuple: (image, label_tensor) where label_tensor is the class label.
        """
        # getting the root directory path and the name of the file from the csv
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index,0])
        if self.image_file_type:
            img_path = f"{img_path}.{self.image_file_type}"
        
        # reads in the image with torchvision.io read_image function
        image = read_image(img_path)

        # sets the randomness for reproducability, but makes each image have its own 
        # random seed (for randomness in transforms)
        np.random.seed(self.random_seed)
        seed = np.random.randint(len(self.annotations)) 
        torch.manual_seed(seed)

        # reads in correct label -> format 2018 example: [MEL,NV,BCC,AKIEC,BKL,DF,VASC]
        label_tensor = torch.tensor(self.annotations.iloc[index, 1:])

        # The transform is applied to the image to pre-process it. This can include data 
        # augmentation, normalization, etc.
        if self.transform:
            image = self.transform(image)

        # The target_transform is applied to the label to pre-process it.
        if self.target_transform:
            label_tensor = self.target_transform(label_tensor)


        return image, label_tensor
    
