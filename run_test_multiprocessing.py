from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time
import multiprocessing as mp
from config import (
    IMAGE_FILE_TYPE, TRAIN_2018_LABELS, TRAIN_2018_ROOT_DIR, BATCH_SIZE, TRAIN_NROWS
)
from customDataset import ISICDataset


if __name__ == '__main__':
    mp.freeze_support()
    mp.set_start_method('spawn')

    transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
    ])

    train_dataset_2018 = ISICDataset(
        csv_file=TRAIN_2018_LABELS, 
        root_dir=TRAIN_2018_ROOT_DIR, 
        transform=transform,
        image_file_type=IMAGE_FILE_TYPE,
        nrows=TRAIN_NROWS
    )

    for num_workers in range(2, mp.cpu_count(), 2):  
        print("Number of workers:", num_workers)
        train_loader = DataLoader(
            train_dataset_2018, 
            batch_size=BATCH_SIZE,
            num_workers=num_workers,
            pin_memory=True
        )
        
        start = time()
        for epoch in range(1, 3):
            for i, data in tqdm(enumerate(train_loader, 0)):
                pass
        end = time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))
