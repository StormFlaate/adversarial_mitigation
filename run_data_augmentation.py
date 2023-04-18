# Import necessary libraries
import argparse
from torchvision import transforms

# Import custom modules
from config import (
    MIN_MAX_ROTATION_RANGE, 
    RANDOM_HORIZONTAL_FLIP_PROBABILITY, 
    RANDOM_VERTICAL_FLIP_PROBABILITY
)
from helper_functions.data_augmentation_helper import (
    augment_2018_images_and_save_to_file, 
    augment_2019_images_and_save_to_file
)


# Define image pre-processing steps
# Define the transforms to apply to the training data
augmentation_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=RANDOM_HORIZONTAL_FLIP_PROBABILITY),
    transforms.RandomVerticalFlip(p=RANDOM_VERTICAL_FLIP_PROBABILITY),
    transforms.RandomRotation(degrees=MIN_MAX_ROTATION_RANGE),
    transforms.ToPILImage()
])


def main(year):
    if year == '2018':
        augment_2018_images_and_save_to_file(augmentation_transform)
    elif year == '2019':
        # 2019 has an extra class "UNK" (unknown) - which we remove
        augment_2019_images_and_save_to_file(augmentation_transform)
        
    else:
        raise ValueError("Invalid year. Please use '2018' or '2019'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Augment images for a specific year.')
    parser.add_argument(
        '--year', required=True, choices=['2018', '2019'],
        help='Year for which to augment images (2018 or 2019).')
    args = parser.parse_args()
    main(args.year)


