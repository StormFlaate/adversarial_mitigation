# Import necessary libraries
import argparse
from torchvision import transforms

# Import custom modules
from config import (AUGMENTED_DATASET_2019_LABELS, AUGMENTED_TEST_2018_ROOT_DIR,
                    AUGMENTED_TRAIN_2018_LABELS,
                    AUGMENTED_TRAIN_2018_ROOT_DIR, DATASET_2019_LABELS,
                    DATASET_2019_ROOT_DIR,
                    MIN_MAX_ROTATION_RANGE, MIN_NUMBER_OF_EACH_CLASS_2018,
                    MIN_NUMBER_OF_EACH_CLASS_2019,
                    RANDOM_HORIZONTAL_FLIP_PROBABILITY, 
                    RANDOM_VERTICAL_FLIP_PROBABILITY, 
                    TRAIN_2018_LABELS, TRAIN_2018_ROOT_DIR)
from helper_functions.data_augmentation_helper import augment_images_and_save_to_file



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
        augment_images_and_save_to_file(
            TRAIN_2018_ROOT_DIR,
            AUGMENTED_TRAIN_2018_ROOT_DIR,
            TRAIN_2018_LABELS,
            AUGMENTED_TRAIN_2018_LABELS,
            augmentation_transform, 
            min_number_of_each_class=MIN_NUMBER_OF_EACH_CLASS_2018)
    elif year == '2019':
        # 2019 has an extra class "UNK" (unknown) - which we remove
        augment_images_and_save_to_file(
            DATASET_2019_ROOT_DIR,
            AUGMENTED_TEST_2018_ROOT_DIR,
            DATASET_2019_LABELS,
            AUGMENTED_DATASET_2019_LABELS,
            augmentation_transform, 
            min_number_of_each_class=MIN_NUMBER_OF_EACH_CLASS_2019,
            exclude_last_class=True)
    else:
        raise ValueError("Invalid year. Please use '2018' or '2019'.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Augment images for a specific year.')
    parser.add_argument(
        '--year', required=True, choices=['2018', '2019'],
        help='Year for which to augment images (2018 or 2019).')
    args = parser.parse_args()
    main(args.year)


