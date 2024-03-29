import os
from typing import Any
import pandas as pd
import torch
from PIL import Image
from torchvision.io import read_image
from tqdm import tqdm

from config import RANDOM_SEED
# Import custom modules
from config import (
    AUGMENTED_DATASET_2019_LABELS,
    AUGMENTED_DATASET_2019_ROOT_DIR,
    AUGMENTED_TRAIN_2018_LABELS,
    AUGMENTED_TRAIN_2018_ROOT_DIR,
    DATASET_2019_LABELS,
    DATASET_2019_ROOT_DIR,
    MIN_NUMBER_OF_EACH_CLASS_2018,
    MIN_NUMBER_OF_EACH_CLASS_2019,
    TRAIN_2018_LABELS, TRAIN_2018_ROOT_DIR
)


def augment_2018_images_and_save_to_file(augmentation_transform):
    _augment_images_and_save_to_file(
        TRAIN_2018_ROOT_DIR,
        AUGMENTED_TRAIN_2018_ROOT_DIR,
        TRAIN_2018_LABELS,
        AUGMENTED_TRAIN_2018_LABELS,
        augmentation_transform, 
        min_number_of_each_class=MIN_NUMBER_OF_EACH_CLASS_2018
    )


def augment_2019_images_and_save_to_file(augmentation_transform):
    _augment_images_and_save_to_file(
        DATASET_2019_ROOT_DIR,
        AUGMENTED_DATASET_2019_ROOT_DIR,
        DATASET_2019_LABELS,
        AUGMENTED_DATASET_2019_LABELS,
        augmentation_transform, 
        min_number_of_each_class=MIN_NUMBER_OF_EACH_CLASS_2019,
        exclude_last_class=True
    )


###################################################
# ============== PRIVATE FUNCTIONS ============== #
###################################################

def _augment_images_and_save_to_file(
        root_dir: str,
        new_root_dir: str,
        csv_file: str,
        new_csv_file: str,
        transform,
        min_number_of_each_class: int = 1000,
        image_file_type: str = "jpg",
        random_seed: int = RANDOM_SEED,
        exclude_last_class: bool = False):
    """
    Load images and annotations from a CSV file, augment the images using a given
    transform, and save the new images and annotations to a new CSV file.

    Args:
        root_dir: The root directory where the original images are stored.
        new_root_dir: The root directory where the new images will be saved.
        csv_file: Path to CSV file containing the annotations for the original images.
        new_csv_file: The name of the new CSV file to be saved.
        transform: A transform to be applied to the images.
        min_number_of_each_class: Minimum number of images of each class.
        image_file_type: The file type of the images.
        random_seed: A random seed to be used for reproducibility.
        exclude_last_class: Whether to exclude the last class from the annotations.
    """

    # Make the new root directory if it doesn't exist
    if not os.path.exists(new_root_dir):
        print(f"Creating {new_root_dir}")
        os.makedirs(new_root_dir)

    # Set the random seed
    torch.manual_seed(random_seed)

    # Load the annotations from the CSV file
    annotations = pd.read_csv(csv_file)

    # Get the names of all the different classes
    if exclude_last_class:
        class_names = annotations.columns[1:-1]
    else:
        class_names = annotations.columns[1:]
    
    images = annotations.iloc[:, 0]

    # Create a dictionary mapping each class name to a list of image names
    class_images: dict[str, Any] = _create_class_images_dict(annotations, class_names)

    # Define the column names for the new CSV file
    columns = ["image"] + list(class_names)

    # Instantiate a new DataFrame for the new CSV file
    df = _instantiate_skinlesion_dataframe(columns)

    # Loop through each class and display transformed images in the subplots
    total_image_counter: int = 1
    for class_name, (class_index, images) in class_images.items():
        # max of the minimum required images and total images for specific class
        for img_num in tqdm(range(max(min_number_of_each_class, len(images)))):
            # Choose an image from the class images
            image_name = images[img_num % len(images)]
            img_path = f"{os.path.join(root_dir, image_name)}.{image_file_type}"
            # Read in the image with torchvision.io read_image function
            image = read_image(img_path)

            # Apply the transform to the image
            image_variant: Image.Image = transform(image)

            # Generate a new padded image name and add it to the DataFrame with the
            # appropriate class label
            new_image_name = f"ISIC_{str(total_image_counter).zfill(8)}"
            full_image_path = os.path.join(
                new_root_dir,
                f"{new_image_name}.{image_file_type}"
            )

            image_variant.save(full_image_path)

            df.loc[len(df)] = [new_image_name] + _generate_list_with_1_at_index(
                len(class_names), class_index
            )

            total_image_counter += 1

    # Save the new DataFrame to a new CSV file
    df.to_csv(new_csv_file, index=False)

def _instantiate_skinlesion_dataframe(columns: list[str]) -> pd.DataFrame:
    """
    Create a new DataFrame with columns and default value of 0.0 for each column except
    the first one.

    Args:
        columns: A list of strings representing the names of the columns.

    Returns:
        A new DataFrame with columns and default value of 0.0 for each column except the
        first one. The first one will contain the image-name
    """
    df = pd.DataFrame(columns=columns)
    # Set the default value of each column to 0.0
    for col in columns[1:]:
        df[col] = 0.0

    return df


def _generate_list_with_1_at_index(length, index):
    """
    Generate a list of zeros with a single 1 at the specified index.

    Args:
        length: An integer representing the length of the list.
        index: An integer representing the index where the value should be 1.

    Returns:
        A list of zeros with a single 1 at the specified index.
    """
    # Create a list of the desired length filled with 0's
    lst = [0.0] * length
    # Set the element at the specified index to 1
    lst[index] = 1.0
    return lst


def _create_class_images_dict(annotations, class_names):
    """
    Create a dictionary of class names and their corresponding images.

    Args:
        annotations: A Pandas DataFrame representing the annotations.
        class_names: A list of strings representing the names of the classes.

    Returns:
        A dictionary of class names and their corresponding images.
    """
    class_images = {}
    for class_name in class_names:
        class_images[class_name] = []

    for index, key in enumerate(class_images.keys()):
        class_rows = annotations[annotations[key] == 1.0]
        image_list = class_rows['image'].tolist()
        class_images[key] = (index, image_list)

    return class_images

