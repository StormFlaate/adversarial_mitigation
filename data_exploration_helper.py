import matplotlib.pyplot as plt
import numpy as np
from customDataset import ISICDataset
import torch # Import the Pytorch library
import pandas as pd
from tqdm import tqdm
from typing import Dict

def dataset_overview(
    dataset: ISICDataset, 
    title:str="Default title",
    xlabel:str="",
    ylabel:str="",
    check_image_size:bool=False) -> Dict[str, torch.Size | int | pd.core.indexes.base.Index | np.ndarray]:
    """
    Generates a bar plot that provides an overview of the ISICDataset,
    and provides some key statistics about the dataset.

    The plot shows the number of occurrences of each label in the dataset. 

    Args:
        dataset (ISICDataset): The ISICDataset to be plotted.
        title (str, optional): The title for the plot. Defaults to "Default title".
        xlabel (str, optional): The x-axis label for the plot. Defaults to an empty string.
        ylabel (str, optional): The y-axis label for the plot. Defaults to an empty string.

    Returns:
        None
    """

    # CONSTANTS
    COLORS:tuple = ('navy', 'gold', 'green', 'brown', "lightblue", "yellowgreen", "lightgreen", "blue", "red")
    info: Dict = {}
    
    sizes: set = set()
    
    # will create a set of all the unique sizes/dimensions of the images
    # takes some time to go through all images
    if check_image_size:
        for image, label in tqdm(dataset):
            size = image.size()
            if size not in sizes:
                sizes.add(size)
        
        print(f"Image size:", sizes)

    info["image_sizes"] = sizes
    info["dataset_len"] = len(dataset)
    df = dataset.annotations
    info["dataset_labels"] = df.columns[1:] # exclude the first one which only contains image
    info["label_occurences"] = np.array([0]*len(info["dataset_labels"]))

    # calculate the occurences of the different values
    for index, label in enumerate(info["dataset_labels"] ):
        # count the values for a given category
        count = df[label].value_counts()
        if 1.0 in count.index:
            info["label_occurences"][index] = count[1.0]
    info["label_occurences"]
    
    # plotting
    plt.bar(info["dataset_labels"], info["label_occurences"], color=COLORS)

    # Labels and Titles
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.show()

    print(f"Dataset length:", info["dataset_len"])
    print(f"Dataset labels:", info["dataset_labels"])
    print(f"Dataset occurences:", [*zip(iter(info["dataset_labels"]), iter(info["label_occurences"]))])




    return info