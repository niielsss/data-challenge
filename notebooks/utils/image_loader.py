import os
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

class LazyImageLoader:
    def __init__(self, root):
        """
        Initializes the LazyImageLoader object.

        Args:
            root (str): Path to the folder containing images.
        """
        self.filesA, self.filesB = self.get_file_paths(root)
        self.len = min(len(self.filesA), len(self.filesB))
        
    def __len__(self):
        """
        Returns the total number of image pairs.

        Returns:
            int: Number of image pairs.
        """
        return self.len
    
    def __getitem__(self, index):
        """
        Loads and returns the image at the given index of filesA and filesB.

        Args:
            index (int): Index of the image pair to load.

        Returns:
            tuple: A tuple of the loaded images (PIL Image objects).

        Raises:
            IndexError: If the index is out of range.
            ValueError: If the image could not be loaded.
        """
        if index < 0 or index >= self.len:
            raise IndexError("Index out of range.")

        try:
            imageA = Image.open(self.filesA[index]).convert('RGB')
            imageB = Image.open(self.filesB[index]).convert('RGB')
        except Exception as e:
            raise ValueError(f"Image could not be loaded: {e}")

        return imageA, imageB
    
    def get_file_paths(self, root):
        """
        Recursively retrieves file paths from 'trainA' and 'trainB' subdirectories.

        Args:
            root (str): The root directory of the EUVP dataset.

        Returns:
            tuple: Two lists, filesA and filesB, containing the file paths.
        """
        sub_dirs = ['underwater_imagenet', 'underwater_dark', 'underwater_scenes']
        filesA, filesB = [], []

        for sub_dir in sub_dirs:
            sub_dir_path = os.path.join(root, sub_dir)
            if os.path.exists(sub_dir_path):
                for dirpath, _, filenames in os.walk(sub_dir_path):
                    if 'trainA' in dirpath:
                        for filename in filenames:
                            filesA.append(os.path.join(dirpath, filename))
                    elif 'trainB' in dirpath:
                        for filename in filenames:
                            filesB.append(os.path.join(dirpath, filename))
            else:
                print(f"Warning: Subdirectory '{sub_dir}' does not exist in the root path.")

        return filesA, filesB
    
    def display(self, index):
        imageA, imageB = self[index]
        
        filenameA = os.path.basename(self.filesA[index])
        filenameB = os.path.basename(self.filesB[index])
        
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].imshow(imageA)
        axes[0].set_title(f"Image A - {filenameA}")
        axes[0].axis('off')

        axes[1].imshow(imageB)
        axes[1].set_title(f"Image B - {filenameB}")
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()