import os
import cv2
import numpy as np



def calculate_and_append_density(image_dir):
    """
    Reads images from a directory, calculates their density, and appends the density to the file name.

    Args:
        image_dir (str): Path to the directory containing images.
    """
    if not os.path.isdir(image_dir):
        print(f"Error: {image_dir} is not a valid directory.")
        return

    for file_name in os.listdir(image_dir):
        file_path = os.path.join(image_dir, file_name)

        # Skip if it's not a file
        if not os.path.isfile(file_path):
            continue

        # Read the image
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Warning: Unable to read {file_name}. Skipping.")
            continue

        # Calculate density (non-zero pixels / total pixels)
        total_pixels = image.size
        non_zero_pixels = np.count_nonzero(image)
        density = non_zero_pixels / total_pixels
        density_rounded = round(density, 3)

        # Create new file name
        name, ext = os.path.splitext(file_name)
        new_file_name = f"{name}_density_{density_rounded}{ext}"
        new_file_path = os.path.join(image_dir, new_file_name)

        # Rename the file
        os.rename(file_path, new_file_path)
        print(f"Renamed {file_name} to {new_file_name}")


calculate_and_append_density("../InpaintingSolver/ch5/5.7/double/masks")
