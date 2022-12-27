from tensorflow.keras.preprocessing import image_dataset_from_directory
from quickdraw import QuickDrawDataGroup, QuickDrawData
from pathlib import Path
import sys
import os
import numpy as np
dirname = os.path.dirname(__file__)

image_size = (28, 28)

def generate_class_images(name, max_drawings, recognized):
    directory = os.path.join(dirname, '../../datasets/drawings/' + name) # Path("quickdraw/" + name)

    # Check if the directory exists
    if not os.path.exists(directory):
        # Create the directory if it does not exist
        os.makedirs(directory)

    images = QuickDrawDataGroup(name, max_drawings=max_drawings, recognized=recognized)
    for img in images.drawings:
        filename = directory + "/" + str(img.key_id) + ".png"
        img.get_image(stroke_width=3).resize(image_size).save(filename)

def load_drawings_dataset():

    for label in QuickDrawData().drawing_names:
        generate_class_images(label, max_drawings=1, recognized=True)

    batch_size = 32
    
    # TODO: modify this, we don't want yet to split training and validation
    train_ds = image_dataset_from_directory(
        "datasets/drawings/",
        validation_split=0.2,
        subset="training",
        seed=123,
        color_mode="grayscale",
        image_size=image_size,
        batch_size=batch_size
    )

    val_ds = image_dataset_from_directory(
        "datasets/drawings/",
        validation_split=0.2,
        subset="validation",
        seed=123,
        color_mode="grayscale",
        image_size=image_size,
        batch_size=batch_size
    )

    print(train_ds, val_ds)

    return (train_ds, val_ds)

if __name__ == '__main__':
    globals()[sys.argv[1]]()