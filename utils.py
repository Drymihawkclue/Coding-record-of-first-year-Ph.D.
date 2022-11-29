import numpy as np
import os
import random
import imageio
from scipy import misc

def LoadData(num_classes = 50, num_samples_per_class_train = 15, num_samples_per_class_test = 5, seed = 1):
    """
    Load data and split it into training and testing
    Args:
        num_classes: number of classes adopted, -1 represents using all the classes
        num_samples_per_class_train: number of samples per class used for training
        num_samples_per_class_test: number of samples per class used for testing
        seed: random seed to ensure consistent results
    Returns:
        a tuple of (1) images for training (2) labels for training (3) images for testing, and (4) labels for testing
            (1) numpy array of shape [num_classes * num_samples_per_class_train, 784], binary pixels
            (2) numpy array of shape [num_classes * num_samples_per_class_train], integers of the class label
            (3) numpy array of shape [num_classes * num_samples_per_class_test, 784], binary pixels
            (4) numpy array of shape [num_classes * num_samples_per_class_test], integers of the class label
    """
    random.seed(seed)
    np.random.seed(seed)
    num_samples_per_class = num_samples_per_class_train + num_samples_per_class_test
    assert num_classes <= 1623
    assert num_samples_per_class <= 20
    dim_input = 28 * 28   # 784
    
    # construct folders
    data_folder = './omniglot_resized'
    character_folders = [os.path.join(data_folder, family, character)
                         for family in os.listdir(data_folder)
                         if os.path.isdir(os.path.join(data_folder, family))
                         for character in os.listdir(os.path.join(data_folder, family))
                         if os.path.isdir(os.path.join(data_folder, family, character))]
    random.shuffle(character_folders)
    if num_classes == -1:
        num_classes = len(character_folders)
    else:
        character_folders = character_folders[: num_classes]
    
    # read images
    all_images = np.zeros(shape = (num_samples_per_class, num_classes, dim_input))
    all_labels = np.zeros(shape = (num_samples_per_class, num_classes))
    label_images = get_images(character_folders, list(range(num_classes)), nb_samples = num_samples_per_class, shuffle = True)
    temp_count = np.zeros(num_classes, dtype=int)
    for label,imagefile in label_images:
        temp_num = temp_count[label]
        all_images[temp_num, label, :] = image_file_to_array(imagefile, dim_input)
        all_labels[temp_num, label] = label
        temp_count[label] += 1
    
    # split and random permutate
    train_image = all_images[:num_samples_per_class_train].reshape(-1,dim_input)
    test_image  = all_images[num_samples_per_class_train:].reshape(-1,dim_input)
    train_label = all_labels[:num_samples_per_class_train].reshape(-1)
    test_label  = all_labels[num_samples_per_class_train:].reshape(-1)
    train_image, train_label = pair_shuffle(train_image, train_label)
    test_image, test_label = pair_shuffle(test_image, test_label)
    return train_image, train_label, test_image, test_label 
 
 
def get_images(paths, labels, nb_samples=None, shuffle=True):
    """
    Takes a set of character folders and labels and returns paths to image files
    paired with labels.
    Args:
        paths: A list of character folders
        labels: List or numpy array of same length as paths
        nb_samples: Number of images to retrieve per character
    Returns:
        List of (label, image_path) tuples
    """
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images_labels = [(i, os.path.join(path, image))
                     for i, path in zip(labels, paths)
                     for image in sampler([pathstr for pathstr in os.listdir(path) if pathstr[-4:] == '.png' ])]
    if shuffle:
        random.shuffle(images_labels)
    return images_labels


def image_file_to_array(filename, dim_input):
    """
    Takes an image path and returns numpy array
    Args:
        filename: Image filename
        dim_input: Flattened shape of image
    Returns:
        1 channel image
    """
    image = imageio.imread(filename)
    image = image.reshape([dim_input])
    image = image.astype(np.float32) / 255.0
    image = 1.0 - image
    return image

def pair_shuffle(array_a, array_b):
    """
    Takes an image array and a label array
    Returns:
        the shuffled image array and label array
    """
    temp_perm = np.random.permutation(array_a.shape[0])
    array_a = array_a[temp_perm]
    array_b = array_b[temp_perm]
    return array_a, array_b