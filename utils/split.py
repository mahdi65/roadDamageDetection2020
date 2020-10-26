####### splot single folder dataset to 
####### ///val ///train change last line of this file to change output
import os
import random
from shutil import copyfile,copy2

def img_train_test_split(img_source_dir, train_size):
    """
    Randomly splits images over a train and validation folder, while preserving the folder structure
    
    Parameters
    ----------
    img_source_dir : string
        Path to the folder with the images to be split. Can be absolute or relative path   
        
    train_size : float
        Proportion of the original images that need to be copied in the subdirectory in the train folder
    """    
    if not (isinstance(img_source_dir, str)):
        raise AttributeError('img_source_dir must be a string')
        
    if not os.path.exists(img_source_dir):
        raise OSError('img_source_dir does not exist')
        
    if not (isinstance(train_size, float)):
        raise AttributeError('train_size must be a float')
        
    # Set up empty folder structure if not exists
    if not os.path.exists('data'):
        os.makedirs('data')
    else:
        if not os.path.exists('data/train'):
            os.makedirs('data/train')
        if not os.path.exists('data/validation'):
            os.makedirs('data/validation')
            
    # Get the subdirectories in the main image folder
    subdirs = [subdir for subdir in os.listdir(img_source_dir) if os.path.isdir(os.path.join(img_source_dir, subdir))]
    
    for subdir in subdirs:
        subdir_fullpath = os.path.join(img_source_dir, subdir)
        if len(os.listdir(subdir_fullpath)) == 0:
            print(subdir_fullpath + ' is empty')
            break

        train_subdir = os.path.join('data/train', subdir)
        validation_subdir = os.path.join('data/validation', subdir)

        # Create subdirectories in train and validation folders
        if not os.path.exists(train_subdir):
            os.makedirs(train_subdir)

        if not os.path.exists(validation_subdir):
            os.makedirs(validation_subdir)

        train_counter = 0
        validation_counter = 0
        # print(subdir_fullpath)
        # Randomly assign an image to train or validation folder
        for filename in os.listdir(subdir_fullpath):
            # print(filename)
            if filename.endswith(".jpg") or filename.endswith(".png"): 
                fileparts = filename.split('.')
                print(fileparts)
                print("train_subdir",train_subdir)
                print("subdir_fullpath",subdir_fullpath)

                if random.uniform(0, 1) <= train_size:
                    copy2(os.path.join(subdir_fullpath, filename), os.path.join(train_subdir))
                    copy2(os.path.join('/home/fuzzy/bigdata/train_mod/data_mod/xml/', filename[:-3]+'xml'), '/home/fuzzy/bigdata/train_mod/data/train/xml')
                    train_counter += 1
                else:
                    copy2(os.path.join(subdir_fullpath, filename), os.path.join(validation_subdir))
                    copy2(os.path.join('/home/fuzzy/bigdata/train_mod/data_mod/xml/', filename[:-3]+'xml'), "/home/fuzzy/bigdata/train_mod/data/validation/xml")

                    validation_counter += 1
                    
        print('Copied ' + str(train_counter) + ' images to data/train/' + subdir)
        print('Copied ' + str(validation_counter) + ' images to data/validation/' + subdir)

img_train_test_split("/home/fuzzy/bigdata/train_mod/data_mod",0.90)
