import os
import random
import shutil

# Define paths to data directory and output directory
import argparse

parser = argparse.ArgumentParser()

# Thêm các đối số cho chương trình
parser.add_argument('--input', type=str, help='Dataset path',default= './data/Images')
parser.add_argument('--output', type=str, help='New dataset path',default='./data/dog_breed')

# Lấy các đối số từ dòng lệnh
args = parser.parse_args()

# Sử dụng các đối số trong chương trình
data_dir = args.input
output_dir = args.output


# Define subdirectories for train, validate, and test
train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'val')
test_dir = os.path.join(output_dir, 'test')

# Define the train, validate, and test ratios
train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2

# Create the train, validate, and test directories if they do not exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Loop through each subdirectory in the data directory
for subdir in os.listdir(data_dir):
    # Define the path to the subdirectory
    subdir_path = os.path.join(data_dir, subdir)

    # Create a list of the files in the subdirectory
    if not os.path.isdir(subdir_path):
        continue
    files = os.listdir(subdir_path)

    # Shuffle the files randomly
    random.shuffle(files)

    # Calculate the number of files for each split
    num_files = len(files)
    num_train = int(train_ratio * num_files)
    num_val = int(val_ratio * num_files)
    num_test = num_files - num_train - num_val

    # Create the train, validate, and test subdirectories for the current class
    train_subdir = os.path.join(train_dir, subdir)
    val_subdir = os.path.join(val_dir, subdir)
    test_subdir = os.path.join(test_dir, subdir)
    os.makedirs(train_subdir, exist_ok=True)
    os.makedirs(val_subdir, exist_ok=True)
    os.makedirs(test_subdir, exist_ok=True)

    # Copy the files to the train, validate, and test subdirectories
    for i, file in enumerate(files):
        src_path = os.path.join(subdir_path, file)
        if i < num_train:
            dst_path = os.path.join(train_subdir, file)
        elif i < num_train + num_val:
            dst_path = os.path.join(val_subdir, file)
        else:
            dst_path = os.path.join(test_subdir, file)

        shutil.copyfile(src_path, dst_path)