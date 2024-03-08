import os
import shutil
from sklearn.model_selection import train_test_split

# Define the source directory and the target directories
source_dir = './data set/all/'
train_dir = './data set/train/'
test_dir = './data set/test/'

# Create the target directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Get a list of all the dog and cat image filenames
dog_images = [i for i in os.listdir(source_dir) if 'dog' in i]
cat_images = [i for i in os.listdir(source_dir) if 'cat' in i]

# Split the filenames into training and testing sets
train_dogs, test_dogs = train_test_split(dog_images, test_size=0.25)
train_cats, test_cats = train_test_split(cat_images, test_size=0.25)

# Function to move files
def move_files(files, target_dir):
    for file in files:
        shutil.move(source_dir + file, target_dir + file)

# Move the corresponding files into the appropriate directories
move_files(train_dogs, train_dir)
move_files(test_dogs, test_dir)
move_files(train_cats, train_dir)
move_files(test_cats, test_dir)
