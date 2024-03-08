import os
import uuid


all_entities_path = '../data set 2/'
all_entities_names = os.listdir(all_entities_path)

print("giving temp unique names...")
# First pass to rename all files to a temporary unique name to avoid renaming a file with a name that belongs to another file in the same folder
for entity_name in all_entities_names:
    entity_path = os.path.join(all_entities_path, entity_name)
    for filename in os.listdir(entity_path):
        temp_filename = str(uuid.uuid4()) + ".jpg"  # generate a unique filename
        source = os.path.join(entity_path, filename)
        destination = os.path.join(entity_path, temp_filename)
        os.rename(source, destination)



print("renaming...")
# then rename every file in every folder in the given path

for entity_name in all_entities_names:
    entity_path = os.path.join(all_entities_path, entity_name)
    i = 1
    for filename in os.listdir(entity_path):
        entity_name = entity_name.lower()
        new_filename = entity_name + '.' + str(i) + ".jpg"
        source = os.path.join(entity_path, filename)
        destination = os.path.join(entity_path, new_filename)
        os.rename(source, destination)  
        i += 1

print("done")        


