import os
num = 0

for dir_name in os.listdir('data/OWDETR/object_crops'):
    source = os.path.join('data/OWDETR/object_crops',dir_name)
    num+=len(os.listdir(source))
    print("{}:have {} samples".format(dir_name,len(os.listdir(source))))