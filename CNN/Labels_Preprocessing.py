import cv2
import os.path
import numpy as np
from numpy import genfromtxt


train_labels = genfromtxt('Labels/test_answers.csv', delimiter=',',dtype=np.int32)

# image path and valid extensions
imageDir = "Datasets/Small/"  # specify your path here
image_path_list = []
valid_image_extensions = [".jpg"]  # specify your vald extensions here
valid_image_extensions = [item.lower() for item in valid_image_extensions]

# create a list all files in directory and
# append files with a vaild extention to image_path_list
for file in os.listdir(imageDir):
    extension = os.path.splitext(file)[1]
    if extension.lower() not in valid_image_extensions:
        continue
    image_path_list.append(os.path.join(imageDir, file))

n=0
count=0
a=200
list=[]
# loop through image_path_list to open each image
for imagePath in image_path_list:
    name = imagePath.replace('Datasets/Small/', '')
    name=name[1:4]    #e.g. 001, 043, 232
    for b in range(1,10):    #for removing 0s as prefixes
        name = name.replace('00' + str(b), str(b))
        for c in range(1,10):
            name = name.replace('0' + str(b)+str(c), str(b)+str(c))


    if int (name)>199:   #for testing labels
        count += 1
        if int(name)!=a:    #for next writer
            count=count-1
            a+=1
            for d in range(count-1):

                train_labels=np.insert(train_labels,n,train_labels[n])
            list.append(count)

            n=n+count
            print("n ", a, "count",count)

            count = 1



print(n,len(list))


print("Saving the new labels in a csv file..")
np.savetxt("Labels/test_labels.csv", train_labels, delimiter=",", fmt="%i")
print("Labels have been saved")




