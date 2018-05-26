import cv2
import os.path
import numpy as np
from numpy import genfromtxt

def split(arr,name, m, n):

    count = 0  # for keeping count of splitting of a image
    rows = arr.shape[0]  # width & height of the image i.e. no. of pixels
    cols = arr.shape[1]
    c = 0
    d = 0
    filter_col = n
    filter_row = m
    for a in range(0, (rows * cols)):
        if filter_col > cols:  # for last column pixel, move the window to the next row
            d += m
            filter_row += m
            c = 0
            filter_col = n
        if filter_row > rows:  # after last row limit of window, exit
            break
        b = arr[int(d):int(filter_row), int(c):int(filter_col)]
        c += n  # for moving the window ahead one pixel at a time (same for rows also)
        filter_col += n
        if 0 in b:     #to remove all 255(i.e. white) subimages
            name = name.replace('Datasets/Big/','')
            name = name.replace('.jpg', '')
            b=np.reshape(b, b.shape + (1,))     #converting them back to 3d
            cv2.imwrite("Datasets/Small/"+name+"_"+str(count)+".jpg",b)
            count += 1
        else:
            continue
    return count

def csvhandeling(labels,n1,n2):       #to the labels for all newly created 32x32 images (n1 being the no. of 32x32 images for one document, n2 being the location where to start adding the labels)
    doc_label= labels[n2]   #extracting label for the whole document
    for a in range(n1):
        labels=np.insert(labels,n2,doc_label)
    return labels

train_labels = genfromtxt('Labels/labels.csv', delimiter=',',dtype=np.int32)

# image path and valid extensions
imageDir = "Datasets/Big/"  # specify your path here
image_path_list = []
valid_image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]  # specify your vald extensions here
valid_image_extensions = [item.lower() for item in valid_image_extensions]

# create a list all files in directory and
# append files with a vaild extention to image_path_list
for file in os.listdir(imageDir):
    extension = os.path.splitext(file)[1]
    if extension.lower() not in valid_image_extensions:
        continue
    image_path_list.append(os.path.join(imageDir, file))

count=0
# loop through image_path_list to open each image
for imagePath in image_path_list:
    image = cv2.imread(imagePath,0)
    ret, thresh1 = cv2.threshold(image, 230, 255, cv2.THRESH_BINARY)
    thresh1 = cv2.medianBlur(thresh1, 7)
    # find where the signature is and make a cropped region
    points = np.argwhere(thresh1 == 0)  # find where the black pixels are
    points = np.fliplr(points)  # store them in x,y coordinates instead of row,col indices
    x = min(points[:, 0])   #finding the first & last of black pixels of both axes
    y = min(points[:, 1])
    w = max(points[:, 0])
    h = max(points[:, 1])
    crop = image[y:h, x:w]  # create a cropped region of the gray image
    # get the thresholded crop
    retval, crop = cv2.threshold(crop, thresh=200, maxval=255, type=cv2.THRESH_BINARY)
    print("Saving processed 32x32 images for the document ", imagePath)
    n=split(crop,imagePath, 32, 32)
    print("n",n)
    count=count+n              #taking 0th image into consideration
    # exit when escape key is pressed
    key = cv2.waitKey(0)
    if key == 27:  # escape
        break
print("Total images ",count)
train_labels=np.reshape(train_labels,(-1,1))

# close any open windows
cv2.destroyAllWindows()