from deepface import DeepFace
import cv2
from matplotlib import pyplot as plt

# uncomment to remove GPU support

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# image list
imagesList = ["./detection/img1.jpg"
    , "./detection/img2.jpg"
    , "./detection/img3.jpg"
    , "./detection/img4.jpg"
    , "./detection/img5.jpg"
    , "./detection/img6.jpg"]

for img in imagesList:
    try:
        img = DeepFace.detectFace(img)
        plt.imshow(img)
        plt.show()
    except:
        print("Lice nije prepoznato.")
        img = cv2.imread(img)
        plt.imshow(img[:,:,::-1])
        plt.suptitle('Lice nije prepoznato.', fontsize=14, fontweight='bold')
        plt.show()