from deepface import DeepFace
from deepface.basemodels import Facenet, FbDeepFace, DeepID
import time

# uncomment to remove GPU support

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# dataset path
DATASET_PATH = "./dataset"

# Loading models
facenetModel = Facenet.loadModel()
deepfaceModel = FbDeepFace.loadModel()
deepidModel = DeepID.loadModel()

# FaceNet -----------------------------

# starting timer for FaceNet
tikFacenet = time.time()

result1 = DeepFace.find(img_path = "./anchors/test1_sidro_wozniacki.jpg"
    , db_path = DATASET_PATH
    , model_name='Facenet'
    , distance_metric='euclidean_l2'
    , model=facenetModel)

tokFacenet = time.time()

# DeepFace -----------------------------

# starting timer for DeepFace
tikDeepFace = time.time()

result2 = DeepFace.find(img_path = "./anchors/test1_sidro_wozniacki.jpg"
    , db_path = DATASET_PATH
    , model_name='DeepFace'
    , distance_metric='euclidean_l2'
    , model=deepfaceModel)

tokDeepface = time.time()

# DeepID -----------------------------

# starting timer for DeepID
tikDeepID = time.time()

result3 = DeepFace.find(img_path = "./anchors/test1_sidro_wozniacki.jpg"
    , db_path = DATASET_PATH
    , model_name='DeepID'
    , distance_metric='euclidean_l2'
    , model=deepidModel)

tokDeepID = time.time()

# results
print("\n")
print(result1, "\nVrijeme potrebno: ", tokFacenet-tikFacenet, "\n")
print(result2, "\nVrijeme potrebno: ", tokDeepface-tikDeepFace, "\n")
print(result3, "\nVrijeme potrebno: ", tokDeepID-tikDeepID, "\n")