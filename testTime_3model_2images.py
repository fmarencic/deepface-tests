from deepface import DeepFace
from deepface.basemodels import Facenet, FbDeepFace, DeepID
import time

# uncomment to remove GPU support

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# Loading models
facenetModel = Facenet.loadModel()
deepfaceModel = FbDeepFace.loadModel()
deepidModel = DeepID.loadModel()

# FaceNet -----------------------------

# starting timer for FaceNet
tikFacenet = time.time()

result1  = DeepFace.verify("./anchors/test1_sidro_dokovic.jpg", "./dataset/test1_nadal.jpg", model_name='Facenet', distance_metric='euclidean_l2', model=facenetModel)

tokFacenet = time.time()

# DeepFace -----------------------------

# starting timer for DeepFace
tikDeepFace = time.time()

result2  = DeepFace.verify("./anchors/test1_sidro_dokovic.jpg", "./dataset/test1_nadal.jpg", model_name='DeepFace', distance_metric='euclidean_l2', model=deepfaceModel)

tokDeepface = time.time()

# DeepID -----------------------------

# starting timer for DeepID
tikDeepID = time.time()

result3  = DeepFace.verify("./anchors/test1_sidro_dokovic.jpg", "./dataset/test1_nadal.jpg", model_name='DeepID', distance_metric='euclidean_l2', model=deepidModel)

tokDeepID = time.time()

print(result1, "\n", tokFacenet-tikFacenet, "\n")
print(result2, "\n", tokDeepface-tikDeepFace, "\n")
print(result3, "\n", tokDeepID-tikDeepID, "\n")