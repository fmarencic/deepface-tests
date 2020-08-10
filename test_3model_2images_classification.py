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

result1_1  = DeepFace.verify("./anchors/test1_sidro_dokovic.jpg", "./dataset/test1_dokovic.jpg", model_name='Facenet', distance_metric='cosine', model=facenetModel)
result1_2  = DeepFace.verify("./anchors/test1_sidro_dokovic.jpg", "./dataset/test1_dokovic.jpg", model_name='Facenet', distance_metric='euclidean', model=facenetModel)
result1_3  = DeepFace.verify("./anchors/test1_sidro_dokovic.jpg", "./dataset/test1_dokovic.jpg", model_name='Facenet', distance_metric='euclidean_l2', model=facenetModel)

tokFacenet = time.time()

# DeepFace -----------------------------

# starting timer for DeepFace
tikDeepFace = time.time()

result2_1  = DeepFace.verify("./anchors/test1_sidro_dokovic.jpg", "./dataset/test1_dokovic.jpg", model_name='DeepFace', distance_metric='cosine', model=deepfaceModel)
result2_2  = DeepFace.verify("./anchors/test1_sidro_dokovic.jpg", "./dataset/test1_dokovic.jpg", model_name='DeepFace', distance_metric='euclidean', model=deepfaceModel)
result2_3  = DeepFace.verify("./anchors/test1_sidro_dokovic.jpg", "./dataset/test1_dokovic.jpg", model_name='DeepFace', distance_metric='euclidean_l2', model=deepfaceModel)

tokDeepface = time.time()

# DeepID -----------------------------

# starting timer for DeepID
tikDeepID = time.time()

result3_1  = DeepFace.verify("./anchors/test1_sidro_dokovic.jpg", "./dataset/test1_dokovic.jpg", model_name='DeepID', distance_metric='cosine', model=deepidModel)
result3_2  = DeepFace.verify("./anchors/test1_sidro_dokovic.jpg", "./dataset/test1_dokovic.jpg", model_name='DeepID', distance_metric='euclidean', model=deepidModel)
result3_3  = DeepFace.verify("./anchors/test1_sidro_dokovic.jpg", "./dataset/test1_dokovic.jpg", model_name='DeepID', distance_metric='euclidean_l2', model=deepidModel)

tokDeepID = time.time()

print(result1_1, "\n", result1_2, "\n", result1_3, "\n")
print(result2_1, "\n", result2_2, "\n", result2_3, "\n")
print(result3_1, "\n", result3_2, "\n", result3_3, "\n")