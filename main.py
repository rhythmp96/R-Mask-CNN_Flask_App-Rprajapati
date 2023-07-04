# # https://youtu.be/pI0wQbJwIIs
# """
# For training, watch videos (202 and 203): 
#     https://youtu.be/qB6h5CohLbs
#     https://youtu.be/fyZ9Rxpoz2I

# The 7 classes of skin cancer lesions included in this dataset are:
# Melanocytic nevi (nv)
# Melanoma (mel)
# Benign keratosis-like lesions (bkl)
# Basal cell carcinoma (bcc) 
# Actinic keratoses (akiec)
# Vascular lesions (vas)
# Dermatofibroma (df)

# """



# import numpy as np
# from PIL import Image
# from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras.models import load_model


# def getPrediction(filename):
    
#     classes = ['Actinic keratoses', 'Basal cell carcinoma', 
#                'Benign keratosis-like lesions', 'Dermatofibroma', 'Melanoma', 
#                'Melanocytic nevi', 'Vascular lesions']
#     le = LabelEncoder()
#     le.fit(classes)
#     le.inverse_transform([2])
    
    
#     #Load model
#     my_model=load_model("model/HAM10000_100epochs.h5")
    
#     SIZE = 32 #Resize to same size as training images
#     img_path = 'static/images/'+filename
#     img = np.asarray(Image.open(img_path).resize((SIZE,SIZE)))
    
#     img = img/255.      #Scale pixel values
    
#     img = np.expand_dims(img, axis=0)  #Get it tready as input to the network       
    
#     pred = my_model.predict(img) #Predict                    
    
#     #Convert prediction to class name
#     pred_class = le.inverse_transform([np.argmax(pred)])[0]
#     print("Diagnosis is:", pred_class)
#     return pred_class

###################################################################################################################################################

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import os
from PIL import Image
from mrcnn.config import Config



def getPrediction(filename):
    
    CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush']
    
    class SimpleConfig(mrcnn.config.Config):
        # Give the configuration a recognizable name
        NAME = "coco_inference"
        # set the number of GPUs to use along with the number of images per GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

        # Number of classes = number of classes + 1 (+1 for the background). The background class is named BG
        NUM_CLASSES = len(CLASS_NAMES)
            
    class CocoConfig(Config):
        """Configuration for training on MS COCO.
        Derives from the base Config class and overrides values specific
        to the COCO dataset.
        """
        # Give the configuration a recognizable name
        NAME = "coco"

        # We use a GPU with 12GB memory, which can fit two images.
        # Adjust down if you use a smaller GPU.
        IMAGES_PER_GPU = 1

        # Uncomment to train on 8 GPUs (default is 1)
        # GPU_COUNT = 8

        # Number of classes (including background)
        NUM_CLASSES = 1 + 80  # COCO has 80 classes
        
    class InferenceConfig(CocoConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        ############################### CHANGE HYPERPARAMETERS HERE ##################
        RPN_NMS_THRESHOLD = 0.01
        DETECTION_MIN_CONFIDENCE = 0.7
        DETECTION_NMS_THRESHOLD = 0.7
            
    config = InferenceConfig()
    model = mrcnn.model.MaskRCNN(mode="inference", 
                              config=config,
                              model_dir=os.getcwd())
    
    # Load the weights into the model.
    model.load_weights(filepath="mask_rcnn_coco.h5", 
                    by_name=True)
    
    # load the input image, convert it from BGR to RGB channel
    filelocation = 'static/images/'+filename
    image = cv2.imread(filelocation)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform a forward pass of the network to obtain the results
    r = model.detect([image], verbose=0)
    
    # Get the results for the first image.
    r = r[0]
    
    #Visualize the detected objects.
    image_to_show = mrcnn.visualize.display_instances(image=image, 
                                     boxes=r['rois'], 
                                     masks=r['masks'], 
                                     class_ids=r['class_ids'], 
                                     class_names=CLASS_NAMES, 
                                     scores=r['scores'])
    
    # image_to_show = mrcnn.visualize.display_images(image=image,cols = 4)
    
    return image_to_show
  
# filelocation2 = 'D:\Projects\DL_App\street_tutorial.jpeg'  

# getPrediction(filelocation2)   
    



