## Description
This repository contains scripts which allow one to train a model to predict face beauty score and 
create heatmap with the most beautiful parts of the face. <br>
Scripts for web application and Telegram bot are also provided


## Notes
1. Models which use face backbones provide the best explanations results 
(mainly Grad-CAM and also Guided backpropagation/Guided Grad-CAM for Facenet)
2. Guided Grad-CAM (and Guided backpropagation) techniques work with backbones which use Relu activation 
(e.g. Facenet, ResNet). If backbone uses another activation (e.g. Insightface) some tricks are to be
applied to get meaningful results
3. Shap explanations results are very unstable and generally much worse then those obtained with Grad-CAM
4. Training models based on Facenet on images of size 300x300 allows to get 8x8 heatmap conv activation 
in contrast with 3x3 when using default size 160x160 
(5x5 for image size 224x224 which was used by the dataset authors with most of their models)
5. Face_evoLVe_ir50 and Insightface backbones allow to get 7x7 heatmap conv activation
6. Models based on Facenet can process images of any size when using pretrained weights
7. Models based on Face_evoLVe_ir50 and Insightface must operate on images of size 112x112 to use pretrained 
weights
8. Architecture of Insightface model was changed: preprocessing removed from network. One should consider fixing this 
(it has not been changed here as there were some trained models)
9. Explanations results are very unstable when using Adam optimizer for fine-tuning. 
SGD with momentum generally works quite well 
10. Models converge better when using scalar in range (0, 1) as target variable. 
Training with 5 (or even 10) classes might not be a good idea
11. Not all backbones are trained faster with mixed precision. 
Models based on Facenet are recommended to be trained in fp32 and Face_evoLVe_ir50 and Insightface in mixed precision
11. Models trained in mixed precision mode work very slow on CPU, 
so for inference it is strongly recommended to convert them to fp 32 models

## Useful links
1. Dataset with faces: https://github.com/HCIILAB/SCUT-FBP5500-Database-Release
2. Keras face toolbox (code for models architecture and weights are also available): https://github.com/shaoanlu/face_toolbox_keras
3. Tensorflow 2x / Keras Facenet: https://github.com/nyoki-mtl/keras-facenet
4. Transfer learning & fine-tuning in Tensorflow 2x / Keras: https://keras.io/guides/transfer_learning/
5. EfficientNet fine-tuning: https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/image_classification_efficientnet_fine_tuning.ipynb#scrollTo=4BpQqKIeglKl
6. Model interpretation with Grad-CAM in Tensorflow 2x / Keras: https://keras.io/examples/vision/grad_cam/
7. Different techniques for CNN interpretation: https://github.com/utkuozbulak/pytorch-cnn-visualizations
8. Explaining image classifier predictions: https://towardsdatascience.com/knowing-what-and-why-explaining-image-classifier-predictions-680a15043bad
9. Shap for explanation of CNN predictions: https://github.com/slundberg/shap/blob/master/notebooks/kernel_explainer/ImageNet%20VGG16%20Model%20with%20Keras.ipynb
10. Face detection with OpenCV and Dlib in Python and C++: https://www.learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/
11. Face detection in OpenCV with DNN (pretrained models are available): https://github.com/sr6033/face-detection-with-OpenCV-and-DNN
