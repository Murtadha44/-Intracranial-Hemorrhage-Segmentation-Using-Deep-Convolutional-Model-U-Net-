# -Intracranial-Hemorrhage-Segmentation-Using-Deep-Convolutional-Model-U-Net-
After traumatic brain injury, intracranial hemorrhage (ICH) may occur that could lead to death or disability if it is not accurately diagnosed and treated in a time-sensitive procedure. Currently, Computerized Tomography (CT) scans are examined by radiologists to diagnose ICH and localize its regions. A dataset of 82 CT scans of patients with traumatic brain injury was collected and the ICH regions were segmented. You can find the dataset at https://physionet.org/content/ct-ich/1.3.1/, doi:10.13026/w8q8-ky94. We developed a deep FCN (U-Net) to segment the ICH regions from the CT scans in a fully automated manner. The performance of U-Net in this project is the preliminary proof-of-concept.

The code was updated to work with CT scans in the NIfTI format after updating the dataset (You can find the dataset at https://physionet.org/content/ct-ich/1.3.1/). Please use the following instructions:
To use the code, first download the dataset as a zip file and move it to the same directory of main.py, second run main.py. main.py performs 5-fold cross-validation to train and evaluate U-Net. Prepare_data.py is to 
1-Load the ICH segmentation dataset (zip file) and unzip it to ich_data 
2-Load all CT scans (NIfTI format) and window them using a brain window. 
3-Split them for training, validation and testing folders DataV1\CV0\train ,...\validate ,...\test 
4-Divide each slice into 49 crops using a 128x128 window with a stride of 64. model.py contains the U-Net model, and data_process.py has all the functions required to generate the training and testing data to train and test the model, and also to save the generated masks. 

To run this code, create a Python environment that contains the following libraries (numpy, os, pickle, cv2, nibabel, glob, skimage, keras, tensorflow, sklearn, scipy, pathlib, pandas, zipfile, imageio), then run main.py.
