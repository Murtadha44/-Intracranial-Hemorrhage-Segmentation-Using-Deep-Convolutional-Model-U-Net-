# -Intracranial-Hemorrhage-Segmentation-Using-Deep-Convolutional-Model-U-Net-
After traumatic brain injury, intracranial hemorrhage (ICH) may occur that could lead to death or disability if it is not accurately diagnosed and treated in a time-sensitive procedure. Currently, Computerized Tomography (CT) scans are examined by radiologists to diagnose ICH and localize its regions. 
A dataset of 82 CT scans of patients with traumatic brain injury was collected and the ICH regions were segmented. You can find the dataset at http://alpha.physionet.org/content/ct-ich/1.0.0/, doi:10.13026/w8q8-ky94. We developed a deep FCN (U-Net) to segment the ICH regions from the CT scans in a fully automated manner. The performance of U-Net in this project is the preliminary proof-of-concept. 

To run this code, create a Python environment that contains the following libraries (numpy, os, pickle, cv2, glob, skimage, keras, tensorflow, sklearn, scipy, pathlib, pandas, urllib, zipfile), then run main.py.
