# ImageMatching
an image matching algorithm for Cogniac image database

instructions to run:

The script takes in arguments to read paths for modified and original images as --pathQueryImage and --pathTrainImage

The core concept used in this algorithm is template base image matching using SIFT descriptors. 
The program loops over all original images and calculate the distance from each modified image in 
SIFT feature domain. Keypoints are detected using harris corner detection algorithm on the image and SIFT 
descriptor is calculated for each keypoint. All keypoints from each image are matched against the target image 
using k-nearestneighbour classifier based on euclidean distance. The matches returned from classifier are pruned 
to filter out matches with ratio test suggested by Lowe in SIFT. A confusion matrix is created with dimensions 
number of orignal images to number of modified images and is pouplated by the number of good matches found for each 
original image againt all modified images. The index against maximum value is picked as a match for original image 
in the modified images. SIFT descriptors are robust to rotation, scale and illumination changes, but fail to 
discriminate in presence of noise and blur. Also due to local similarity of some images, when matching the SIFT feature
points, some false matches occur. The ranking algorithm is based on number of good matches, which may induce quantization
error.

3rd party Libs:

The algorithm is implemented using:
opencv library for calculating descriptors and matching. 
Numpy is used for vector manipulation,
argparse is used for argument parsing,
matplotlib is used for plotting and visualizations

Future work:
SIFT descriptors are robust to rotation, scale and illumination changes, but fail to discriminate 
in presence of noise and blur. Also due to local similarity of some images, when matching the SIFT 
feature points, some false matches occur. Given more time I will experiment with various fast feature 
detection algorithms and matching algorithms to improve the accuracy and speed of the algorithm. 

For Accuracy:
Explore ranking algorithm criteria based on distances of each feature descriptor
Explore geometric similarity along with SIFT
Explore combination of different descriptors to form more robust discriminative feature descriptors.
Experiment with hyperparameter and optimize for best performance.

For Time:
Explore faster matching algorithms like decision trees
Explore faster feature descriptors like FERNs, FAST, BRIEF



