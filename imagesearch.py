__author__ = 'imamanoor'
import cv2
import argparse
import logging
import os
import numpy as np
import csv
from matplotlib import pyplot as plt

class FeatureSet(object):
    def __init__(self, img, name):
        self.kp, self.desc = cv2.SIFT().detectAndCompute(img, None)
        self.name = name
# Given a list of template features ([FeatureSet])
# and one target feature,
# output the best match as a tuple (templateName, perspectiveTransform, numOfMatchingFeatures).
# *Outputs None if no match was found.

def match(template, target, templateName, targetName):

    # Helper function
    def init_SIFT():
        detector = cv2.SIFT()
        norm = cv2.NORM_L2
        matcher = cv2.BFMatcher(norm)
        return detector, matcher

    # Init SIFT feature detector + matcher
    detector, matcher = init_SIFT()

    # Generate (keypoint, descriptor) features for the images
    templateFeatures = FeatureSet(template, templateName)

    #templateKp, templateDesc = detector.detectAndCompute(template, None)
    targetFeatures = FeatureSet(target, targetName)

    # Given the features (keypoints, descriptors), a matcher, and a detector,
    # perform matching with OpenCV.
    confidence = match_features(templateFeatures, targetFeatures, detector, matcher)
    return confidence
def match_sift(templateFeatures, targetFeatures):
    def init_SIFT():
        detector = cv2.SIFT()
        norm = cv2.NORM_L2
        matcher = cv2.BFMatcher(norm)
        return detector, matcher

    # Init SIFT feature detector + matcher
    detector, matcher = init_SIFT()
    confidence = match_features(templateFeatures, targetFeatures, detector, matcher)
    return confidence

def match_features(templateFeatures, targetFeatures, detector, matcher):

    templateKp, templateDesc = (templateFeatures.kp, templateFeatures.desc)
    targetKp, targetDesc = (targetFeatures.kp, targetFeatures.desc)

    # Match image features by k-Nearest-Neighbor algo
    matches = matcher.knnMatch(templateDesc, targetDesc, k = 2)

    # Apply ratio test from D. Lowe
    good = []
    def filter_matches(kp1, kp2, matches, ratio = 0.75):
        mkp1, mkp2 = [], []
        for m in matches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                m = m[0]
                mkp1.append( kp1[m.queryIdx] )
                mkp2.append( kp2[m.trainIdx] )
        p1 = np.float32([kp.pt for kp in mkp1])
        p2 = np.float32([kp.pt for kp in mkp2])
        kp_pairs = zip(mkp1, mkp2)
        return p1, p2, kp_pairs
    p1, p2, good_pairs = filter_matches(templateKp, targetKp, matches)

    return (len(good_pairs))


defaultDetectorType = "SURF"
defaultDescriptorType = "SURF"
defaultMatcherType = "FlannBased"
defaultQueryImageName = "/image_match_challenge/modified_images"
defaultTrainImageDir = '/image_match_challenge/original_images'
defaultDirToSaveResImages = "/results"

parser = argparse.ArgumentParser()
parser.add_argument('--pathTrainImage', type=str, required=True)
parser.add_argument('--pathQueryImage', type=str, required=True)
parser.add_argument('--pathResult', type=str,required=True)
args = parser.parse_args()


logging.debug("Read directory: " + defaultTrainImageDir)
filenames_train = [os.path.join(args.pathTrainImage, f1) for f1 in os.listdir(args.pathTrainImage)]
filenames_query = [os.path.join(args.pathQueryImage, f2) for f2 in os.listdir(args.pathQueryImage)]
NumSampletrain = len(filenames_train)
NumSamplequery = len(filenames_query)
num = NumSampletrain*NumSampletrain
im_train = np.zeros(num,dtype = np.uint8).reshape((NumSampletrain, NumSampletrain))
im_query = np.zeros(num,dtype = np.uint8).reshape((NumSampletrain, NumSampletrain))
conf_matrix = np.zeros((NumSampletrain, NumSamplequery))
idx = np.zeros(NumSamplequery)

sift_descp_train = [ FeatureSet(im_train,0) for i in range(NumSampletrain)]
sift_descp_query = [ FeatureSet(im_train,0) for i in range(NumSamplequery)]

for i in range(0,NumSampletrain):
    filefullpath_train = filenames_train[i]
    im1 = cv2.imread(filefullpath_train, 0)
    sift_descp_train[i] = FeatureSet(im1, filefullpath_train)

for i in range(0, NumSamplequery):
    filefullpath_query = filenames_query[i]
    im2 = cv2.imread(filefullpath_query, 0)
    sift_descp_query[i] = FeatureSet(im2, filefullpath_query)

with open("result_all.csv", "w") as fp:
    file = csv.writer(fp, delimiter=',')
    for i in range(0, NumSampletrain):
        for j in range(0, NumSamplequery):

            conf_matrix[i, j] = match_sift(sift_descp_train[i], sift_descp_query[j])

        best_match_idx = conf_matrix[i,:].argmax()
        matchedpair = filenames_query[best_match_idx] + ',' + filenames_train[i]
        idx[i] = best_match_idx
        print idx[i]
        print matchedpair
        file.writerow(matchedpair)


file.close()
