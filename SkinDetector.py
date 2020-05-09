from skimage import color
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import selem, dilation

class SkinDetector:
    """ Skin Detector using yCbCr colorspace. """
    # Initialize SkinDetector.
    def __init__(self, hist_range = (1, 99), dilate = 0):
        self.load_data('./Training-Dataset', './Validation-Dataset')
        self.lower_percentile = hist_range[0]
        self.upper_percentile = hist_range[1]
        if dilate > 0:
            self.dilate = True
            self.dilation_size = dilate
        else:
            self.dilate = False

    # Method for reading Training and Validation data.
    def load_data(self, TR_path, VD_path):
        self.TR_DATA = []
        self.VD_DATA = []

        self.TR_MASK = []
        self.VD_MASK = []

        self.TR_LABEL = []
        self.VD_LABEL = []

        for shot in range(1,4): 
            for person in range(1,9): 
                for finger in range(1,6):
                    try:
                        self.TR_DATA.append(np.array(Image.open(f'{TR_path}/Images/{finger}_P_hgr1_id0{person}_{shot}.jpg')))
                        self.TR_LABEL.append(finger)
                    except IOError: pass
                    try:
                        self.TR_MASK.append(np.array(Image.open(f'{TR_path}/Masks-Ideal/{finger}_P_hgr1_id0{person}_{shot}.bmp')))
                    except IOError: pass
                    try:
                        self.VD_DATA.append(np.array(Image.open(f'{VD_path}/Images/{finger}_P_hgr1_id0{person}_{shot}.jpg')))     
                        self.VD_LABEL.append(finger)
                    except IOError: pass
                    try:
                        self.VD_MASK.append(np.array(Image.open(f'{VD_path}/Masks-Ideal/{finger}_P_hgr1_id0{person}_{shot}.bmp')))
                    except IOError: pass
    
    # Method for adjusting the chroma thresholds according to training data.
    def train(self):
        ONLY_IMAGE_MASK = []
        for id in range(0, len(self.TR_DATA)):
            masked_img = self.TR_DATA[id].copy()
            masked_img[self.TR_MASK[id] == True] = 0
            ONLY_IMAGE_MASK.append(color.rgb2ycbcr(masked_img)[self.TR_MASK[id] != True])

        # Extract chroma.
        CHROMA_R = CHROMA_B = np.array([])
        for id in ONLY_IMAGE_MASK:
            CHROMA_R = np.hstack((CHROMA_R, id[:,1]))
            CHROMA_B = np.hstack((CHROMA_B, id[:,2]))
        
        # Define thresholds.
        self.LOWER_THRESHOLD_R = np.percentile(CHROMA_R, self.lower_percentile)
        self.MEDIAN_R = np.percentile(CHROMA_R, 50)
        self.UPPER_THRESHOLD_R = np.percentile(CHROMA_R, self.upper_percentile)

        self.LOWER_THRESHOLD_B = np.percentile(CHROMA_B, self.lower_percentile)
        self.MEDIAN_B = np.percentile(CHROMA_B, 50)
        self.UPPER_THRESHOLD_B = np.percentile(CHROMA_B, self.upper_percentile)

    # Method for segmenting an image.
    # It allows using a dilation parameter.
    # Set plot = True to visualize the results.
    def segment (self, img, plot = False):
        #print("Segmenting shape ", np.asarray(img).shape)

        new_mask = np.zeros(shape = color.rgb2ycbcr(img).shape)
        img_transformed = color.rgb2ycbcr(img)
        new_mask[(img_transformed[:,:,1] > self.LOWER_THRESHOLD_R) & (img_transformed[:,:,1] < self.UPPER_THRESHOLD_R) & (img_transformed[:,:,2] > self.LOWER_THRESHOLD_B) &(img_transformed[:,:,2] < self.UPPER_THRESHOLD_B)] = 1

        if self.dilate:
            structuring_elem = selem.disk(self.dilation_size)
            new_mask_dilated = dilation(color.rgb2gray(new_mask), structuring_elem)
            if plot: 
                fig, ax = plt.subplots(1, 3, figsize = (10,5))
                ax[0].imshow(img)
                ax[1].imshow(new_mask)
                ax[2].imshow(new_mask_dilated, cmap = plt.get_cmap('gray'))
                plt.show()

            return color.rgb2gray(new_mask_dilated)

        if plot:
            fig, ax = plt.subplots(1, 2, figsize = (10,5))
            ax[0].imshow(img)
            ax[1].imshow(new_mask)
            plt.show()   
        
        return color.rgb2gray(new_mask)
    
    # Assess the obtained prediction with four metrics.
    def assess (self, prediction, truth):
        prediction = np.array(prediction, dtype=bool)
        truth = np.invert(truth)
        TOTAL = np.zeros(shape=(truth.shape[0],truth.shape[1]))

        TP = TOTAL.copy()
        TN = TOTAL.copy()
        FP = TOTAL.copy()
        FN = TOTAL.copy()

        # True Positive = intersection of truth and prediction.
        TP[np.logical_and(prediction, truth)] = 1 
        # False Positive = intersection of true background and prediction, where true background = inverse truth.
        FP[np.logical_and(prediction, np.invert(truth))] = 1
        # False Negative = intersection of truth and prediction background.
        FN[np.logical_and(np.invert(prediction), truth)] = 1
        # True Negative = intersection of true background and prediction background.
        TN[np.logical_and(np.invert(prediction), np.invert(truth))] = 1

        accuracy = (np.sum(np.abs(TP)) + np.sum(np.abs(TN))) / (np.sum(np.abs(TP)) + np.sum(np.abs(TN)) + np.sum(np.abs(FP)) + np.sum(np.abs(FN)))
        precision = (np.sum(np.abs(TP))) / (np.sum(np.abs(TP)) + np.sum(np.abs(FP)))
        recall = (np.sum(np.abs(TP))) / (np.sum(np.abs(TP)) + np.sum(np.abs(FN)))
        F1score = (np.sum(np.abs(TP)) * 2) / (2 * np.sum(np.abs(TP)) + np.sum(np.abs(FP)) + np.sum(np.abs(FN)))

        return (accuracy, precision, recall, F1score)

    # Validate the Validation set.
    def validate (self, prediction_set, truth_set):
        self.accuracy, self.precision, self.recall, self.F1score = [], [], [], []
        for (prediction, truth) in zip(prediction_set, truth_set):
            metrics = self.assess(color.rgb2gray(prediction), truth)
            self.accuracy.append(metrics[0])
            self.precision.append(metrics[1])
            self.recall.append(metrics[2])
            self.F1score.append(metrics[3])

    
    def segment_dataset(self, data):
        datanpy = np.asarray(data)
        for i in range(0, datanpy.shape[0]):
            datanpy[i] = self.segment(datanpy[i])
        return datanpy
    