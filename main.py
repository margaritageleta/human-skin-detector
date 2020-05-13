from SkinDetector import SkinDetector
import matplotlib.pyplot as plt
from skimage import color
import numpy as np
from PIL import Image

# This function segments the validation dataset.
def segmentation_experiments():
    sd = SkinDetector(hist_range = (3,97), dilate = 2)
    sd.train()

    segmented = sd.segment_dataset(sd.VD_DATA)
    
    # Uncomment this to visualize the segmented images.
    
    """
    for idx in range(0, len(segmented)):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(sd.VD_DATA[idx])
        ax[1].imshow(segmented[idx], cmap = plt.get_cmap('gray'))
        plt.show()
    """
    
    sd.validate(segmented, sd.VD_MASK)

    # Uncomment this to obtain the median of the segmentation.
    """
    print('\nMedian')
    
    print(f'{np.round(np.median(sd.accuracy), 3) * 100}%')
    print(f'{np.round(np.median(sd.precision), 3) * 100}%')
    print(f'{np.round(np.median(sd.recall),  3) * 100}%')
    print(f'{np.round(np.median(sd.F1score), 3) * 100}%')
    """

    # Uncomment this to obtain the mean of the segmentation.
    
    print('\nMean')
    
    print(f'{np.round(np.mean(sd.accuracy) * 100, 3)}%')
    print(f'{np.round(np.mean(sd.precision) * 100, 3)}%')
    print(f'{np.round(np.mean(sd.recall) * 100, 3)}%')
    print(f'{np.round(np.mean(sd.F1score) * 100, 3)}%')

    print(f'{np.std(sd.accuracy)}%')
    print(f'{np.std(sd.precision)}%')
    print(f'{np.std(sd.recall)}%')
    print(f'{np.std(sd.F1score)}%')
    

    print('\n')

# This function segments an image you want.
# Note: change the path of the image.
def segment_your_image():
    sd = SkinDetector(hist_range=(3, 97), dilate=2)
    sd.train()
    basewidth = 300
    img = Image.open(f'../../../../Desktop/mano5.jpg')
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    segmented = sd.segment(np.asarray(img), plot=True)

# This function makes all the assessment to find the optimal range
# yields the plot from the paper
# and stores the data in a dictionary in file data.txt.
def assessment_paper():
    TRmedianAcc = []
    TRmedianPre = []
    TRmedianRec = []
    TRmedianF1 = []

    TRmeanAcc = []
    TRmeanPre = []
    TRmeanRec = []
    TRmeanF1 = []

    TRstdAcc = []
    TRstdPre = []
    TRstdRec = []
    TRstdF1 = []

    VDmedianAcc = []
    VDmedianPre = []
    VDmedianRec = []
    VDmedianF1 = []

    VDmeanAcc = []
    VDmeanPre = []
    VDmeanRec = []
    VDmeanF1 = []

    VDstdAcc = []
    VDstdPre = []
    VDstdRec = []
    VDstdF1 = []

    for hist_range in zip(np.arange(0, 10, 0.1), 100 -np.arange(0, 10, 0.1)): 
        sd = SkinDetector(hist_range = hist_range, dilate = 0)
        sd.train()
        print(f"Trained {sd}!")
        segmentedTrain = sd.segment_dataset(sd.TR_DATA)
        sd.validate(segmentedTrain, sd.TR_MASK)

        TRmedianAcc.append(np.median(sd.accuracy))
        TRmedianPre.append(np.median(sd.precision))
        TRmedianRec.append(np.median(sd.recall))
        TRmedianF1.append(np.median(sd.F1score))

        TRmeanAcc.append(np.mean(sd.accuracy))
        TRmeanPre.append(np.mean(sd.precision))
        TRmeanRec.append(np.mean(sd.recall))
        TRmeanF1.append(np.mean(sd.F1score))

        TRstdAcc.append(np.std(sd.accuracy))
        TRstdPre.append(np.std(sd.precision))
        TRstdRec.append(np.std(sd.recall))
        TRstdF1.append(np.std(sd.F1score))

        segmentedValid = sd.segment_dataset(sd.VD_DATA)
        sd.validate(segmentedValid, sd.VD_MASK)

        VDmedianAcc.append(np.median(sd.accuracy))
        VDmedianPre.append(np.median(sd.precision))
        VDmedianRec.append(np.median(sd.recall))
        VDmedianF1.append(np.median(sd.F1score))

        VDmeanAcc.append(np.mean(sd.accuracy))
        VDmeanPre.append(np.mean(sd.precision))
        VDmeanRec.append(np.mean(sd.recall))
        VDmeanF1.append(np.mean(sd.F1score))

        VDstdAcc.append(np.std(sd.accuracy))
        VDstdPre.append(np.std(sd.precision))
        VDstdRec.append(np.std(sd.recall))
        VDstdF1.append(np.std(sd.F1score))

    data = {
        'TR': {
            'median': {
                'accuracy': TRmedianAcc,
                'precision': TRmedianPre,
                'recall': TRmedianRec,
                'F1': TRmedianF1
            },
            'mean': {
                'accuracy': TRmeanAcc,
                'precision': TRmeanPre,
                'recall': TRmeanRec,
                'F1': TRmeanF1
            },
            'std': {
                'accuracy': TRstdAcc,
                'precision': TRstdPre,
                'recall': TRstdRec,
                'F1': TRstdF1
            }
        },
        'VD': {
            'median': {
                'accuracy': VDmedianAcc,
                'precision': VDmedianPre,
                'recall': VDmedianRec,
                'F1': VDmedianF1
            },
            'mean': {
                'accuracy': VDmeanAcc,
                'precision': VDmeanPre,
                'recall': VDmeanRec,
                'F1': VDmeanF1
            },
            'std': {
                'accuracy': VDstdAcc,
                'precision': VDstdPre,
                'recall': VDstdRec,
                'F1': VDstdF1
            }
        }
    }

    with open('data.txt', 'w') as f:
        f.write(str(data))

# Experiments with reconstruction operators
def reconstruction_experiments():
    from skimage.morphology import reconstruction
    from skimage import exposure

    sd = SkinDetector(hist_range=(3, 97), dilate=0)
    sd.train()
    basewidth = 300
    img = Image.open(f'../../../../Desktop/caras.jpg')
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = np.asarray(img.resize((basewidth,hsize), Image.ANTIALIAS))

    from skimage import data
    img = data.astronaut()
    segmented = sd.segment(np.asarray(img), plot=True)

    better_contrast =  exposure.equalize_adapthist(color.rgb2gray(img), clip_limit=0.001)

    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    ax[0].imshow(color.rgb2gray(img), cmap = plt.get_cmap('gray'))
    ax[1].imshow(better_contrast, cmap = plt.get_cmap('gray'))
    plt.show()
    

    maxval = np.max(color.rgb2gray(better_contrast))
    minval = np.min(color.rgb2gray(better_contrast))
    #print("Max=",maxval,", Min=",minval)
    marker = np.copy(better_contrast)
    # Intensity of seed image must be greater than that of the mask image for reconstruction by erosion.
    print("Before ", len(np.where(marker <= color.rgb2gray(better_contrast)[0])))
    marker[marker <= minval] = maxval
    print("After ", len(np.where(marker <= color.rgb2gray(better_contrast)[0])))

    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    ax[0].imshow(segmented, cmap = plt.get_cmap('gray'))
    ax[1].imshow(marker, cmap = plt.get_cmap('gray'))
    plt.show()

    print(len(np.where(marker > color.rgb2gray(np.asarray(img)))[0]))


    rec = reconstruction(marker, better_contrast, method='erosion')

    fig, ax = plt.subplots(1, 3, figsize=(10,5))
    ax[0].imshow(segmented, cmap = plt.get_cmap('gray'))
    ax[1].imshow(rec, cmap = plt.get_cmap('gray'))
    ax[2].imshow(better_contrast, cmap = plt.get_cmap('gray'))
    plt.show()


if __name__ == "__main__":

    segmentation_experiments()

    



    


