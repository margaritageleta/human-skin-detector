from SkinDetector import SkinDetector
import matplotlib.pyplot as plt
from skimage import color
import numpy as np
from PIL import Image

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


if __name__ == "__main__":

    """
    for hist_range in zip(np.arange(0, 10, 0.1), 100 -np.arange(0, 10, 0.1)): 
        sd = SkinDetector(hist_range = hist_range, dilate = 0)

    with open('data.txt', 'w') as f:
        f.write()
    """

    """
    sd = SkinDetector(hist_range=(3, 97), dilate=2)
    sd.train()
    basewidth = 300
    img = Image.open(f'../../../../Desktop/mano5.jpg')
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    segmented = sd.segment(np.asarray(img), plot=True)
    """

    segmentation_experiments()

    """
    sd = SkinDetector(dilate=1)
    sd.train()
    basewidth = 300
    img = Image.open(f'../../../../Desktop/caras.jpg')
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    segmented = sd.segment(np.asarray(img), plot=True)



    from skimage.morphology import reconstruction

    print(color.rgb2gray(np.asarray(img)).shape)
    print(segmented.shape)

    maxval = np.amax(color.rgb2gray(np.asarray(img)))
    minval = np.amin(color.rgb2gray(np.asarray(img)))
    print("Max=",maxval,", Min=",minval)
    marker = np.copy(segmented)
    marker[marker > minval] = maxval

    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    ax[0].imshow(segmented, cmap = plt.get_cmap('gray'))
    ax[1].imshow(marker, cmap = plt.get_cmap('gray'))
    plt.show()

    rec = reconstruction(marker, color.rgb2gray(np.asarray(img)), method='erosion')

    fig, ax = plt.subplots(1, 3, figsize=(10,5))
    ax[0].imshow(segmented, cmap = plt.get_cmap('gray'))
    ax[1].imshow(rec, cmap = plt.get_cmap('gray'))
    ax[2].imshow(color.rgb2gray(np.asarray(img)), cmap = plt.get_cmap('gray'))
    plt.show()

    """
"""
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

    """

    


