from SkinDetector import SkinDetector
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def segmentation_experiments():
    sd = SkinDetector(hist_range = (5,95), dilate = 2)
    sd.train()

    segmented = sd.segment_dataset(sd.TR_DATA)
    
    # Uncomment this to visualize the segmented images.
    
    for idx in range(0, len(segmented)):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(sd.TR_DATA[idx])
        ax[1].imshow(segmented[idx], cmap = plt.get_cmap('gray'))
        plt.show()
    
    sd.validate(segmented, sd.TR_MASK)

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

    # segmentation_experiments()

    
    sd = SkinDetector(dilate=1)
    sd.train()
    basewidth = 300
    img = Image.open(f'../../../../Desktop/manos.jpg')
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    sd.segment(np.asarray(img), plot=True)
    





