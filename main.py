from SkinDetector import SkinDetector
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    sd = SkinDetector(hist_range=(5,95), dilate = 2)
    sd.train()

    sd.segment(sd.TR_DATA[5])

    segmented = sd.segment_dataset(sd.TR_DATA)
    
    """
    for idx in range(0, len(segmented)):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(sd.TR_DATA[idx])
        ax[1].imshow(segmented[idx], cmap = plt.get_cmap('gray'))
        plt.show()
    """
    sd.validate(segmented, sd.TR_MASK)

    """
    print('\nMedian')
    
    print(f'{np.round(np.median(sd.accuracy), 3) * 100}%')
    print(f'{np.round(np.median(sd.precision), 3) * 100}%')
    print(f'{np.round(np.median(sd.recall),  3) * 100}%')
    print(f'{np.round(np.median(sd.F1score), 3) * 100}%')
    """

    print('\nMean')

    print(f'{np.round(np.mean(sd.accuracy) * 100, 3)}%')
    print(f'{np.round(np.mean(sd.precision) * 100, 3)}%')
    print(f'{np.round(np.mean(sd.recall) * 100, 3)}%')
    print(f'{np.round(np.mean(sd.F1score) * 100, 3)}%')

    print('\n')


