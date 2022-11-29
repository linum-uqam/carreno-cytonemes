import matplotlib.pyplot as plt
import numpy as np

def hsv_depth(volume):
    # TODO
    pass


def figure_vol_slc(volumes, titles, slc=20, ncol=3):
    plt.figure(figsize=(10,10))
    
    nline = np.ceil(len(volumes) / ncol).astype(int)
    
    for i in range(len(volumes)):
        plt.subplot(nline, ncol, i+1)
        try:
            plt.title(titles[i])
        except:
            pass
        if volumes[i].dtype == bool:
            plt.imshow(volumes[i][slc].astype(float))
        else:
            plt.imshow(volumes[i][slc])

    plt.show()
