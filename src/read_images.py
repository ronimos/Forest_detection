import os
from osgeo import gdal
import numpy as np
import cv2
from glob import glob
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from scipy import stats

IMAGE_PATH = "../data/images"
colors = ["Blue", "Green", "Red", "NIR"]


def read_image_data(path=os.path.join(IMAGE_PATH, "*Raw*.tiff"),
                    show=False):
    img = {}
    data_files = glob(path)
    for i, f in enumerate(data_files):
        ds = gdal.Open(f)
        im = np.array(ds.GetRasterBand(1).ReadAsArray())
        img[colors[i]] = im
        geotransform = ds.GetGeoTransform()
        ds = None
    if show:
        fig, ax = plt.subplots(nrows=3, ncols=1)
        # True Colors:
        TC = np.stack([img["Red"], img["Green"], img["Blue"]], axis=-1)
        ax[0].imshow(TC)
        ax[0].set_title("True colors")
        # NDVI: (NIR - Red) / (NIR + Red)
        NDVI = (img["NIR"]-img["Red"])/(img["NIR"]+img["Red"])
        ax[1].imshow(NDVI)
        ax[1].set_title("NDVI")
        # False Colors (B8, B4, B3)
        FC = np.stack([img["NIR"], img["Red"], img['Green']], axis=-1)
        ax[2].imshow(FC)
        ax[2].set_title("False colors")
        plt.tight_layout()
        # plt.show()
        
    return img, geotransform



def convert_to_clasification_data(arr, kernel=9):
    pass


if __name__ == "__main__":
    
    bands, projection = read_image_data(show=False)
    TC = np.stack([bands["Red"], bands["Green"], bands["Blue"]], axis=-1)
    img = cv2.convertScaleAbs(TC * 4, alpha=255)
    cv2.imwrite('../data/img.png', img[...,::-1])
    plt.imshow(cv2.convertScaleAbs(TC * 4, alpha=255))
    plt.show()

    

