import numpy as np
import tifffile as tif

anno_path = 'C:/Users/User/Desktop/Etude/LINUM/Carreno segmentation/annotation/'
data_path = 'C:/Users/User/Desktop/Etude/LINUM/Carreno segmentation/data/'

volumes = [
    anno_path + 'GFP1-2&SLIK1-2/Slik1.tif',
    anno_path + 'GFP1-2&SLIK1-2/Slik2.tif',
    anno_path + 'GFP1-2&SLIK1-2/Ctrl1.tif',
    anno_path + 'GFP1-2&SLIK1-2/Ctrl2.tif',
    anno_path + 'GFP/Ctrl3.tif',
    anno_path + 'GFP/Ctrl4.tif',
    anno_path + 'SLIK/Slik3.tif',
    anno_path + 'SLIK/Slik4.tif',
    anno_path + 'SLIK/Slik5.tif',
    anno_path + 'Slik6/Slik6.tif'
]

anno_bodies = [
    anno_path + 'GFP1-2&SLIK1-2/Mask cell body Slik1.tif',
    anno_path + 'GFP1-2&SLIK1-2/Mask cell body Slik2.tif',
    anno_path + 'GFP1-2&SLIK1-2/Mask cell body Ctrl1.tif',
    anno_path + 'GFP1-2&SLIK1-2/Mask cell body Ctrl2.tif',
    anno_path + 'GFP/Mask cell body Ctrl3.tif',
    anno_path + 'GFP/Mask cell body Ctrl4.tif',
    anno_path + 'SLIK/Mask cell body Slik3.tif',
    anno_path + 'SLIK/Mask cell body Slik4.tif',
    anno_path + 'SLIK/Mask cell body Slik5.tif',
    anno_path + 'Slik6/Slik-6 deconvoluted-annotation-cell_body.tif'
]

anno_cytos = [
    anno_path + 'GFP1-2&SLIK1-2/Mask cytoneme Slik1.tif',
    anno_path + 'GFP1-2&SLIK1-2/Mask cytoneme Slik2.tif',
    anno_path + 'GFP1-2&SLIK1-2/Mask cytoneme Ctrl1.tif',
    anno_path + 'GFP1-2&SLIK1-2/Mask cytoneme Ctrl2.tif',
    anno_path + 'GFP/Mask cytoneme Ctrl3.tif',
    anno_path + 'GFP/Mask cytoneme Ctrl4.tif',
    anno_path + 'SLIK/Mask Cytoneme Slik3.tif',
    anno_path + 'SLIK/Mask Cytoneme Slik4.tif',
    anno_path + 'SLIK/Mask Cytoneme Slik5.tif',
    anno_path + 'Slik6/Slik-6 deconvoluted-annotation-cytonemes.tif'
]