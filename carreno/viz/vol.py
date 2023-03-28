import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import marching_cubes
import matplotlib.pyplot as plt


def hsv_depth(volume):
    # TODO
    pass


def ndarray2obj(pred, angle=[0,0], prec=2):
    """
    Visualize categorical segmentation with a 3D plot
    Parameters
    ----------
    pred : array-like
        Categorical segmentation of objects in volume
    angle : [float, float]
        Angles in degrees for point of view for the object
    pred : int
        Step size for marching cubes when finding vertices in pred
    Returns
    -------
    None
    """
    obj = []
    fig = plt.figure(figsize=(20,20))

    # 1 channel at a time
    nch = pred.shape[-1]
    for ch in range(nch):
        vol_ch = pred[..., ch]
        verts, faces, normals, values = marching_cubes(vol_ch, step_size=prec)
        
        # add subplot
        ax = fig.add_subplot(1, nch, ch+1, projection='3d')
        # add mesh
        ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2], linewidth=0.2, antialiased=True)
        # place camera
        ax.view_init(*angle)
        # clip space
        ax.set_xlim(0, vol_ch.shape[0])
        ax.set_ylim(0, vol_ch.shape[1])
        ax.set_zlim(0, vol_ch.shape[2])

        obj.append([
            verts,
            faces,
            normals,
            values
        ])

    plt.show()

    return obj


def figure_vol_slc(volumes, slc=20, ncol=3):
    """
    Show specified slice for all given volumes
    Parameters
    ----------
    volumes : [array-like]
        List of volumes
    slc : int
        Slice index to show from a volume (over axis 0)
    ncol : int
        Number of columns in resulting figure
    Returns
    -------
    none
    """
    plt.figure(figsize=(10,10))
    
    nline = np.ceil(len(volumes) / ncol).astype(int)
    
    for i in range(len(volumes)):
        plt.subplot(nline, ncol, i+1)
        plt.title("Slice", i)
        if volumes[i].dtype == bool:
            plt.imshow(volumes[i][slc].astype(float))
        else:
            plt.imshow(volumes[i][slc])

    plt.show()


def get_3_planes(volume, ax0=0, ax1=0, ax2=0):
    """
    Get volume using 3 perpendicular planes
    (aka sagittal, coronal and transversal)
    Parameters
    ----------
    volume : array-like
        Volume to visualize
    ax0 : int
        Index for axis 0 plane (sagittal)
    ax1 : int
        Index for axis 1 plane (coronal)
    ax2 : int
        Index for axis 2 plane (transversal)
    Returns
    -------
    ax0_plane : array-like
        Volume sagittal slice
    ax1_plane : array-like
        Volume coronal slice
    ax2_plane : array-like
        Volume transversal slice
    """
    ax0_plane = volume[ax0]
    ax1_plane = np.moveaxis(np.moveaxis(volume, [1], [2]), [0], [2])[ax1]
    ax2_plane = np.moveaxis(volume, [0], [1])[ax2]

    return ax0_plane, ax1_plane, ax2_plane


def plot_3_planes(sagittal, coronal, transversal, ratios=[1,1], pads=[0, 0], cmap='gray', filename=None):
    """
    Plot sagittal, coronal and transversal plane.
    Parameters
    ----------
    sagittal : array-like
        Image in top left corner
    coronal : array-like
        Image in top right corner
    transversal : array-like
        Image in bottom left corner
    ratios : [float, float]
        gridspec ratios for ax1 width and ax2 height [W, H]
    pads : [float, float]
        width and height pad between subplots [W, H]
    cmap : matplotlib.cm.ColormapRegistry
        Color map for plot
    filename : Path
        Path to save figure if provided
    Returns
    -------
    None
    """
    fig, axes = plt.subplots(nrows=2,
                             ncols=2,
                             squeeze=True,
                             dpi=150,
                             gridspec_kw={'width_ratios':  [1, ratios[0]],
                                          'height_ratios': [1, ratios[1]]})
    
    axes[0, 0].imshow(sagittal, cmap=cmap)
    axes[0, 1].imshow(coronal, cmap=cmap)
    axes[1, 0].imshow(transversal, cmap=cmap)
    axes[1, 1].set_visible(False)

    axes[0, 0].axis('off')
    axes[0, 0].set_aspect('auto')
    axes[0, 1].yaxis.set_label_position("right")
    axes[0, 1].yaxis.tick_right()

    plt.tight_layout(h_pad=pads[0], w_pad=pads[1])

    if filename:
        plt.savefig(filename, bbox_inches='tight')
    
    plt.show()