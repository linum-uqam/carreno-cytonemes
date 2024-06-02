import yaml
import os
import tifffile as tif
import pathlib
import skimage.restoration

infos = {}
with open("config.yml", 'r') as file:
  infos = yaml.safe_load(file)

psf = tif.imread(os.path.join(infos['DIR']['psf'], "Averaged PSF.tif"))
test1 = tif.imread(os.path.join(infos['VOLUME']['input'], "slik1.tif"))
#plt.imshow(psf);plt.show()

print(psf.min(), psf.max())
print(test1.min(), test1.max())

inpdir = infos['VOLUME']['input']
outdir = infos['VOLUME']['rl_input']
inpdir2 = infos['VOLUME']['unlabeled']
outdir2 = infos['VOLUME']['rl_unlabeled']

def create_denoised_folder(folder, input_folder):
    """
    Create W folder using Y folder
    Parameters
    ----------
    folder : str, Path
        Path where to create/override the richardson-lucy denoised volumes folder
    input_folder : str, Path
        Path where volumes are
    Returns
    -------
    None
    """
    # remove patches folders if they already exist
    if os.path.isdir(folder):
        print(folder, "is in the way!")
        return
    
    # create folder
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

    filenames = []
    for f in os.listdir(input_folder):
        filenames.append(f)
        
    for f in filenames:
        # save weighted volume
        x = tif.imread(os.path.join(input_folder, f))
        y = skimage.restoration.richardson_lucy(x, psf, 25)
        
        tif.imwrite(os.path.join(folder, f), y, photometric="minisblack")

create_denoised_folder(outdir, inpdir)
create_denoised_folder(outdir, inpdir)