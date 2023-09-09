# -*- coding: utf-8 -*-
import wandb
import tifffile as tif
import skimage.filters
import numpy as np
import os
import tensorflow as tf
from skimage.restoration import richardson_lucy, denoise_nl_means, estimate_sigma

# local imports
import utils
import carreno.nn.metrics as mtc

filters = {
    'iso':      skimage.filters.threshold_isodata,
    'li':       skimage.filters.threshold_li,
    'mean':     skimage.filters.threshold_mean,
    'min':      skimage.filters.threshold_minimum,
    'otsu':     skimage.filters.threshold_otsu,
    'triangle': skimage.filters.threshold_triangle,
    'yen':      skimage.filters.threshold_yen
}

sweep_configs = {
    0 : {
        'method': 'grid',
        'name':   'sweep',
        'project': 'threshold_global',
        'metric': {
            'goal': 'maximize',
            'name': 'val_dicecldice'
        },
        'parameters': {
            'filter': {'values': list(filters.keys())}
        }
    },
    1 : {
        'method': 'grid',
        'name':   'sweep',
        'project': 'denoising_psf',
        'metric': {
            'goal': 'maximize',
            'name': 'val_dicecldice'
        },
        'parameters': {
            'filter': {'value': 'otsu'},
            'iter':   {'values': list(range(5, 26))}
        }
    },
    2 : {
        'method': 'grid',
        'name':   'sweep',
        'project': 'denoising_nlm',
        'metric': {
            'goal': 'maximize',
            'name': 'val_dicecldice'
        },
        'parameters': {
            'filter': {'value': 'otsu'},
            'size':   {'values': list(range(5, 10))},
            'dist':   {'values': list(range(5, 10))},
            'std':    {'values': np.array(range(0, 5)) / 2}
        }
    },
    3 : {
        'method': 'grid',
        'name':   'sweep',
        'project': 'restore_unsharp',
        'metric': {
            'goal': 'maximize',
            'name': 'val_dicecldice'
        },
        'parameters': {
            'filter': {'value': 'otsu'},
            'size':   {'value': 7},
            'dist':   {'value': 5},
            'std':    {'value': 1.5},
            'radius': {'values': list(range(0, 50, 5))},
            'aratio': {'values': np.array(range(0, 11)) / 10}
        }
    },
    4 : {
        'method': 'grid',
        'name':   'sweep',
        'project': 'restore_frangi',
        'metric': {
            'goal': 'maximize',
            'name': 'val_dicecldice'
        },
        'parameters': {
            'filter': {'value': 'otsu'},
            'size':   {'value': 7},
            'dist':   {'value': 15},
            'std':    {'value': 1.5},
            'radius': {'value': 1.5},
            'aratio': {'value': 0.5},
            'fratio': {'values': np.array(range(0, 11)) / 10}
        }
    },
    5 : {
        'method': 'grid',
        'name':   'sweep',
        'project': 'threshold_adaptive',
        'metric': {
            'goal': 'maximize',
            'name': 'val_dicecldice'
        },
        'parameters': {
            'filter': {'value': 'otsu'},
            'size':   {'value': 7},
            'dist':   {'value': 15},
            'std':    {'value': 1.5},
            'radius': {'value': 1.5},
            'aratio': {'value': 0.3},
            'fratio': {'value': 0.3},
            'dsize':  {'values': list(range(3, 10, 2))},
            'ssize':  {'values': list(range(3, 28, 2))}
        }
    }
}

class Sweeper:
    def __init__(self, config):
        """
        Setup sweeps for wandb.
        Parameters
        ----------
        config : dict
            Sweeps configuration.
            Refer to Wandb for format https://docs.wandb.ai/guides/sweeps/define-sweep-configuration
        """
        self.prj_config = utils.get_config()
        self.swp_config = config
        self.params = {
            'filter':   'otsu',     # filter method
            'iter':     10,         # number of iterations for Richardson-Lucy
            'size':     5,          # patch size for nlm denoising
            'dist':     5,          # max distance for patch comparisons with nlm
            'std':      1.5,        # standard deviation for nlm
            'radius':   2,          # gaussian filter sigma for unsharp filter
            'uratio':   0.5,        # ratio of unsharp filter when summed with img
            'fratio':   0.5,        # ratio of frangi filter when summed with img
            'dsize':    5,          # patch size depth for adaptive threshold
            'ssize':    11,         # patch size height and width for adaptive threshold
        }
    
    def sweep(self):
        wandb.init()

        print("###############")
        print("# UPDATE ATTR #")
        print("###############")
        for param in self.swp_config['parameters'].keys():
            if param in self.params:
                print("-update {} from {} to {}".format(param, self.params[param], getattr(wandb.config, param, None)))
                self.params[param] = getattr(wandb.config, param, self.params[param])

        print("###############")
        print("# SET LOADERS #")
        print("###############")

        files = os.listdir(self.prj_config['VOLUME']['input'])
        fullpath = lambda dir, files : [os.path.join(dir, name) for name in files]
        xs = fullpath(self.prj_config['VOLUME']['input'],  files)
        ys = fullpath(self.prj_config['VOLUME']['target'], files)

        print("Dataset")
        print("-nb of instances :", len(xs), "/", len(ys))

        print("###########")
        print("# DENOISE #")
        print("###########")

        for i in range(len(xs)):
            x, y = tif.imread(xs[i]), tif.imread(ys[i])
            
            # richardson-lucy
            psf = tif.imread(os.path.join(self.prj_config['DIR']['psf'], "Averaged PSF.tif"))
            x   = richardson_lucy(x, psf, iterations=self.params['iter'])

            # non-local means
            sigma_est = estimate_sigma(x)
            x = denoise_nl_means(x,
                                 patch_size=self.params['size'],
                                 patch_distance=self.params['dist'],
                                 h=0.8*sigma_est,  # recommend slightly less than standard deviation
                                 sigma=sigma_est,
                                 fast_mode=True,   # cost more memory
                                 preserve_range=True)

            # unsharp filter
            x = skimage.filters.unsharp_mask(x, radius=self.params['radius'], amount=self.params['uratio'])
            x = np.clip(x, 0, 1)

            """ # sloooow
            # frangi filter
            weighted_frangi = skimage.filters.frangi(x) * self.params['fratio']
            x = np.clip(x + weighted_frangi, 0, 1)
            """
            
            # make ground-truth binary
            y = np.logical_or(y[..., 1], y[..., 2])  # only keep body and cyto

            xs[i], ys[i] = x, y

        print("############")
        print("# EVALUATE #")
        print("############")

        filter = filters[self.params['filter']]
        iters = 10
        ndim  = 3
        pad   = 2
        dice       = mtc.Dice()
        cldice     = mtc.ClDice(    iters=iters, ndim=ndim, mode=pad)
        dicecldice = mtc.DiceClDice(iters=iters, ndim=ndim, mode=pad)
        metrics = [m.coefficient for m in [dice, cldice, dicecldice]]
        results = [[]] * len(metrics)
        ps = []

        if 'values' in self.swp_config['parameters'].get('dsize', []):
            print("######################")
            print("# ADAPTIVE THRESHOLD #")
            print("######################")

            for i in range(len(xs)):
                x, y = xs[i], ys[i]
                block_size = [self.params['dsize'], *[self.params['ssize']] * 2]
                local_thresh = skimage.filters.threshold_local(x,
                                                               block_size=block_size,
                                                               method='generic',
                                                               param=filter)
                mask = x > local_thresh
                ps.append(mask)  
        else:
            print("####################")
            print("# GLOBAL THRESHOLD #")
            print("####################")

            for i in range(len(xs)):
                x, y = xs[i], ys[i]
                global_thresh = filter(x)
                mask = x > global_thresh
                ps.append(mask)

        # find segmentation score vs targets
        nd_to_tensor = lambda x: tf.convert_to_tensor(np.expand_dims(np.expand_dims(x, axis=-1), axis=0), dtype=tf.float32)
        log_dict = {}
        for i in range(len(ps)):
            tensor_ys = nd_to_tensor(ys[i])
            tensor_ps = nd_to_tensor(ps[i])
            for j in range(len(metrics)):
                score = metrics[j](tensor_ys, tensor_ps).numpy()
                results[j].append(score)
                
        for i in range(len(metrics)):
            log_dict[metrics[i].__name__] = sum(results[i]) / len(results[i])

        wandb.log(log_dict)


def main():
    current_test = 0
    sweep_config = sweep_configs[current_test]
    sweep_id = wandb.sweep(sweep_config)
    sweeper = Sweeper(sweep_config)
    wandb.agent(sweep_id, function=sweeper.sweep)
    

if __name__ == "__main__":
    main()