import os
import tensorflow as tf

import utils
from carreno.nn.metrics import Dice, ClDice, DiceClDice
from carreno.processing.transforms import Read
from carreno.processing.categorical import categorical_multiclass

config = utils.get_config()
ilastik_dir = config['VOLUME']['ilastik']
tresh_dir   = config['VOLUME']['threshold']
unet2d_dir  = config['VOLUME']['unet2d']
unet3d_dir  = config['VOLUME']['unet3d']
target_dir  = config['VOLUME']['target']

# evaluation metrics
iters = 10
ndim  = 3
pad   = 2
smooth = 1e-6
dice          = Dice(smooth=smooth).coefficient
cldice        = ClDice(iters=iters, ndim=ndim, mode=pad, smooth=smooth).coefficient
dicecldice    = DiceClDice(iters=iters, ndim=ndim, mode=pad, smooth=smooth).coefficient
v1, v2        = [vol_name + ".tif" for vol_name in config["TRAINING"]["evaluation"]]
sep = "=" * 10

def read_volume(dir, vol):
    y_path = os.path.join(target_dir, vol)
    p_path = os.path.join(dir, vol)
    y, p, _ = Read()(y_path, p_path)
    return y, categorical_multiclass(p)

def to_tensor(v):
    tv = tf.expand_dims(tf.convert_to_tensor(v, dtype=tf.float32), 0)
    return tv

def eval_volumes(dir, vol1, vol2):
    # read volumes
    y1, p1 = read_volume(dir, vol1)
    y2, p2 = read_volume(dir, vol2)
    # volume to tensors for evaluation
    ty1 = to_tensor(y1)
    tp1 = to_tensor(p1)
    ty2 = to_tensor(y2)
    tp2 = to_tensor(p2)
    # evaluate
    metrcs = [dice, cldice, dicecldice]
    print("For dir:", dir)
    for i in range(len(metrcs)):
        mtc = metrcs[i]
        s1 = mtc(ty1, tp1)
        s2 = mtc(ty2, tp2)
        avg = (s1 + s2) / 2
        print(" -", mtc.__name__, "score:", avg)
    return

if __name__ == "__main__":
    for dir in [ilastik_dir, tresh_dir, unet2d_dir, unet3d_dir]:
        eval_volumes(dir, v1, v2)
        print(sep)

# RESULT
"""
For dir: data/output/ilastik
 - dice score: tf.Tensor(0.82408404, shape=(), dtype=float32)
 - cldice score: tf.Tensor(0.38683397, shape=(), dtype=float32)
 - dicecldice score: tf.Tensor(0.605459, shape=(), dtype=float32)
==========
For dir: data/output/threshold
 - dice score: tf.Tensor(0.92373824, shape=(), dtype=float32)
 - cldice score: tf.Tensor(0.5574617, shape=(), dtype=float32)
 - dicecldice score: tf.Tensor(0.7406, shape=(), dtype=float32)
==========
For dir: data/output/unet2d
 - dice score: tf.Tensor(0.92305696, shape=(), dtype=float32)
 - cldice score: tf.Tensor(0.5822147, shape=(), dtype=float32)
 - dicecldice score: tf.Tensor(0.75263584, shape=(), dtype=float32)
==========
For dir: data/output/unet3d
 - dice score: tf.Tensor(0.9284917, shape=(), dtype=float32)
 - cldice score: tf.Tensor(0.61384153, shape=(), dtype=float32)
 - dicecldice score: tf.Tensor(0.77116656, shape=(), dtype=float32)
"""
