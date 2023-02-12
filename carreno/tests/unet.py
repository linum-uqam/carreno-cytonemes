# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import carreno.nn.unet

def main():
    def test_train_n_pred(name, shape, n_class, depth, n_feat, backbone):
        print(name, ":")

        # generate data
        n = np.prod(np.array(shape[:-1]))
        x = np.arange(n).reshape(shape[:-1])
        classes = x % n_class
        c = [classes == i for i in range(n_class)]
        x = np.stack([x]*shape[-1], axis=-1)
        y = np.stack([c[0], c[1]], axis=-1) if n_class == 2 else np.stack(c, axis=-1)

        # show input shapes
        #print(x.shape, y.shape)

        # create architecture
        print(" - {:.<9}... ".format("assemble "), end="")
        model = carreno.nn.unet.UNet(shape=shape,
                                     n_class=n_class,
                                     depth=depth,
                                     n_feat=n_feat,
                                     backbone=backbone,
                                     pretrained=False)
        print("done")

        # show architecture once assembled
        #model.summary()

        # compile model
        print(" - {:.<9}... ".format("compile "), end="")
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                      loss=tf.keras.losses.CategoricalCrossentropy())
        print("done")

        # train model
        print(" - {:.<9}... ".format("train "), end="")
        xs = x[np.newaxis, :]  # otherwise x is iterated over axis 0 in batch
        ys = y[np.newaxis, :]  # ^ same for y
        history = model.fit(x=xs,
                            y=ys,
                            batch_size=1,
                            epochs=10,
                            verbose=0)
        print("done")

        # model prediction
        print(" - {:.<9}... ".format("predict "), end="")
        model.predict(np.stack([x]*3), verbose=0)
        print("done")

        return

    tests = [
        {"name":"UNet/2D/gray/sigmoid/none ", "shape":[32,32,1],    "n_class":2, "depth":3, "n_feat":8,  "backbone":None   },
        {"name":"UNet/2D/gray/softmax/none ", "shape":[32,32,1],    "n_class":3, "depth":3, "n_feat":8,  "backbone":None   },
        {"name":"UNet/2D/rgb /softmax/none ", "shape":[32,32,3],    "n_class":3, "depth":3, "n_feat":8,  "backbone":None   },
        {"name":"UNet/2D/gray/softmax/vgg16", "shape":[32,32,1],    "n_class":3, "depth":5, "n_feat":64, "backbone":"vgg16"},  # backbone TODO
        {"name":"UNet/2D/rgb /softmax/vgg16", "shape":[32,32,3],    "n_class":3, "depth":5, "n_feat":64, "backbone":"vgg16"},
        {"name":"UNet/3D/gray/sigmoid/none ", "shape":[32,32,32,1], "n_class":2, "depth":3, "n_feat":8,  "backbone":None   },
        {"name":"UNet/3D/gray/softmax/none ", "shape":[32,32,32,1], "n_class":3, "depth":3, "n_feat":8,  "backbone":None   },
        {"name":"UNet/3D/rgb /softmax/none ", "shape":[32,32,32,3], "n_class":3, "depth":3, "n_feat":8,  "backbone":None   },
        {"name":"UNet/3D/gray/softmax/vgg16", "shape":[32,32,32,1], "n_class":3, "depth":5, "n_feat":64, "backbone":"vgg16"},  # backbone TODO
        {"name":"UNet/3D/rgb /softmax/vgg16", "shape":[32,32,32,3], "n_class":3, "depth":5, "n_feat":64, "backbone":"vgg16"},
    ] 

    print()
    sep = "-----------------------------"
    for i in range(len(tests)):
        test_train_n_pred(**tests[i])
        if i == len(tests)-1:
            continue
        print(sep)


if __name__ == "__main__":
    main()