from skimage.measure import marching_cubes
import matplotlib.pyplot as plt

def ndarray2obj(pred, angle=[0,0], prec=2):
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


def write_obj(filename, v, f, n):
    # https://stackoverflow.com/questions/48844778/create-a-obj-file-from-3d-array-in-python
    faces = f.copy()
    if 0 in faces:
        faces += 1

    obj_file = open(filename, 'w')
    for item in v:
        obj_file.write("v {0} {1} {2}\n".format(item[0], item[1], item[2]))

    for item in n:
        obj_file.write("vn {0} {1} {2}\n".format(item[0], item[1], item[2]))

    for item in faces:
        obj_file.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(item[0], item[1], item[2]))  

    obj_file.close()