# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt


def write_obj(filename, v, f, n):
    """
    Write a wavefront obj file from given vertices, faces and normals
    Taken from : https://stackoverflow.com/questions/48844778/create-a-obj-file-from-3d-array-in-python
    Parameters
    ----------
    filename : Path
        Path to file to save to
    v : 
    """
    # faces index must start at 1
    faces = f.copy()
    if 0 in faces:
        faces += 1

    obj_file = open(filename, 'w')
    for item in v:
        if len(item) == 3:
            obj_file.write("v {0} {1} {2}\n".format(item[0], item[1], item[2]))
        else:
            obj_file.write("v {0} {1}\n".format(item[0], item[1]))

    for item in n:
        if len(item) == 3:
            # 3D
            obj_file.write("vn {0} {1} {2}\n".format(item[0], item[1], item[2]))
        else:
            # 2D
            obj_file.write("vn {0} {1}\n".format(item[0], item[1]))

    for item in faces:
        if len(item) == 3:
            # 3D
            obj_file.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(item[0], item[1], item[2]))
        else:
            obj_file.write("f {0}//{0} {1}//{1}\n".format(item[0], item[1]))

    obj_file.close()