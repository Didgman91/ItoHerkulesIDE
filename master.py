def bin2npy(Re, Im=""):
    """Converting a Binary File-Array into an Numpy-Array"""

    import numpy as np

    # Datainput:
    data = Re
    datacompl = Im

    # Header:
    dt = np.dtype('uint32')
    dtbin = np.dtype(np.float32)

    # Real parts of Array:
    aint = np.fromfile(data, dt, count=-1, sep='')
    afloat = np.fromfile(data, dtbin, count=-1, sep='')

    a = aint[:2]
    
    if a[0] != 272625313:
        R = "Error: Wrong File"
    
    elif a[1] == 2:
        R1 = np.array(aint[2:4], dtype='uint32')
        R = np.array(afloat[4:], dtype='float32')
    
    elif a[1] == 4:
        R1 = np.array(aint[2:6], dtype='uint32')
        R0 = np.array(afloat[6:], dtype='float32')
        R = np.reshape(R0, (R1[3], R1[1]))

    # Imaginary parts of Array:
    if Im != "":

        aintcompl = np.fromfile(datacompl, dt, count=-1, sep='')
        afloatcompl = np.fromfile(datacompl, dtbin, count=-1, sep='')
    
        acompl = aintcompl[:2]
        
        if acompl[0] != 272625313:
            I = "Error: Wrong File"
        
        elif acompl[1] == 2:
            I1 = np.array(aintcompl[2:4], dtype='uint32')
            I = np.array(afloatcompl[4:], dtype='float32')
            imagnumber = 1j*np.ones([len(I)], dtype='cfloat')
            I_compl = np.multiply(I, imagnumber)
            # Combining Arrays:
            return np.add(R, I_compl)

        elif acompl[1] == 4:
            I1 = np.array(aintcompl[2:6], dtype='uint32')
            I0 = np.array(afloatcompl[6:], dtype='float32')
            I = np.reshape(I0, (I1[3], I1[1]))
            imagnumber = 1j*np.ones((I1[3], I1[1]), dtype='cfloat')
            I_compl = np.multiply(I, imagnumber)
            # Combining Arrays:
            return np.add(R, I_compl)

    else:
        return R

def rename_bin(path):
    """renaming a binary file generatet with F2 into <xshift_yshift_dtype>"""

    import os

    os.chdir(path)

    for f in os.listdir():
    
        if os.path.isdir(f) == True:
            print("file is a directory")

        elif f == "desktop.ini":
            print("desktop.ini")
        
        else:
            f_split = f.split('_')
            dtype = f_split[0:1]
            xshift = f_split[1:2]
            yshift = f_split[2:3]
            # layer = f_split[3:4]
            new_array = xshift, yshift, dtype
            new_string = " ".join(map(str, new_array))
            filename = new_string.replace(".mm","").replace("] [","").replace("]","").replace("[","").replace("''","_").replace(".bin","").replace(".","").strip("'")
            filename_ext = filename + ".bin"
            
            if (f_split[1:2] == filename.split("_")[0:1]):
                print("already renamed")
            else:
                os.rename(f, filename_ext)
                print("file is being renamed")

def find_min_max_shift(path):
    """finding minimum and maximum shifts in each axis"""
    
    import os

    xarray = []
    yarray = []

    os.chdir(path)
    
    i = 0

    for f in os.listdir():
        if os.path.isdir(f) == True:
            print("file is a directory")
        
        elif f == "desktop.ini":
            print("desktop.ini")
        
        else:
            xshift = f.split("_")[0]
            xarray.append(int(xshift))

            yshift = f.split("_")[1]
            yarray.append(int(yshift))

            i = i + 1
    
    xshiftmin = min(xarray)
    xshiftmax = max(xarray)
    yshiftmin = min(yarray)
    yshiftmax = max(yarray)
    numberoffiles = i

    return xshiftmin, xshiftmax, yshiftmin, yshiftmax, numberoffiles

def visualize_beamshift(path):
    """Visualization of x- and y-shift of laser beam"""

    from master import find_min_max_shift, bin2npy
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import math

    os.chdir(path)

    dirinf = find_min_max_shift(path + "/Real/")

    xmin = dirinf[0]
    xmax = dirinf[1]
    ymin = dirinf[2]
    ymax = dirinf[3]
    files = dirinf[4]

    arraydim = int(math.sqrt(files))
    interval = (xmax - xmin) / (arraydim - 1)
    imagedim = 32
    array = np.empty([arraydim,arraydim,imagedim,imagedim])
    print("array dimensions: ", array.shape)

    for f in os.listdir(path + "/Real/"):
        f_Re = os.path.splitext(f)
        Re = path + "/Real/" + f

        for f in os.listdir(path + "/Imaginary/"):
            f_Im = os.path.splitext(f)
            Im = path + "/Imaginary/" + f

            if f_Re[0].split("_Re") == f_Im[0].split("_Im"):
                m = int((int(f_Re[0].split("_")[0]) - xmin) / interval)
                n = int((int(f_Re[0].split("_")[1]) - xmin) / interval)
                array[m,n] = bin2npy(Re, Im)

    array_plt = np.flipud(np.reshape(np.swapaxes(array, axis1=1, axis2=2).astype(float), ((arraydim*imagedim),(arraydim*imagedim))))
    # plt.imshow(array_plt, interpolation="gaussian")
    # plt.show()
    return array