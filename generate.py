import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.type_check import imag
from scipy.io import loadmat
import numpy
import random
import os
import xml.dom.minidom

def ShearX(img, v):  # [-0.3, 0.3]
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))

def ShearY(img, v):  # [-0.3, 0.3]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))

def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    v = v*img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))

def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    v = v*img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))

def Rotate(img, v):  # [-30, 30]
    return img.rotate(v)

def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)

def Invert(img, _):
    return PIL.ImageOps.invert(img)

def Equalize(img, _):
    return PIL.ImageOps.equalize(img)

def Flip(img, _):  # not from the paper
    return PIL.ImageOps.mirror(img)

def Solarize(img, v):  # [0, 256]
    return PIL.ImageOps.solarize(img, v)

def Posterize(img, v):  # [4, 8]
    v = int(v)
    return PIL.ImageOps.posterize(img, v)

def Contrast(img, v):  # [0.1,1.9]
    return PIL.ImageEnhance.Contrast(img).enhance(v)

def Color(img, v):  # [0.1,1.9]
    return PIL.ImageEnhance.Color(img).enhance(v)

def Brightness(img, v):  # [0.1,1.9]
    return PIL.ImageEnhance.Brightness(img).enhance(v)

def Sharpness(img, v):  # [0.1,1.9]
    return PIL.ImageEnhance.Sharpness(img).enhance(v)

def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
    w, h = img.size
    v = v*img.size[0]
    x0 = np.random.uniform(w-v)
    y0 = np.random.uniform(h-v)
    xy = (x0, y0, x0+v, y0+v)
    color = (127, 127, 127)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img

def SamplePairing(imgs):  # [0, 0.4]
    def f(img1, v):
        i = np.random.choice(len(imgs))
        # img2 = PIL.Image.fromarray(imgs[i])
        img2 = imgs[i]
        return PIL.Image.blend(img1, img2, v)
    return f

def get_transformations(imgs):
    return [
        (ShearX, -0.3, 0.3),
        (ShearY, -0.3, 0.3),
        (TranslateX, -0.45, 0.45),
        (TranslateY, -0.45, 0.45),
        (Rotate, -30, 30),
        (AutoContrast, 0, 1),
        (Invert, 0, 1),
        (Equalize, 0, 1),
        (Solarize, 0, 256),
        (Posterize, 4, 8),
        (Contrast, 0.1, 1.9),
        (Color, 0.1, 1.9),
        (Brightness, 0.1, 1.9),
        (Sharpness, 0.1, 1.9),
        (Cutout, 0, 0.2),
        (SamplePairing(imgs), 0, 0.4),
    ]


def best_policy():
    policy = [
        # Subpolicy 1
        'Operation 14  P=0.600 M=0.044  Operation  1  P=0.900 M=-0.167',
        'Operation 12  P=0.400 M=1.500  Operation  0  P=0.700 M=-0.033',
        'Operation  3  P=0.300 M=-0.150  Operation 14  P=0.100 M=0.067',
        'Operation  7  P=0.400 M=0.111  Operation  7  P=0.400 M=0.222',
        'Operation  4  P=0.200 M=3.333  Operation  6  P=0.200 M=0.667',
        # Subpolicy 2
        'Operation  4  P=0.000 M=10.000  Operation 14  P=0.800 M=0.111',
        'Operation  5  P=0.300 M=0.778  Operation 13  P=0.600 M=0.500',
        'Operation  9  P=1.000 M=4.444  Operation  7  P=0.200 M=0.667',
        'Operation  4  P=0.800 M=16.667  Operation  8  P=0.500 M=170.667',
        'Operation  7  P=0.600 M=0.556  Operation 15  P=1.000 M=0.089',
        # Subpolicy 3
        'Operation 15  P=0.100 M=0.133  Operation 12  P=0.300 M=0.500',
        'Operation  5  P=0.700 M=0.667  Operation  9  P=0.100 M=7.556',
        'Operation  3  P=0.400 M=-0.350  Operation  3  P=0.500 M=0.150',
        'Operation  3  P=0.800 M=-0.150  Operation  5  P=0.400 M=1.000',
        'Operation  8  P=0.600 M=227.556  Operation  2  P=0.300 M=-0.150',
        # Subpolicy 4
        'Operation 13  P=0.100 M=1.500  Operation  4  P=0.100 M=-3.333',
        'Operation  0  P=0.600 M=0.233  Operation 12  P=0.100 M=0.700',
        'Operation  3  P=0.000 M=-0.450  Operation  2  P=0.800 M=-0.350',
        'Operation  5  P=0.100 M=0.222  Operation  9  P=1.000 M=5.333',
        'Operation  6  P=0.500 M=0.333  Operation  1  P=0.100 M=-0.033',
        # Subpolicy 5
        'Operation 13  P=0.400 M=1.300  Operation  6  P=0.200 M=0.111',
        'Operation 10  P=0.500 M=0.700  Operation  1  P=0.100 M=-0.100',
        'Operation 12  P=0.500 M=1.700  Operation  9  P=0.400 M=6.222',
        'Operation 15  P=0.600 M=0.133  Operation 11  P=0.400 M=1.500',
        'Operation 14  P=0.300 M=0.200  Operation  1  P=0.800 M=0.100',
    ]
    return policy

if __name__ == '__main__':
    dirpath = 'test'
    ProcessedPath_Img = 'results'
    for filename in os.listdir(dirpath):
        imgs = []
        tr = PIL.Image.open(os.path.join(dirpath, filename)).convert('RGB')
        imgs.append(tr)
        transfs = get_transformations(imgs)
        for i in range(3):
            img2 = imgs[0]
            best_policy_ = best_policy()
            policy_      = best_policy_[random.randint(0, len(best_policy_) - 1)]
            print(policy_)
            op=[x.strip() for x in policy_.split(' ') if x.strip() != '']
            for t, min, max in [transfs[int(op[1])]]:
                if random.random() < float(op[2].split('=')[-1]):
                    v = float(op[3].split('=')[-1])
                    img2 = t(img2, v)
            for t, min, max in [transfs[int(op[5])]]:
                if random.random() < float(op[6].split('=')[-1]):
                    v = float(op[7].split('=')[-1])
                    img2 = t(img2, v)
            img2.save(ProcessedPath_Img  + '/' + filename.split('.')[0]+ '_'+str(i)+'.jpg')            
    