import nevergrad as ng
import numpy as np
from scipy import misc
import cv2
import skimage


#quick experiment using nevergrad.
#the goal of the experiment is to reproduce as well as possible a target image
#by finding a list of gaussian (positive or negative) which will sum up to a good
#approximation of the image.
#Each gaussian have 4 parameters, center (x,y), intensity, and FWHM


SIZE = 256

def show(name, array):
    cv2.imshow(name, array)
    cv2.waitKey(1)


def makeGaussian(size, fwhm = 3, center=None):
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]
    
    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]
    
    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)


def build_image(inputs):
    count = len(inputs) // 4

    idx = 0
    array = np.zeros((SIZE,SIZE))
    for i in range(count):
        x = (inputs[idx]) * SIZE
        y = (inputs[idx + 1]) * SIZE
        size = 1 + np.abs(inputs[idx + 2]) * 64.0
        val = 0.3 * (inputs[idx + 3])
        idx = idx + 4

        array = array + makeGaussian(SIZE, size, (x,y)) * val

    return array


def gen_image(x):
    #print(x)
    temp_image = build_image(x)
    error = temp_image - target_image
    err_v = np.mean(np.square(error))
    return err_v


budget = 4400

target_image = skimage.data.astronaut()[:,:,0]

print(target_image.shape)
target_image = target_image[0:SIZE, 100:100+SIZE]/255.0
show("target", target_image)


for tool in ["Shiwa","CMA","TBPSA"]:

    optim = ng.optimizers.registry[tool](parametrization=128, budget=budget)
    optim.parametrization.set_bounds(-4, 4)

    for iter in range(budget ):

        x1 = optim.ask()
        y1 = gen_image(*x1.args)
        optim.tell(x1, y1)

        if (iter % 20 == 0):
            print(iter, y1)
            recommendation = optim.recommend()
            best = build_image(*recommendation.args)
            show("best", best)
            cv2.waitKey(1)
    
    recommendation = optim.recommend()
    best = build_image(*recommendation.args)
    show("best", best)
    print("* ", tool, " provides a vector of parameters with test error ", gen_image(*recommendation.args))
cv2.waitKey(0)
