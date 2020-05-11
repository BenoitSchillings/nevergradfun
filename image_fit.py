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


def makeGaussian(size, fwhm, center):
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]
    

    x0 = center[0]
    y0 = center[1]
    
    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)


def build_image(inputs):
    count = len(inputs) // 4

    idx = 0
    array = np.zeros((SIZE,SIZE))
    for i in range(count):
        x = 128 + (inputs[idx]) * (SIZE/4.0)
        y = 128 + (inputs[idx + 1]) * (SIZE/4.0)
        size = 1 + np.abs(inputs[idx + 2]) * 32.0
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


budget = 114400

target_image = skimage.data.astronaut()[:,:,0]

print(target_image.shape)
target_image = target_image[0:SIZE, 100:100+SIZE]/255.0
show("target", target_image)


for tool in ["DoubleFastGADiscreteOnePlusOne","CMA","TBPSA","Shiwa"]:

    optim = ng.optimizers.registry[tool](parametrization=1024, budget=budget, num_workers=10)
    #optim.parametrization.set_bounds(-4, 4)

    for iter in range(budget ):

        x1 = optim.ask()
        y1 = gen_image(*x1.args)
        optim.tell(x1, y1)

        if (iter % 300 == 0):
            recommendation = optim.recommend()
            print(tool, iter, gen_image(*recommendation.args))
            best = build_image(*recommendation.args)
            show("best", best)
            cv2.waitKey(1)
    
cv2.waitKey(0)

