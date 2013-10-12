''' Loads image data. 
    Requires Python Imaging Library (PIL).
'''
import bnpy.data.XData as XData
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def get_data(seed=123,**kwargs):
    ''' Reads image and packages it as XData
    '''
    #if kwargs is not None:
    #impath = kwargs['impath']; 
    impath = '/Users/sghosh/Documents/Research/Projects/Data/Images/Images/8023.jpg'
    try:
        im = np.array(Image.open(impath))
        return XData(np.reshape(im,(-1,3))) 
    except IOError as e:
        print "I/O error({0}): {1}".format(e.errno, e.strerror)