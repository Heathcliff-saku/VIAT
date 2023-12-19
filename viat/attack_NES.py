from rendering_image import render_image
import numpy as np
from PIL import Image
from NES import NES_search
#from xNES import xNES_search
from NES_GMM import NES_GMM_search

from datasets.opts import get_opts

args = get_opts()

if args.optim_method == 'NES':
    #NES_search()
    NES_GMM_search()
else:
    xNES_search()