from rendering_image import render_image
import numpy as np

random = np.zeros([100, 6])
gamma = 0.0
th = np.linspace(-180, 180, 100)
phi = 0.0
r = 4.0
a = 0.0
b = 0.0

random[:, 0] = gamma
random[:, 1] = th
random[:, 2] = phi
random[:, 3] = r
random[:, 4] = a
random[:, 5] = b

render_image(random, is_over=True)