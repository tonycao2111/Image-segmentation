import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import clear_border
from skimage import measure
from skimage.measure import label,regionprops
from scipy import ndimage as ndi
from scipy.ndimage import measurements, center_of_mass, binary_dilation, zoom
import plotly.graph_objects as go


# Source:https://github.com/lukepolson/youtube_channel/blob/main/Python%20Tutorial%20Series/image_processing1.ipynb

filepath = 'input/CT_scan.npy'
img = np.load(filepath)


plt.pcolormesh(img[160])
# plt.show()

mask = img < -320
# Select out the area of air 
# print(mask)
plt.pcolormesh(mask[170])
plt.show()