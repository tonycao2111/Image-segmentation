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

x = img.shape
# print(x)


# [] to choose the slice
plt.pcolormesh(img[170])
# plt.colorbar()
# plt.show()


# Create a threshold mask , use -320 as the lower limit
mask = img < -320
# Select out the area of air 
print(mask)
plt.pcolormesh(mask[170])
# plt.show()

# Use clear_border function to remove the outer border
# use np.vectorize to automatically create a function that can be applied to each element of the array (loop)
# Note: only looping through the 2d array (512,512)

mask = np.vectorize(clear_border,signature='(m,n)->(m,n)')(mask)
plt.pcolormesh(mask[170])
# plt.colorbar()


# To remove the table, we can create different reagions for each of the separated volumes in the image

mask_labeled = np.vectorize(label, signature='(n,m)->(n,m)')(mask)
plt.pcolormesh(mask_labeled[150])
# plt.colorbar()
# plt.show()



# Now we only need to keep the 3 largest regions, which should be the lungs, and unfortunately the table
# Because if we take the largest 2, there are imgs that the lungs are smaller than the table, so yeah



# Quick look at the code:
# slc = mask[170]  # Select the slice at index 170
# slc.shape --> (512, 512)
# rps = regionprops(slc)  # Get properties of regions in the slice --> this is to find the area of all the regions that are
# in the slice with the regionprops.area
# areas = [r.area for r in rps]  # Get the area of each region
# np.argsort(areas)  # Sort the areas in ascending order
# np.argsort(areas)[::-1]  # Sort the areas in descending order
# idxs = np.argsort(areas)[::-1]  # Get the indices of the sorted areas in descending order


slc = mask_labeled[170]
rps = regionprops(slc)
areas = [r.area for r in rps]
idxs = np.argsort(areas)[::-1] # we want largest to smallest

# Now, we add this 3 largest regions to a new slice
new_slc = np.zeros_like(slc)

# Lets look at the largest region here, which is the table
# rps[0].coords  # This gives us the coordinates of the largest region (table)
# Now we want to index the new slice, but before that we want to transpose the coordinates to match the shape of the new slice
# rps[0].coords.T # This gives us the coordinates in the form of (x,y) which is what we want
# Turn into tuple: tuple(rps[0].coords.T)
# Now we can index what we created new_slc[tuple(rps[0].coords.T)] = i + 1 <-- this is the largest region (table) and we want to set it to 1


# Now we loop through the indices of the largest regions and add them to the new slice

    

new_slc = np.zeros_like(slc)
for i in idxs[:3]:
    new_slc[tuple(rps[i].coords.T)] = i+1

plt.pcolormesh(new_slc)
# plt.colorbar()
# plt.show()


# Now we automate this process for all slices

def keep_top_3(slc):
    new_slc = np.zeros_like(slc)
    rps = regionprops(slc)
    areas = [r.area for r in rps]
    idxs = np.argsort(areas)[::-1]
    for i in idxs[:3]:
        new_slc[tuple(rps[i].coords.T)] = i+1
    return new_slc


mask_labeled = np.vectorize(keep_top_3, signature='(n,m)->(n,m)')(mask_labeled)
# plt.pcolormesh(mask_labeled[180])
# plt.show()


# Next we fill out the holes in the lungs

# First we need to turn it into a binary array/image
mask = mask_labeled > 0
# Now we can use the binary_fill_holes function to fill out the holes in the lungs
mask = np.vectorize(ndi.binary_fill_holes, signature='(n,m)->(n,m)')(mask)

# Some slices there's still a trachea, which we also want to remove
# In a 512x512 image, the trachea typically takes up less than 0.69% of the area. 
# We can delete all regions that have any area smaller than this percentage:
# We look at where the area is less than 0.69% of the total area of the slice
# Basically get rid of the small areas like before again, we turn them into zeros and return the new slice    

def remove_trachea(slc, c=0.0069):
    new_slc = slc.copy()
    labels = label(slc,connectivity=1,background=0)
    rps = regionprops(labels)
    areas = np.array([r.area for r in rps])
    idxs = np.where(areas/512**2 < c)[0]
    for i in idxs:
        new_slc[tuple(rps[i].coords.T)] = 0
    return new_slc

mask = np.vectorize(remove_trachea, signature='(n,m)->(n,m)')(mask)

# Now the trachea is removed in the slice we were considering:

# Its time to remove the table 
# Think that every elements has a center of mass, the table would have the lowest center of mass in the y-axis
# using plt.colorbar() we can can know what number is the table
# Lets test with the right lung: plt.pcolormesh(label==3)
# using center_of_mass(label==3) we can know the exact coords of the center of mass of the right lung


# First thing we do is we copy the slice and label it:
# new_slc = slc.copy()
# labels = label(slc, background=0)

# We take the unique labels that are not equal to 0 indexing from 1 onwards because we dont want the background
# Then we compute the center of mass of each seperate regions then we loop through the corresponding indexes
# if y < 30% -> delete, if y > 60% -> delete


def delete_table(slc):
    new_slc = slc.copy()
    labels = label(slc, background=0)
    idxs = np.unique(labels)[1:]
    COM_ys = np.array([center_of_mass(labels==i)[0] for i in idxs])
    for idx, COM_y in zip(idxs, COM_ys):
        if (COM_y < 0.3*slc.shape[0]):
            new_slc[labels==idx] = 0
        elif (COM_y > 0.6*slc.shape[0]):
            new_slc[labels==idx] = 0
    return new_slc

mask_new = np.vectorize(delete_table, signature='(n,m)->(n,m)')(mask)


# Finally, we can expand the area of the lungs a little bit by growing the border. 
# For this, we can use the binary_dilation function:

mask_new = binary_dilation(mask_new, iterations=5)

# <----------------------------------------------------------------------------------------------------->


# Lets plot the full 3D image in plotly and create an interactive plot:

# First decrease the resolution a little bit: ( by 40%)

im = zoom(1*(mask_new), (0.4,0.4,0.4))

# Then we get the arrays of x, y, and z. In CT scans, the difference between the pixels in the z direction is 4 times larger than x and y
z, y, x = [np.arange(i) for i in im.shape]
z*=4

# This is for plotting in plotly



# Now we create a meshgrid 

X,Y,Z = np.meshgrid(x,y,z, indexing='ij')

# Create a 3d plotly plot

# Flatten the axis, transpose the image to make sure it fits 

fig = go.Figure(data=go.Volume(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=np.transpose(im,(1,2,0)).flatten(),
    isomin=0.1,
    opacity=0.1, # needs to be small to see through all surfaces
    surface_count=17, # needs to be a large number for good volume rendering
    ))
fig.write_html("test.html")

plt.pcolormesh(mask_new[170])
# plt.show()
