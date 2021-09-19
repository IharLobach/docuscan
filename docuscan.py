import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
from skimage.filters import threshold_local

with open("config.json") as f:
    conf = json.load(f)

if len(sys.argv) != 2:
    raise ValueError("There should be one argument --- file name of the image!")

filename = sys.argv[1]
print(filename)

image = cv2.imread(filename)
if image.shape[0] < image.shape[1]:
    image = cv2.rotate(image, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# reducing image size and shifting intensity by -127.5
src = gray
scale = conf['resolution_reduction']
width = int(src.shape[1] * scale)
height = int(src.shape[0] * scale)
dsize = (width, height)
reduced = cv2.resize(src, dsize) - 127.5

# applying 4 filters (which find edges) to 4 quadrants
a = int(conf['bubble_size'] * reduced.shape[1])
zeros = np.zeros(shape=(a, a))
ones = np.ones(shape=(a, a))
ll = 1/a**2/6*np.concatenate([
    np.concatenate((-ones, -ones), axis=1),
    np.concatenate((-ones, 3*ones), axis=1)])
i_half_max = int(reduced.shape[0]/2)+1
j_half_max = int(reduced.shape[1]/2)+1
ll_im = cv2.filter2D(reduced[:i_half_max, :j_half_max], -1, ll)
lh = 1/a**2/6*np.concatenate([
    np.concatenate((-ones, -ones), axis=1),
    np.concatenate((3*ones, -ones), axis=1)])
lh_im = cv2.filter2D(reduced[:i_half_max, j_half_max:], -1, lh)
hh = 1/a**2/6*np.concatenate([
    np.concatenate((3*ones, -ones), axis=1),
    np.concatenate((-ones, -ones), axis=1)])
hh_im = cv2.filter2D(reduced[i_half_max:, j_half_max:], -1, hh)
hl = 1/a**2/6*np.concatenate([
    np.concatenate((-ones, 3*ones), axis=1),
    np.concatenate((-ones, -ones), axis=1)])
hl_im = cv2.filter2D(reduced[i_half_max:, :j_half_max], -1, hl)
ijs0 = [np.unravel_index(np.argmax(xx_im), xx_im.shape)
        for xx_im in [ll_im, lh_im, hh_im, hl_im]]
didjs = [(0, 0), (0, j_half_max), (i_half_max, j_half_max), (i_half_max, 0)]
ijs = []
xys = []
for ij, didj in zip(ijs0, didjs):
    i, j = ij
    di, dj = didj
    ri, rj = i+di, j+dj
    ijs.append(np.array([ri, rj]))
    xys.append(np.array([rj, ri]))


# drawing the contour
scalex = reduced.shape[1]/image.shape[1]
scaley = reduced.shape[0]/image.shape[0]
xys0 = [np.array([int(x/scalex), int(y/scaley)]) for x, y in xys]
ctr0 = np.array(xys0).reshape((-1, 1, 2)).astype(np.int32)
ctr0_im = cv2.drawContours(image.copy(), [ctr0], 0, (0, 255, 0), 20)



# cropping and transforming perspective
height = max(np.linalg.norm(xys0[1] - xys0[2]),
             np.linalg.norm(xys0[3] - xys0[0]))


width = height*8.5/11
rect = np.array([[0, 0],
                [0, width],
                [height, width],
                [height, 0]], np.float32)
xys_fl = np.array(xys0).astype(np.float32)

M = cv2.getPerspectiveTransform(xys_fl, rect)

res = cv2.warpPerspective(image, M, (int(height), int(width)))
res = cv2.flip(res, 0)
res = cv2.rotate(res, cv2.cv2.ROTATE_90_CLOCKWISE)


if conf['use_threshold']:
    # giving black-and-white feel
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    T = threshold_local(res, conf['threshold_block_size'],
                        offset=conf['threshold_offset'],
                        method="gaussian")
    res = (res > T).astype("uint8")

    # setting specified margin (line of certain thickness) to white pixels
    imax, jmax = res.shape
    margin = 0.01
    mar = int(imax * margin)
    mar_fi = np.ones(shape=res.shape)
    mar_fi[mar:-mar, mar:-mar] = 0
    res = ((mar_fi + res) > 0).astype("uint8") * 255



# writing result to file
base_name, ext = filename.split(".")
res_name = base_name + "_cropped." + ext

cv2.imwrite(res_name, res)


# showing the contour in matplotlib
if conf['show_contour']:
    fig, ax = plt.subplots()
    plt.imshow(ctr0_im)
    for x, y in xys0:
        circle1 = plt.Circle((x, y), a/scale, color='red', fill=False)
        ax.add_patch(circle1)
    plt.axis('off')
    plt.show()
