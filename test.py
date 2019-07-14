#! /usr/bin/env python 
# -*- coding: utf-8 -*-
#Строки для корректной работы с киррилицей в python. Работают даже в закоментированном состоянии

from skimage.future import graph
from skimage import data, segmentation, color, filters, io
from matplotlib import pyplot as plt


#img = data.coffee()
img=io.imread('/home/vector/Documents/Скулы на уровне глаз/rsaBUvuFsFw_aligned.jpg')
gimg = color.rgb2gray(img)

labels = segmentation.slic(img, compactness=30, n_segments=1000)
edges = filters.sobel(gimg)
edges_rgb = color.gray2rgb(edges)

g = graph.rag_boundary(labels, edges)
lc = graph.show_rag(labels, g, edges_rgb, img_cmap=None, edge_cmap='viridis',
                    edge_width=1)

plt.colorbar(lc, fraction=0.1)
io.show()

"""

from skimage import data, segmentation, color
from skimage.future import graph
from matplotlib import pyplot as plt


img = data.coffee()

labels1 = segmentation.slic(img, compactness=30, n_segments=400)
out1 = color.label2rgb(labels1, img, kind='avg')

g = graph.rag_mean_color(img, labels1, mode='similarity')
labels2 = graph.cut_normalized(labels1, g)
out2 = color.label2rgb(labels2, img, kind='avg')

fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(6, 8))

ax[0].imshow(out1)
ax[1].imshow(out2)

for a in ax:
    a.axis('off')

plt.tight_layout()
"""