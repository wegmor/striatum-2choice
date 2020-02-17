import pandas as pd
import numpy as np

def roiPlot(rois, colors, axis):
    rois = np.array([rois[n].unstack('x').values for n in rois])
    if len(rois) != len(colors):
        raise ValueError("Colors must have the same length as rois.")
    rs = []
    for roi, color in zip(rois, colors):
        roi /= roi.max()
        roi = roi**1.5
        roi = np.clip(roi-.1, 0, .85)
        roi /= roi.max()
        r = np.array([(roi > 0).astype('int')]*3) * color[:, np.newaxis, np.newaxis]
        r = np.concatenate([r, roi[np.newaxis]], axis=0)
        rs.append(r.transpose((1,2,0)))    
    rs = np.array(rs)

    for img in rs:
        axis.imshow(img)