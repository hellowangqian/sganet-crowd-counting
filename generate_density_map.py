import numpy as np
import scipy.spatial
import pdb
def generate_multi_density_map(shape=(5,5),points=None,f_sz=15,sigma=4,num=3):
    '''
    generate multiple density maps according to the density
    '''
    # calculate the distance of each point to the nearest neighbour
    dist = scipy.spatial.distance.cdist(points,points,metric='euclidean')
    dist.sort()
    k = 3
    f_sz_vec = [15,15,15]
    meanDist = dist[:,1:k+1].mean(axis=1)
    thresholds = np.array([[0,20],[20,50],[50,1e9]])
    density_map = np.zeros((num,shape[0],shape[1]))
    for i in range(num):
        selector = (meanDist>thresholds[i,0]) & (meanDist<=thresholds[i,1])
        points_subset = points[selector,:]
        density_map[i,] = generate_density_map(shape,points_subset,f_sz_vec[i],sigma)
    return density_map
        
def generate_density_map(shape=(5,5),points=None,f_sz=15,sigma=4):
    """
    generate density map given head coordinations
    """
    im_density = np.zeros(shape[0:2])
    h, w = shape[0:2]
    if len(points) == 0:
        return im_density
    for j in range(len(points)):
        H = matlab_style_gauss2D((f_sz,f_sz),sigma)
        x = np.minimum(w,np.maximum(1,np.abs(np.int32(np.floor(points[j,0])))))
        y = np.minimum(h,np.maximum(1,np.abs(np.int32(np.floor(points[j,1])))))
        if x>w or y>h:
            continue
        x1 = x - np.int32(np.floor(f_sz/2))
        y1 = y - np.int32(np.floor(f_sz/2))
        x2 = x + np.int32(np.floor(f_sz/2))
        y2 = y + np.int32(np.floor(f_sz/2))
        dfx1 = 0
        dfy1 = 0
        dfx2 = 0
        dfy2 = 0
        change_H = False
        if x1 < 1:
            dfx1 = np.abs(x1)+1
            x1 = 1
            change_H = True
        if y1 < 1:
            dfy1 = np.abs(y1)+1
            y1 = 1
            change_H = True
        if x2 > w:
            dfx2 = x2 - w
            x2 = w
            change_H = True
        if y2 > h:
            dfy2 = y2 - h
            y2 = h
            change_H = True
        x1h = 1+dfx1
        y1h = 1+dfy1
        x2h = f_sz - dfx2
        y2h = f_sz - dfy2
        if change_H:
            H =  matlab_style_gauss2D((y2h-y1h+1,x2h-x1h+1),sigma)
        im_density[y1-1:y2,x1-1:x2] = im_density[y1-1:y2,x1-1:x2] +  H;
    return im_density
     
def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

