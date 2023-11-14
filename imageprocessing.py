#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from numpy import asarray
from numpy import array
from scipy.linalg import svd
image = Image.open(r"C:\Users\sidar\OneDrive\Documents\sunflower.jpeg")

data = np.array(image)
print(data.shape)
plt.imshow(data)
plt.show()
plt.close()

gray_image = 0.289 * data[:,:,0] +  0.587 * data[:,:,1] +  0.114 * data[:,:,2]  
plt.imshow(gray_image, cmap = 'gray')


# In[ ]:


# original is 853, 640
nr = np.zeros((640,853))
for x in range(0,853) :
    for y in range(0,640) :
        [nx,ny] = np.floor([x,y] @ get_rot_matrix(90)).astype(int)
        
        nr[nx,ny] = gray_image[x,y]
plt.imshow(nr, cmap = "gray")        


# In[ ]:


# (0,640), (853,640),(853,0)(0,0), These are the dimensions of the corners. The objective here is to get the new corners.
# Its needed because since we are rotating by 40 degrees, the x and y of the corners aren't being swapped and switched, but will
# will have values that cannot be determined just by knowing what the original is. The dimensions of the new that fits the rotation
# won't be the same.
[nx,ny] = np.floor([0,640] @ get_rot_matrix(40)).astype(int)
print([nx,ny])
[nx,ny] = np.floor([853,640] @ get_rot_matrix(40)).astype(int)
print([nx,ny])
[nx,ny] = np.floor([853,0] @ get_rot_matrix(40)).astype(int)
print([nx,ny])
[nx,ny] = np.floor([0,0] @ get_rot_matrix(40)).astype(int)
print([nx,ny])


# In[ ]:


ydim = (490 - (-549))
xdim = (1064)
ymin = -549
xmin = 0
nr40 = np.zeros((xdim, ydim))
# a new empty matrix is made using dimensions of 0s matrix that is made bigger to fit. 

for x in range(0,853):
    for y in range(0,640):
# this gives the new positions respective of a 40 degree rotation
        [nx,ny] = np.floor([x,y] @ get_rot_matrix(40)).astype(int) 
        nx = nx - xmin
        ny = ny - ymin
        nr40[nx,ny] = gray_image[x,y]
plt.imshow(nr40, cmap = "gray")


# In[ ]:


# original is 853, 640
nr = np.zeros((640,853))
for x in range(0,853) :
    for y in range(0,640) :
        [nx,ny] = np.floor([x,y] @ get_rot_matrix(90)).astype(int)
             
        nr[nx,ny] = gray_image[x,y]
plt.imshow(nr, cmap = "gray")    


# In[ ]:


# contrast

print(np.mean(gray_image),np.min(gray_image),np.max(gray_image))
plt.imshow(gray_image )
plt.show()
plt.close()
#ni = new image
# The Threshold is what changes the contrast. Higher threshold means higher contrast
threshold = 80
ni = (gray_image - threshold)/(np.max(gray_image-threshold))

ni = np.where(ni<0,0,ni)
# It iterates through the positions and does if statement of whether the value of the position is less than 0,
# replace with 0, if its not 0, the value is just ni.
plt.imshow(ni)
plt.show()
plt.close()


# In[ ]:


def get_rot_matrix(degrees):
    theta = np.radians(degrees)
    return np.round([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]],decimals = 3)
get_rot_matrix(40)


# In[ ]:


print(get_rot_matrix(90))

x,y = 10, 20
[new_x, new_y] = [x,y] @ get_rot_matrix(90)

print(new_x, new_y)


# In[2]:


# decompasing data into U,S,VT, this is an approximation, and it makes it possible without taking super long
U, S, VT = np.linalg.svd(data[:,:,0],full_matrices=False)
print(S.shape)

# takes list 4 values and creates 4x4 matrix where diagonal is the values.
Sd = np.diag(S)


r = int(np.argwhere(S<(S[0]/100))[0])
# S[0] is the largest singular value
#returns amount of indexes having value smaller than S[0]/100
# The singular values below are cut off

X1r = U[:,:r] @ Sd[0:r,:r] @ VT[:r,:]
# takes advantage of svd to approximate original data and takes small fraction of original data
# Its putting the SVD formula with each matrix limited by r. It takes the first r columns of u, r x r of sigma, 
# and r columns of transpose


X1 = U @ Sd @ VT
# restate formula of SVD with new values. Much of the values can be thrown out without affecting data too much.


# conditional number computed
s1 = np.max(S)
cd = np.max(S)/np.min(S)
print(cd)


# shows original
plt.imshow(data[:,:,0])
plt.show()
plt.close()

#printing r isn't entirely necessary, but it shows the amount of singular values used, which can be a lot. 
print(r)
plt.imshow(X1r)
plt.show()
plt.close()

# does everything with different values below nth decimal of s[0] being removed. nth is 1/ n. ex. 1/10 = 0.1

r = int(np.argwhere(S<(S[0]/10))[0])
#returns index of everything smaller than 0
X1r = U[:,:r] @ Sd[0:r,:r] @ VT[:r,:]
X1 = U @ Sd @ VT
print(r,str(1/10))
plt.imshow(X1r)
plt.show()
plt.close()


r = int(np.argwhere(S<(S[0]/2))[0])
#returns index of everything smaller than 0
X1r = U[:,:r] @ Sd[0:r,:r] @ VT[:r,:]
X1 = U @ Sd @ VT
print(r,str(1/2))
plt.imshow(X1r)
plt.show()
plt.close()

r = int(np.argwhere(S<(S[0]/22250))[0])
#returns index of everything smaller than 0
X1r = U[:,:r] @ Sd[0:r,:r] @ VT[:r,:]
X1 = U @ Sd @ VT
print(r)
plt.imshow(X1r)
plt.show()
plt.close()

r = int(np.argwhere(S<(S[0]/1.5))[0])
#returns index of everything smaller than 0
X1r = U[:,:r] @ Sd[0:r,:r] @ VT[:r,:]
X1 = U @ Sd @ VT
print(r,str(1/1.5))
plt.imshow(X1r)
plt.show()
plt.close()

r = int(np.argwhere(S<(S[0]/1))[0])
#returns index of everything smaller than 0
X1r = U[:,:r] @ Sd[0:r,:r] @ VT[:r,:]
X1 = U @ Sd @ VT
print(r,str(1/1))
plt.imshow(X1r)
plt.show()
plt.close()

r = int(np.argwhere(S<(S[0]/5))[0])
#returns index of everything smaller than 0
X1r = U[:,:r] @ Sd[0:r,:r] @ VT[:r,:]
X1 = U @ Sd @ VT
print(r, str(1/5))
plt.imshow(X1r)
plt.show()
plt.close()

r = int(np.argwhere(S<(S[0]/2.5))[0])
#returns index of everything smaller than 0
X1r = U[:,:r] @ Sd[0:r,:r] @ VT[:r,:]
X1 = U @ Sd @ VT
print(r, str(1/2.5))
plt.imshow(X1r)
plt.show()
plt.close()

r = int(np.argwhere(S<(S[0]/3.33))[0])
#returns index of everything smaller than 0
X1r = U[:,:r] @ Sd[0:r,:r] @ VT[:r,:]
X1 = U @ Sd @ VT
print(r, str(1/3.33))
plt.imshow(X1r)
plt.show()
plt.close()

r = int(np.argwhere(S<(S[0]/4))[0])
#returns index of everything smaller than 0
X1r = U[:,:r] @ Sd[0:r,:r] @ VT[:r,:]
X1 = U @ Sd @ VT
print(r, str(1/4))
plt.imshow(X1r)
plt.show()
plt.close()

r = 640
#returns index of everything smaller than 0
X1r = U[:,:r] @ Sd[0:r,:r] @ VT[:r,:]
X1 = U @ Sd @ VT
print(r)
plt.imshow(X1r)
plt.show()
plt.close()

