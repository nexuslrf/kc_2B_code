from scipy.misc import imread,imsave,imresize
img=imread('233.jpg')
print img.dtype, img.shape
img_tinted = img * [1, 0.6, 0.9]

# Resize the tinted image to be 300 by 300 pixels.
img_tinted = imresize(img_tinted, (300, 300))

# Write the tinted image back to disk
imsave('233_tinted.jpg', img_tinted)
