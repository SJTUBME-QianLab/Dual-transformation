import os
import numpy as np
import nibabel as nib

img_path_nii = './nii/image/'
mask_path_nii = './nii/mask/'

img_path_npy = './npy/image/'
mask_path_npy = './npy/mask/'

if not os.path.exists(img_path_npy):
    os.makedirs(img_path_npy)
    os.makedirs(mask_path_npy)

files = os.listdir(img_path_nii)
for volumeID in files[:1]:
    print('Process: ', volumeID)
    if volumeID.endswith('arterial.nii.gz'):
        npy_name = volumeID[:-16] + 'a'
    else:
        npy_name = volumeID[:-16] + 'v'

    img_nii = nib.load(img_path_nii + volumeID)
    mask_nii = nib.load(mask_path_nii + volumeID)
    affine = img_nii.affine

    img = np.transpose(img_nii.get_fdata(), (2, 1, 0))
    mask = np.transpose(mask_nii.get_fdata(), (2, 1, 0))

    np.save(img_path_npy + npy_name + '.npy', img.astype('uint8'))
    np.save(mask_path_npy + npy_name + '.npy', mask.astype('uint8'))
    print(mask.min(), mask.max())
    # print('File ' + volumeID + ' mask is saved.', mask.shape)
