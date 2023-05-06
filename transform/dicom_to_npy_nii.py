import numpy as np
import matplotlib.pyplot as plt
import os
import SimpleITK as sitk
import pydicom
import nibabel as nib
np.set_printoptions(threshold=np.inf)
# 预处理数据，对原始的CT图像调整窗宽窗位


def window_transform(ct_array, windowWidth, windowCenter, normal=False):
    """
    return: trucated image according to window center and window width
    and normalized to [0,1]
    """
    minWindow = float(windowCenter) - 0.5 * float(windowWidth)
    newimg = (ct_array - minWindow) / float(windowWidth)
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    if not normal:
        # newimg = (newimg * 255).astype('uint8')
        newimg = (newimg * 255).astype('float32')
    return newimg


windowWidth = 340
windowCenter = 70  # [-100,240]

# ######### 将DICOM影像存为.npy,并保存为nii ################
path = 'F:/ruijin Lymph node/original_data/first-clinical model/Segmentation/'
maskpath = 'F:/ruijin_Lymph_node/nii/1/mask/'
newpath = 'F:/ruijin_Lymph_node/npy/1/image/'
newniipath = 'F:/ruijin_Lymph_node/nii/1/image/'
newmaskpath = 'F:/ruijin_Lymph_node/npy/1/mask/'
names = os.listdir(path)

h = 512
w = 512
if not os.path.exists(newpath):
    os.makedirs(newpath)
    os.makedirs(newmaskpath)
    os.makedirs(newniipath)
for volumeID_number in range(len(names)):  # names:
    volumeID = names[volumeID_number]
    # npy image
    files = sorted(os.listdir(path + volumeID))
    print('Processing File ' + volumeID)
    c = len(files)
    data = np.zeros([c, h, w])
    instancenumber=[]
    for file in files:
        file1 = os.path.join(path, volumeID, file)
        image = pydicom.read_file(file1, force=True)
        sliceID = image.data_element("InstanceNumber").value
        instancenumber.append(sliceID)
    instancenumber_max = max(instancenumber)
    instancenumber_min = min(instancenumber)
    print(instancenumber_max,instancenumber_min)
    for file in files:
        file1 = os.path.join(path, volumeID, file)
        image = pydicom.read_file(file1)
        sliceID = image.data_element("InstanceNumber").value
        if image.pixel_array.shape[0] != 512 or image.pixel_array.shape[1] != 512:
            exit('Error: DICOM image does not fit ' + str(w) + 'x' + str(h) + '?size!')
        data[c - sliceID + (instancenumber_min - 1), :, :] = image.pixel_array + image.data_element(
            "RescaleIntercept").value  # np.flip(image.pixel_array,0)

    print(data.max(),data.min())
    img_new = window_transform(data, windowWidth, windowCenter)
    np.save(newpath+ volumeID+'.npy', img_new)
    print('File ' + volumeID + ' is saved.', img_new.shape,img_new[c//2,200,200])

    # nii mask
    if volumeID.endswith('a'):
        nii_name = volumeID[:-1]+'_arterial'
    else:
        nii_name = volumeID[:-1] + '_venous'
    mask = nib.load(maskpath + nii_name + '.nii.gz')
    print(mask.shape)
    affine = mask.affine
    mask = np.transpose(mask.get_fdata(), (2, 1, 0))

    # npy mask
    # print(mask.min(),mask.max())
    np.save(newmaskpath + volumeID + '.npy', mask.astype('uint8'))
    print(mask.min(), mask.max())
    print('File ' + volumeID + ' is saved.', mask.shape, mask[c//2, 200, 200])

    # nii image
    img_new = np.transpose(img_new, (2, 1, 0))
    img_new_nii = nib.Nifti1Image(img_new, affine)
    nib.save(img_new_nii, newniipath + nii_name + '.nii.gz')
