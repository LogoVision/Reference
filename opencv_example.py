import cv2
import numpy as np
import random as rd

path = "./toss_logo.png"

image = cv2.imread(path,cv2.IMREAD_ANYCOLOR)
cv2.imshow("logo", image)

def size_256(image:cv2.UMat) -> cv2.UMat:
    """짧은 변의 길이를 256으로 설정"""
    height, width, channel = image.shape
    if height>width:
        k=256/width
    else:
        k=256/height
    image2 = cv2.resize(image, dsize=(0,0),fx=k,fy=k,interpolation=cv2.INTER_LINEAR)
    return image2

def cut_image(image:cv2.UMat)->tuple:
    """긴 변의 길이를 256으로 자르기"""
    dst = list()
    height, width, channel = image.shape
    if height == 256:
        for i in range(width-255):
            dst.append(image[0:256, i:i+256].copy())
    else:
        for i in range(height-255):
            dst.append(image[i:i+256, 0:256].copy())
    return dst

def rotate_image(image:cv2.UMat)->cv2.UMat:
    """회전"""
    height, width, channel = image.shape
    matrix = cv2.getRotationMatrix2D((width/2, height/2), 90, 1)
    image2 = cv2.warpAffine(image, matrix, (width, height))
    return image2

def transform_image(image:cv2.UMat)->tuple:
    """이미지 왜곡"""
    dst = list
    height, width, channel = image.shape
    x=y=rd.randint(0,125)
    x2=rd.randint(126,255)
    y2=rd.randint(0,125)
    x3=rd.randint(126,255)
    y3=rd.randint(126,255)
    x4=rd.randint(0,125)
    y4=rd.randint(126,255)
    imagePoint = np.array([[x,y],[x2,y2],[x3,y3],[x4,y4]], dtype=np.float32)
    imagePoint2 = np.array([[0,0],[width, 0], [width, height], [0, height]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(imagePoint, imagePoint2)
    return cv2.warpPerspective(image, matrix, (width, height))

def synthesis(image):
    dst = list()
    im_256 = size_256(image)
    ims = cut_image(im_256)
    for i in range(len(ims)):
        for j in range(4):
            dst.append(ims[i])
            dst.append(transform_image(ims[i]))
            ims[i] = rotate_image(ims[i])
    return dst

ims = synthesis(image)
print(len(ims))
for i in range(len(ims)):
    cv2.imshow(str(i+1),ims[i])
    cv2.waitKey()
    cv2.destroyAllWindows()