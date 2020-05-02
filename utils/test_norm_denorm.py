import cv2
from utils.img_proc import *

imgpath = "/home/igor/datasets/VOC_2012/trainval/JPEGImages/2007_000250.jpg"

image_src = cv2.imread(imgpath)  # read yxc opencv image with bgr color sequence
cv2.imshow("Source", image_src)
cv2.waitKey(10)

image = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)  # reverse bgr to rgb
image = img_yxc2cyx(image)

norm_img = normalize_img_cyx(image)
denorm_img = denormalize_img_cyx(norm_img)

out_image = img_cyx2yxc(denorm_img)
out_image = cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)  # reverse bgr to rgb
cv2.imshow("Out", out_image)
cv2.waitKey(1000)

print("Completed.")
