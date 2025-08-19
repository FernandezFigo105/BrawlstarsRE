import cv2

image = cv2.imread("C:/Users/Figo/Desktop/BrawlStars/dataset/annotated_dataset/images/train/screenshot_1.png")
resized_image = cv2.resize(image,(448,448))

cv2.imwrite("resized_img2.png",resized_image)