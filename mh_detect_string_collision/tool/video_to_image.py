
import cv2
import time

video = cv2.VideoCapture('/home/kai/Documents/DATASET/minghong/ps_data/video/IMG_1547.MOV')
count = 0
while True:
    ret, image = video.read()
    count += 1
    if not ret:
        break
    if count % 30 == 0:
        # cv2.imwrite(f'/home/kai/Documents/DATASET/minghong/ps_data/temp/{time.time()}_1.jpg', image[0:304,0:image.shape[1]])
        # cv2.imwrite(f'/home/kai/Documents/DATASET/minghong/ps_data/temp/{time.time()}_2.jpg', image[304:608,0:image.shape[1]])
        # cv2.imwrite(f'/home/kai/Documents/DATASET/minghong/ps_data/temp/{time.time()}_3.jpg', image[608:912,0:image.shape[1]])
        cv2.imwrite(f'/home/kai/Documents/DATASET/minghong/ps_data/temp/{time.time()}_1.jpg', image)