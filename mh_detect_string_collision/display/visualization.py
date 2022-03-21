import cv2

def write_image(dir, image_name, image):
    cv2.imwrite(f'{dir}/{image_name}.jpg', image)

def show_image(tab_name, image, is_iteration):
    cv2.imshow(tab_name, image)
    if is_iteration:
        cv2.waitKey(1)
    else:
        cv2.waitKey(0)
