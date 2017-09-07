import cv2
import numpy as np
import pytesseract
import tempfile
import os
from PIL import Image

src_path = "C:/Development/projects/dataset/v1/"

IMAGE_SIZE = 1800
BINARY_THREHOLD = 180


def process_image_for_ocr(file_path):
    # TODO : Implement using opencv
    temp_filename = set_image_dpi(file_path)
    im_new = remove_noise_and_smooth(temp_filename)
    return im_new


def set_image_dpi(file_path):
    im = Image.open(file_path)
    length_x, width_y = im.size
    factor = max(1, int(IMAGE_SIZE / length_x))
    size = factor * length_x, factor * width_y
    # size = (1800, 1800)
    im_resized = im.resize(size, Image.ANTIALIAS)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_filename = temp_file.name
    im_resized.save(temp_filename, dpi=(300, 300))
    return temp_filename


def image_smoothening(img):
    ret1, th1 = cv2.threshold(img, BINARY_THREHOLD, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th2, (1, 1), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3


def remove_noise_and_smooth(file_name):
    img = cv2.imread(file_name, 0)
    filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 3)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    img = image_smoothening(img)
    or_image = cv2.bitwise_or(img, closing)
    return or_image


def original(img_path, conf):
    result = pytesseract.image_to_string(Image.open(src_path + img_path), config=conf)
    return result


def tricks(img_path, conf):
    img = process_image_for_ocr(src_path + img_path)
    # img = cv2.imread()
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # kernel = np.ones((1, 1), np.uint8)
    # img = cv2.dilate(img, kernel, iterations=10)
    # img = cv2.erode(img, kernel, iterations=10)
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    cv2.imwrite(src_path + 'upd/' + img_path + "_updated.png", img)
    result = pytesseract.image_to_string(Image.open(src_path + 'upd/' + img_path + "_updated.png"), config=conf)
    return result


for filename in os.listdir(src_path):
    f = open(filename + ".txt", 'w', encoding='utf8')
    for psm in [1, 3, 4, 5, 6]:
        for oem in range(0, 3):
            for lang in ['eng']:
            # for lang in ['eng', 'rus', 'eng+rus', 'rus+eng']:
                conf = '-psm {} -oem {} -l {} --tessdata-dir "C:/Program Files (x86)/Tesseract-OCR/tessdata/"'.format(
                    psm, oem, lang)
                f.write(conf + '\n')
                f.write(original(filename, conf) + '\n')
                # f.write("---updated image------" + '\n')
                # f.write(tricks(filename, conf) + '\n')
                f.flush()
    f.close()
