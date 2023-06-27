import numpy as np
import os
import sys
import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
import statistics
from scipy.signal import find_peaks
from scipy.signal import argrelmax
from PIL import Image
import pyocr
import time

start = time.time()


def calc_haarlike(crop_img, rect_h):
    pattern_h = rect_h // 2
    height = crop_img.shape[0]
    out = np.zeros([height-rect_h])

    for index in range(height-rect_h):
        a1 = np.mean(crop_img[index: index+pattern_h, :])
        a2 = np.mean(crop_img[index+pattern_h: index+rect_h,:])
        out[index] = a1-a2

    return out,index

def image_roto(deg,img):
    rot_angle=deg
    #画像回転処理開始
    h,w=img.shape[:2]
    ROT= cv2.getRotationMatrix2D(center=(w/2,h/2),angle=rot_angle,scale=1)
    pic1 = cv2.warpAffine(img,ROT,dsize=(w,h))
    return pic1


#文字の平均サイズを所得
def Measure_font_size(img_path):
    import statistics
    path="C:Program Files/Tesseract-OCR/tesseract.exe"
    os.environ['PATH'] = os.environ['PATH'] + path
    tools = pyocr.get_available_tools()
    print(tools[0].get_name())
    tool = tools[0]
    img = Image.open(img_path)
    box_builder = pyocr.builders.WordBoxBuilder(tesseract_layout=6)
    text_position = tool.image_to_string(img,lang="eng",builder=box_builder)
    size_list = []
    for res in text_position:
        p1 = res.position[0]
        p2 = res.position[1]
        y1 = p1[1]
        y2 = p2[1]
        size = y2 - y1
        size_list.append(size)

    return statistics.mean(size_list)

#短径領域のために文字サイズを所得
img_name = "C:/Users/penro/Pictures/AIpy/No7.png"
img = cv2.imread(img_name)
print(type(img))
test = Measure_font_size(img_name)
fonts = math.floor(test)

#最適な角度にする．
#振動数を最小にする
min_dispersion = float('inf')
max_mean = 0
adapt_deg = 0

for deg in range(-45,45,2):
    pic1 = image_roto(deg,img)

    out,index = calc_haarlike(pic1,fonts)
    peak1 = argrelmax(out)[0]
    peak2 = argrelmax(-out)[0]
    peak1_y = []
    peak2_y = []

    for (px1,px2) in zip(peak1,peak2):
        peak1_y.append(out[px1])
        peak2_y.append(out[px2])

    p1_mean = statistics.mean(peak1_y)
    p2_mean = statistics.mean(peak2_y)
    tmp_mean = p1_mean - p2_mean
    if tmp_mean > max_mean:
        max_mean = tmp_mean
        adapt_deg = deg
    # peak = argrelmax(out)[0]


    # med = np.median(peak)
    # peak[peak < med] = 0
    # dispression = np.var(peak)

    # if min_dispersion > dispression:
    # min_dispersion = dispression
    # adapt_deg = deg

print("最適角度:",adapt_deg)

adapt_img = image_roto(adapt_deg,img)
out,index = calc_haarlike(adapt_img,fonts)
#peakはy座標の値
cv2.imwrite("adapt_degree(test-img).jpg",adapt_img)
#cv2.imwrite("20.jpg",adapt_img)
out,index = calc_haarlike(adapt_img,fonts)

end = time.time()

second = end - start

print("runtime=",second)

plt.plot(out)
plt.show()