import os
from PIL import Image
import pyocr

#インストールしたTesseract-OCRのパスを環境変数「PATH」へ追記する。
#OS自体に設定してあれば以下の2行は不要
path='/usr/bin/tesseract'
os.environ['PATH'] = os.environ['PATH'] + path

#pyocrへ利用するOCRエンジンをTesseractに指定する。
tools = pyocr.get_available_tools()
print(tools[0].get_name())
tool = tools[0]

#OCR対象の画像ファイルを読み込む
img = Image.open("C:/Users/penro/Pictures/AIpy/01.png")


#画像を読みやすいように加工。
img=img.convert('RGB')
size=img.size
img2=Image.new('RGB',size)

border=110

for x in range(size[0]):
    for y in range(size[1]):
        r,g,b=img.getpixel((x,y))
        if r > border or g > border or b > border:
            r = 255
            g = 255
            b = 255
        img2.putpixel((x,y),(r,g,b))

#画像から文字を読み込む
builder = pyocr.builders.TextBuilder(tesseract_layout=3)
img2.save("20.jpg")
text = tool.image_to_string(img2, lang="eng", builder=builder)

print(type(text),len(text))
print(text)

