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
img = Image.open("C:/Users/penro/Pictures/AIpy/No6.png")


#画像から文字を読み込む
builder = pyocr.builders.TextBuilder(tesseract_layout=3)
#img.save("10.jpg")
text = tool.image_to_string(img, lang="eng", builder=builder)

print(type(text),len(text))
print(text)

