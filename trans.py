from googletrans import Translator

# 英語を日本語に翻訳する
text="We must remenber that they are people, not numbers."
print(text)

Translator = Translator() #なぜいるのか分かってない
sen=Translator.translate(text, dest="ja", scr="en")
#sen=Translator.translate(text, dest="ja") でもよい。

translated=sen.text
print(translated)
