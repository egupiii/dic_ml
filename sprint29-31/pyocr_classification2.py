import sys
import cv2
import pyocr
import pyocr.builders
from PIL import Image


"""
ここが汎用性が無い。
商品ラベルの比較的大きい文字を手動で入力。
その際、句読点は入れず、文字の重複も考えなくて良いが、アルファベット等の大文字/小文字で形の似ているものは入力する。
"""
# Key words of each correct label
STR_POCARI = "IiOoNSsUuPpPpLYDRIiNKkPpOoCcARIiSsWwEATt無果汁ポカリスエット5５0０0０MmLプラPpETt"
STR_CALPIS = "ASsAHIiカルピス1１0０0０tThカルピスウォー-ター-乳酸菌と酵母発酵がもつチカラここからはがせますCcALPpIiSsPpETtプラアレルゲン(（2２7７品目中)）乳大豆ホー-ムペー-ジ"
STR_ILOHAS_NORMAL = "いろはす日本の天然水5５5５5５mMlNATtUuRALMmIiNERALWwATtERPpETtプラYesS!！リサイクルNoO!！ポイ捨て"
STR_ILOHAS_PEACH = "いろはすもも山梨県産白桃エキス入り無果汁PpETtプラYesS!！リサイクルNoO!！ポイ捨て"
STR_TROPICANA = "TtroOpPiIcCanaREALFfRUuIiTtEXxPpERIiENCcE1１0０0０%％オレンジPpETtプラCcOoLD&＆IiCcE冷やしても凍らせてもおいしい"


# List of the character strings
list_str = [STR_POCARI, STR_CALPIS, STR_ILOHAS_NORMAL, STR_ILOHAS_PEACH, STR_TROPICANA]


# Split the character strings into each character
key_words = []
for string in list_str:
    split_str = list(string)
    key_words.append(list(set(split_str)))


def classify_products(input_path):
# def classify_products(img):
    # Image to text
    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        sys.exit(1)
    tool = tools[0]

    # Original image
    txt = tool.image_to_string(
        Image.open(input_path),
#         img,
        lang="jpn+eng",
        builder=pyocr.builders.TextBuilder(tesseract_layout=6)
    )

    # Make a few adjustments to the original image
    im = cv2.imread(input_path)
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_blur = cv2.GaussianBlur(im_gray, (5, 5), 0)
    th1 = cv2.adaptiveThreshold(im_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    txt_gray = tool.image_to_string(
        Image.fromarray(im_gray),
        lang = 'eng+jpn'
    )

    txt_blur = tool.image_to_string(
        Image.fromarray(im_blur),
        lang = 'eng+jpn'
    )

    txt_th1 = tool.image_to_string(
        Image.fromarray(th1),
        lang = 'eng+jpn'
    )

    # Change the texts to a list of each character
    results_pyocr = []
    results_pyocr.append(list(txt))
    results_pyocr = [x for x in results_pyocr[0] if x]
    
    if len(results_pyocr) == 0:
        return None

    # Count key words and return indix of the label
    count_key_words = []
    for i in range(len(key_words)):
        total = 0
        for result in results_pyocr[0]:
            if result in key_words[i]:
                total += 1
        count_key_words.append(total)
    index = [i for i, x in enumerate(count_key_words) if x == max(count_key_words)]
    
    # Output index
    if len(index) == 1:
        ratio = count_key_words[index[0]]/sum(count_key_words)
        if ratio >= 0.5:
            return index[0]
        else:
            return None
    else:
        return None