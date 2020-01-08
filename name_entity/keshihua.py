

with open('./test_report.txt','r',encoding='utf-8') as f:
    data = f.readlines()
    f.close()

words = []
for line in data:
    line = line.split(',')
    line = line[1]
    line = line.split(';')
    for w in line:
        w = w.split('\t')
        w = w[0]
        words.append(w)
print(words)

from wordcloud import WordCloud
import matplotlib.pyplot as plt
jieba_space_split = ' '.join(words)
backgroud_Image = plt.imread('./woman.jpg')
wc = WordCloud(background_color='white',
               mask= backgroud_Image,
               font_path='./SimHei.ttf')

wc.generate_from_text(jieba_space_split)
plt.imshow(wc)
plt.show()














