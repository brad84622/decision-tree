from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import csv
import numpy as np
# 開啟 CSV 檔案
with open('train.csv', newline='') as csvfile:

  # 讀取 CSV 檔內容，將每一列轉成一個 dictionary
  rows = csv.DictReader(csvfile)
  trainx=np.zeros((2000,6))
  trainy=np.zeros(2000)
  a=0
  for row in rows:
      trainx[a][0]=row['sex']
      trainx[a][1]=row['height']
      trainx[a][2]=row['weight']
      trainx[a][3]=row['attend']
      trainx[a][4]=row['exam']
      trainx[a][5]=row['hw']
      trainy[a]=row['pass'] #0代表過
      a=a+1    


# 建立分類器
tree = DecisionTreeClassifier()
tree.fit(trainx,trainy)

target_name=['pass','not pass']
export_graphviz(tree,out_file='tree.dot',feature_names=['sex', 'height', 'weight','attend','exam','hw'],class_names=target_name,filled=True,rounded=True,
                     special_characters=True)

