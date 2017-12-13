# data-mining-task
import numpy as np
import jieba
import jieba.posseg as pseg 
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn import svm
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

#读数据
fh=open(r"C:\Users\acer\Desktop\带标签短信.txt",'r')
data=[]
for i in range(800):
    line=fh.readline()
    data.append(line)


fh.close()

#将短信数据存成标签列表和内容列表
content=[]
label=[]
for i in range(800):
    message=data[i].split("\t")
    label.append(message[0])
    content.append(message[1])



#分词 去掉符号 再分词 存成字符串列表
wordlist=[]
for i in range(800):
    words=pseg.cut(content[i])
    new_content="".join(w.word for w in words if w.flag!='x')
    words=jieba.cut(new_content)
    strword=" ".join(words).encode('unicode-escape').decode('string_escape')
    wordlist.append(strword)

#训练集测试集随机分割
training_data, test_data, training_target, test_target = train_test_split(wordlist, label, test_size=0.1,random_state=0)

#生成tfidf矩阵

vectorizer=CountVectorizer()#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频 
transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值  
train_tfidf=transformer.fit_transform(vectorizer.fit_transform(training_data))#第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵  
test_tfidf=transformer.transform(vectorizer.transform(test_data))
train_weight=train_tfidf.toarray()#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重  
test_weight=test_tfidf.toarray()

#降维（有问题）


pca = PCA(n_components=40)
pca.fit(train_weight)
tra_weight_tran=pca.fit_transform(train_weight)
tes_weight_tran=pca.fit_transform(test_weight)

        

#训练
clf=svm.SVC(C=50,kernel='linear')
clf.fit(tra_weight_tran,training_target)
#预测
predicted = clf.predict(tes_weight_tran)

#评估
def elevate_result(label,pred):
    """
    函数说明: 对分类器预测的结果进行评估，包括accurancy,precision,recall,F-score
    Parameter:
        label - 真实值
        pred - 预测值
    Return:
        None
   """
    con_mat = metrics.confusion_matrix(label,pred)
    TP = float(con_mat[1,1])
    TN = float(con_mat[0,0])
    FP = float(con_mat[0,1])
    FN = float(con_mat[1,0])
    
    accurancy = (TP+TN)/(TP+TN+FN+FP)
    precison = TP/(TP+FP)
    recall = TP/(TP+FN)
    beta = 1
    F_score = (1+pow(beta,2))*precison*recall/(pow(beta,2)*precison+recall)
    
    
    print("accurancy: %s \nprecison: %s \nrecall: %s \nF-score: %s" % (accurancy,precison,recall,F_score))


label=test_target
pred=predicted
pred=pred.tolist()
elevate_result(label,pred)
