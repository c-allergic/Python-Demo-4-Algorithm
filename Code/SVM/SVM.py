import kagglehub
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
import nltk
nltk.download(['stopwords', 'wordnet', 'omw-1.4'])
# Download latest version
path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
df = pd.read_csv(path + '/IMDB Dataset.csv')
print("Path to dataset files:", path)
import joblib
from datetime import datetime

def clean_text(text):
    # 去除HTML标签
    text = re.sub(r'<[^>]+>', ' ', text)
    # 保留字母和基本标点
    text = re.sub(r"[^a-zA-Z.!?']", ' ', text)
    # 合并连续空格
    text = re.sub(r'\s+', ' ', text)
    # 转换为小写
    text = text.lower()
    return text.strip()

# 文本预处理（词形还原+去停用词）
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    cleaned = clean_text(text)
    # 分词处理
    words = cleaned.split()
    # 词形还原并去除停用词
    processed = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(processed)

print("Start preprocessing...")
tqdm.pandas()  # 激活Pandas的进度条支持
df['processed_review'] = df['review'].progress_apply(preprocess)  # 使用tqdm可显示进度条

# --------------------------------------------------
# 2. Feature Engineering
# --------------------------------------------------
# 使用TF-IDF向量化文本（限制特征维度）
tfidf = TfidfVectorizer(
    max_features=8000,        # 限制特征数量以提升训练速度
    ngram_range=(1,2),       # 包含单字和双字词组
    min_df=5,                # 忽略文档频率<5的词
    max_df=0.7               # 忽略出现在70%以上文档中的词
)

# 划分训练测试集（若数据集未预分割）
X = df['processed_review']
y = df['sentiment'].map({'positive':1, 'negative':0})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------------------------------
# 3. SVM Pipeline
# --------------------------------------------------
svm_pipeline = Pipeline([
    ('tfidf', tfidf),
    ('clf', LinearSVC(
        C=0.5,
        class_weight='balanced',
        random_state=42,
        max_iter=1000
    ))
])

# --------------------------------------------------
# 4. Model Training
# --------------------------------------------------
print("\nTraining model...")
svm_pipeline.fit(X_train, y_train)

# --------------------------------------------------
# 5. Model Evaluation
# --------------------------------------------------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # 输出样例预测
    sample_texts = [
        "This movie was absolutely fantastic! The acting was superb.",
        "A complete waste of time. Terrible plot and bad acting.",
        "The cinematography was good, but the story was boring.",
        "I'm not sure if I can recommend this movie to others. It was okay, but not great.",
        "The movie was great! I loved the plot and the acting.",   
    ]
    print("\nSample Predictions:")
    for text in sample_texts:
        prob = model.decision_function([text])[0]
        pred = "Positive" if model.predict([text])[0] == 1 else "Negative"
        print(f"Text: {text}\nPrediction: {pred} (Confidence: {abs(prob):.2f})\n")

evaluate_model(svm_pipeline, X_test, y_test)

# --------------------------------------------------
# 6. 保存模型
# --------------------------------------------------
# 保存训练好的模型到文件
model_path = f"sentiment_svm_model_{datetime.now().strftime('%Y%m%d')}.joblib"
joblib.dump(svm_pipeline, model_path)
print(f"\n模型已保存到: {model_path}")
