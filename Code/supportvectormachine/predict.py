import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 导入文本预处理函数（需要与训练时相同）
def clean_text(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r"[^a-zA-Z.!?']", ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    cleaned = clean_text(text)
    words = cleaned.split()
    processed = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(processed)

# 加载模型并预测新评论
model_path = "sentiment_svm_model_20250414.joblib"
loaded_model = joblib.load(model_path)

# 示例评论
new_reviews = [
    # AI generated reviews
    # 正面评论
    "This film exceeded all my expectations with its brilliant plot and outstanding performances!",
    "The director's visual style is breathtaking, every frame looks like a work of art.",
    "This is the most touching movie I've seen in the past decade, the ending made me cry.",
    "The character development was exceptional, I was completely drawn into the protagonist's journey.",
    "The soundtrack perfectly complemented the storyline, enhancing the overall viewing experience.",
    
    # 负面评论
    "I really don't understand why this movie has such high ratings, the plot has so many holes.",
    "The acting was incredibly forced and the dialogue was too awkward to continue watching.",
    "The two-hour runtime felt like a waste of my time with its painfully slow pacing.",
    "The special effects were terrible, looking like technology from a decade ago.",
    "The story lacks originality and is merely a poor imitation of other successful films.",
    
    # 中性/混合评论
    "This movie has both strengths and weaknesses; great acting but a somewhat weak script.",
    "The beginning was very engaging, but the quality declined as the plot progressed.",
    "Visually impressive, but the storyline was too complex to follow easily.",
    "The director's ambition is admirable, but the final product doesn't fully realize his vision.",
    "As a sequel in the franchise, it's not bad, but it doesn't surpass the original.",
    
    # 微妙/具体评论
    "Though the plot is formulaic, the excellent performances by the cast make up for it.",
    "This film is technically flawless but lacks emotional depth and resonance.",
    "It's a feast for fans of the genre, but general audiences might find it confusing.",
    "The third act twist was unexpected, but the setup wasn't sufficient.",
    "This new director shows promise, although this debut work is somewhat rough around the edges."
]

# 预测
for review in new_reviews:
    prob = loaded_model.decision_function([review])[0]
    pred = "Positive" if loaded_model.predict([review])[0] == 1 else "Negative"
    print(f"Review: {review}")
    print(f"Prediction: {pred} (Confidence: {abs(prob):.2f})\n")
