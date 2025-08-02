
# 🧠 Personality Prediction using NLP and Machine Learning

This is a web-based application that predicts whether a person is an **Introvert** or **Extrovert** based on their text input. It uses **Natural Language Processing (NLP)** and a **Logistic Regression model** trained on MBTI personality type data.

---

## 🔍 Features

- 🧾 Text input from user (e.g., "I love hanging out with people at parties!")
- 🤖 NLP preprocessing using `nltk` and `TfidfVectorizer`
- 🧠 Logistic Regression model for binary classification (I/E)
- 🧪 Real-time prediction via web interface (Flask)
- 🌐 Deployable on Render or other cloud platforms

---

## 🛠️ Tech Stack

| Component | Tech |
|----------|------|
| Language | Python |
| ML | Scikit-learn |
| NLP | NLTK, TfidfVectorizer |
| Backend | Flask |
| Frontend | HTML, CSS (Jinja templates) |
| Hosting | Render (free tier supported) |

---

## 📁 Project Structure

```
.
├── app.py                  # Flask web server
├── train_model.py          # Script to train and save ML model
├── model.pkl               # Saved Logistic Regression model
├── vectorizer.pkl          # Saved TF-IDF vectorizer
├── requirements.txt        # Python dependencies
├── Procfile                # For Render deployment
└── templates/
    └── index.html          # Frontend form for text input
```

---

## 🚀 How to Run Locally

1. **Clone the repo**
```bash
git clone https://github.com/OmkSawant/personality-predictor.git
cd personality-predictor
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run Flask app**
```bash
python app.py
```

4. Visit: `http://localhost:5000`

---

## 💡 Future Improvements

- Predict all 16 MBTI personality types
- Show confidence scores or probability
- Add chatbot-style UI or sentiment feedback
- Mobile responsive design

---

## 📄 Dataset

Used the [MBTI Personality Dataset](https://www.kaggle.com/datasets/datasnaek/mbti-type) from Kaggle.

---

## 📸 Sample Output

> Input: `I enjoy deep conversations and being alone with my thoughts.`  
> Output: **Introvert**

> Input: `I love going to parties and talking to everyone!`  
> Output: **Extrovert**

---

## 📌 Author

- 👤 [Om Sawant](https://github.com/OmkSawant)
- 💬 Built as a beginner AI/ML project using real-world deployment

### 🚀 Live Demo
[Click here to try the app](https://personality-predictor-gf9f.onrender.com)
