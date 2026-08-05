# 🎓 Student Performance Clustering System

An interactive Machine Learning web application that groups students based on their **Study Hours** and **Attendance Percentage** using **K-Means Clustering**. Built with Python, Flask, HTML5, CSS3, JavaScript, and Plotly.js.

---

## 📖 Project Description

The Student Performance Clustering System analyzes student academic data and identifies students with similar academic behavior by forming clusters. It visualizes the generated clusters interactively and allows educators to predict which group a new student belongs to in real-time.

---

## 🎯 Objectives

- Analyze student academic data (study hours, attendance, scores)
- Group students based on performance patterns using unsupervised learning
- Visualize student clusters with interactive 2D & 3D plots
- Display clustered dataset with persona insights
- Predict cluster and receive targeted academic guidance for any individual student in real-time

---

## 🤖 Machine Learning Model

**K-Means Clustering** (Unsupervised Learning)
- Elbow Method (WCSS) for optimal K selection
- Silhouette Score Analysis for cluster quality evaluation
- Automated Persona Assignment per cluster

### Academic Personas Generated
| Persona | Description |
|---|---|
| 🌟 High Performers | High study hours + high attendance |
| ⚠️ At-Risk / Needs Intervention | Low study hours + low attendance |
| 📚 Passive / Attentive Learners | High attendance, lower self-study |
| ⚡ Independent / Self-Driven | High self-study, lower attendance |

---

## 🛠 Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.x | Core language |
| Flask | Backend REST API server |
| HTML5 / CSS3 / JavaScript | Modern dashboard user interface |
| Plotly.js | Client-side interactive visualizations |
| Scikit-Learn | K-Means clustering & StandardScaler |
| Pandas & NumPy | Data manipulation |

---

## 📁 Project Structure

```
Student Performance Clustering System/
├── app.py                          # Flask backend web application
├── requirements.txt                # Python dependencies
├── test_clustering.py              # Automated test suite
├── data/
│   └── student_performance_sample.csv   # Generated dataset (300 students)
├── templates/
│   └── index.html                  # Dashboard HTML template
├── static/
│   ├── styles.css                  # Modern CSS design system & dark mode
│   └── app.js                      # Client application & Plotly rendering
└── utils/
    ├── data_generator.py           # Synthetic dataset generator
    ├── clustering.py               # K-Means training & persona mapping
    └── visualization.py            # Plotly chart builders
```

---

## ⚙️ Working

1. Load the student dataset (built-in 300 students or upload custom CSV)
2. Perform preprocessing & feature scaling (StandardScaler)
3. Select study hours and attendance as clustering features
4. Evaluate optimal K using Elbow Method & Silhouette Analysis
5. Train the K-Means model
6. Generate student clusters with persona labels
7. Display interactive 2D/3D cluster scatter plots
8. Show clustered dataset table with persona assignments
9. Predict cluster & advice for any new student via live form

---

## 🚀 Installation & Running

```bash
# 1. Clone the repository
git clone https://github.com/ARTHIK22/-Student-Performance-Clustering-System.git
cd "-Student-Performance-Clustering-System"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Flask application
python app.py
```

Open **http://localhost:5000** in your browser.

---

## 🌟 Features

- **🌙 / ☀️ Dark & Light Mode** toggle in sidebar
- **📊 Interactive Visualizations** with hover tooltips (Plotly.js)
- **🎯 Live Student Predictor** — input metrics and receive instant cluster assignment & customized advice
- **📈 Optimal K Advisor** — algorithmic recommendation for best cluster count
- **👥 Persona Cards** — color-coded academic behavior cards per cluster

---

## 📄 License

MIT License — free for educational and research use.
