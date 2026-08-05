# 🎓 Student Performance Clustering System

An end-to-end Machine Learning web application and analytics platform built with **Python**, **Streamlit**, and **Scikit-Learn**.

The system helps educators and academic counselors group students into meaningful **Academic Personas** based on behavioral indicators — weekly study hours, attendance percentage, assignment scores, and exam results — using **Unsupervised K-Means Clustering**.

🔗 **Live Demo:** [student-performance-clustering-syst.vercel.app](https://student-performance-clustering-syst.vercel.app/)

---

## ✨ Features

- **Unsupervised K-Means Clustering** — segments students based on customizable behavioral metrics
- **Optimal K Advisor** — algorithmic recommendation using the Elbow Method and Silhouette Analysis
- **Automated Academic Persona Generation** — categorizes clusters into intuitive profiles:
  - 🌟 **High Performers** (Top Tier) — high study hours + high attendance
  - ⚠️ **At-Risk / Needs Intervention** — low study hours + low attendance
  - 📚 **Passive / Attentive Learners** — low study hours + high attendance
  - ⚡ **Independent / Self-Driven** — high study hours + low attendance
- **Interactive 2D & 3D Visualizations** — hover tooltips, centroid overlays, and rotational 3D plots powered by Plotly
- **Custom Data Support** — upload any CSV dataset or use the built-in 300-student synthetic generator
- **Real-Time Single Student Predictor** — predicts cluster membership for individual student metrics with tailored advisor recommendations
- **Exportable Results** — instant CSV download of the clustered dataset
- **Modern Aesthetic UI** — toggle between Dark Mode and Light Mode themes

---

## 📁 Project Structure

```
Student Performance Clustering System/
├── app.py                              # Main Streamlit web application & UI dashboard
├── requirements.txt                    # Dependency specifications
├── test_clustering.py                  # Automated unit and integration test suite
├── data/
│   └── student_performance_sample.csv  # Pre-generated sample dataset (300 students)
└── utils/
    ├── data_generator.py               # Synthetic dataset generator engine
    ├── clustering.py                   # ML pipeline (scaling, K-Means, Elbow, Personas, Predictor)
    └── visualization.py                # Plotly interactive chart builders
```

---

## 📸 Screenshots

> Drop your screenshot files into a `screenshots/` folder in the project root using the filenames below (or update the paths to match your own), and they'll render automatically here.

### 📊 Data Overview
KPI cards, dataset preview, summary statistics, and correlation heatmap.

![Data Overview](./screenshots/data-overview.png)

### 📈 Optimal K Finder
Elbow Method (WCSS) and Silhouette Score analysis with recommended K.

![Optimal K Finder](./screenshots/optimal-k-finder.png)

### 🎨 Cluster Visualizer
2D/3D interactive scatter plots with centroid overlays and feature boxplots.

![Cluster Visualizer](./screenshots/cluster-visualizer.png)

### 📋 Dataset & Personas
Color-coded persona breakdown cards with cluster filtering and CSV export.

![Dataset & Personas](./screenshots/dataset-personas.png)

### 🔮 Student Predictor
Live prediction form with persona badge and generated action plan.

![Student Predictor](./screenshots/student-predictor.png)

### 🌗 Dark / Light Mode
Theme toggle showcasing both UI modes.

![Dark and Light Mode](./screenshots/theme-toggle.png)

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---|---|---|
| `streamlit` | `>=1.30.0,<1.42.0` | UI Dashboard & Web Framework |
| `starlette` | `<0.40.0` | ASGI framework compatibility |
| `pandas` | `>=2.0.0` | Dataframe manipulation & CSV export |
| `numpy` | `>=1.24.0` | Vectorized numerical operations & data sampling |
| `scikit-learn` | `>=1.3.0` | StandardScaler, KMeans, silhouette_score |
| `plotly` | `>=5.15.0` | Interactive 2D/3D visualizations |
| `matplotlib` | `>=3.7.0` | Base graphics support |
| `seaborn` | `>=0.12.0` | Statistical visualization support |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd "Student Performance Clustering System"

# Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

### Run Tests

```bash
python -m pytest test_clustering.py
```

---

## 🧭 How It Works

```mermaid
flowchart TD
    A[Data Ingestion: Sample CSV or Custom CSV Upload] --> B[Data Preprocessing: StandardScaler Normalization]
    B --> C[Optimal K Advisor: Compute WCSS & Silhouette Scores]
    B --> D[Model Training: Fit K-Means Clustering Model]
    D --> E[Centroid Unscaling & Persona Mapping]
    E --> F[Interactive Visualizations: 2D/3D Scatter, Boxplots, Heatmaps]
    E --> G[Student Persona Breakdown & CSV Export]
    E --> H[Live Student Predictor & Academic Advisor Action Plan]
```

1. **Data Ingestion** — load the built-in 300-student sample dataset or upload a custom CSV
2. **Preprocessing** — standardize selected features using `StandardScaler` (zero mean, unit variance)
3. **Optimal K Advisor** — compute WCSS (Elbow Method) and Silhouette Scores across K = 2 to 8
4. **Model Training** — fit `KMeans` (`n_init=10`) with the chosen number of clusters
5. **Centroid Unscaling & Persona Mapping** — inverse-transform centroids back to real units and assign academic archetypes based on median study hours/attendance splits
6. **Visualization & Insights** — explore results via 2D/3D scatter plots, boxplots, correlation heatmaps, persona breakdowns, CSV export, and a live single-student predictor

---

## 📊 Application Tabs

| Tab | Description |
|---|---|
| 📊 **Data Overview** | KPI cards (total students, features, means), sample preview, summary statistics, correlation heatmap |
| 📈 **Optimal K Finder** | Side-by-side Elbow (WCSS) and Silhouette Score plots with the recommended K highlighted |
| 🎨 **Cluster Visualizer** | 2D scatter with centroid overlays, optional 3D scatter (≥3 features), per-cluster feature boxplots |
| 📋 **Dataset & Personas** | Color-coded persona breakdown, cluster filtering, CSV export |
| 🔮 **Student Predictor** | Live form to predict an individual student's cluster and persona, with a generated action plan |

---

## 🧪 Sample Dataset

The synthetic dataset generator (`utils/data_generator.py`) produces 300 realistic student records across four behavioral archetypes:

| Persona | % of Cohort | Study Hours (μ, σ) | Attendance (μ, σ) |
|---|---|---|---|
| High Performers | 30% (90 students) | 22, 4 | 88%, 5 |
| At-Risk | 25% (75 students) | 6, 2.5 | 52%, 8 |
| Passive Learners | 25% (75 students) | 8, 2.5 | 85%, 6 |
| Independent Learners | 20% (60 students) | 20, 3.5 | 60%, 7 |

Values are clipped to realistic bounds (e.g., Study Hours: 1–35, Attendance: 35–100%) and paired with generated Student IDs (`STU-1001`–`STU-1300`) and names.

---

## 📄 License

Add your preferred license here (e.g., MIT).

---

## 🙋 Contributing

Issues and pull requests are welcome. Please run the test suite (`test_clustering.py`) before submitting changes.
