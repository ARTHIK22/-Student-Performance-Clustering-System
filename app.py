import os
import sys
import json
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
import plotly.io as pio

# Ensure root directory is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_generator import save_sample_dataset, generate_student_dataset
from utils.clustering import (
    preprocess_data,
    compute_elbow_and_silhouette,
    fit_kmeans,
    get_unscaled_centroids,
    assign_personas,
    predict_single_student
)
from utils.visualization import (
    plot_elbow_and_silhouette,
    plot_2d_clusters,
    plot_3d_clusters,
    plot_feature_boxplots,
    plot_correlation_matrix,
    CLUSTER_COLORS
)

app = Flask(__name__, template_folder="templates", static_folder="static")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SAMPLE_CSV = os.path.join(DATA_DIR, "student_performance_sample.csv")
CUSTOM_CSV = os.path.join(DATA_DIR, "custom_student_data.csv")

def get_current_data(use_custom=False):
    os.makedirs(DATA_DIR, exist_ok=True)
    if use_custom and os.path.exists(CUSTOM_CSV):
        return pd.read_csv(CUSTOM_CSV)
    if not os.path.exists(SAMPLE_CSV):
        save_sample_dataset()
    return pd.read_csv(SAMPLE_CSV)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/dataset", methods=["GET"])
def get_dataset_info():
    use_custom = request.args.get("use_custom", "false").lower() == "true"
    df = get_current_data(use_custom=use_custom)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_candidates = [c for c in numeric_cols if c.lower() != "student_id"]
    
    preview_data = df.head(10).to_dict(orient="records")
    summary_stats = df[feature_candidates].describe().T[["mean", "std", "min", "50%", "max"]].reset_index()
    summary_stats.columns = ["feature", "mean", "std", "min", "median", "max"]
    summary_dict = summary_stats.to_dict(orient="records")
    
    return jsonify({
        "success": True,
        "total_students": len(df),
        "columns": list(df.columns),
        "numeric_features": feature_candidates,
        "preview": preview_data,
        "summary_stats": summary_dict,
        "is_custom": use_custom and os.path.exists(CUSTOM_CSV)
    })

@app.route("/api/regenerate", methods=["POST"])
def regenerate_data():
    save_sample_dataset()
    return jsonify({"success": True, "message": "Sample dataset successfully regenerated!"})

@app.route("/api/upload", methods=["POST"])
def upload_data():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename"}), 400

    try:
        df = pd.read_csv(file)
        if len(df) < 5:
            return jsonify({"success": False, "error": "Dataset must contain at least 5 rows."}), 400
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            return jsonify({"success": False, "error": "Dataset must have at least 2 numerical features."}), 400

        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(CUSTOM_CSV, index=False)
        return jsonify({"success": True, "message": f"Uploaded dataset with {len(df)} rows and {len(df.columns)} columns."})
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to parse CSV: {str(e)}"}), 400

@app.route("/api/cluster", methods=["POST"])
def run_clustering():
    try:
        data = request.json or {}
        use_custom = data.get("use_custom", False)
        selected_features = data.get("features", [])
        n_clusters = int(data.get("n_clusters", 4))
        random_state = int(data.get("random_state", 42))

        df = get_current_data(use_custom=use_custom)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_candidates = [c for c in numeric_cols if c.lower() != "student_id"]

        if not selected_features or len(selected_features) < 2:
            selected_features = feature_candidates[:2] if len(feature_candidates) >= 2 else numeric_cols[:2]

        selected_features = [f for f in selected_features if f in df.columns]
        if len(selected_features) < 2:
            return jsonify({"success": False, "error": "At least 2 valid numeric features are required."}), 400

        X_scaled, scaler = preprocess_data(df, selected_features)
        model, labels = fit_kmeans(X_scaled, n_clusters=n_clusters, random_state=random_state)
        
        df_clustered = df.copy()
        df_clustered["Cluster"] = labels

        centroids_df = get_unscaled_centroids(model, scaler, selected_features)
        personas = assign_personas(centroids_df, selected_features)

        cluster_counts = df_clustered["Cluster"].value_counts().to_dict()
        persona_cards = []
        for cid, info in personas.items():
            cnt = cluster_counts.get(cid, 0)
            pct = round((cnt / len(df)) * 100, 1)
            persona_cards.append({
                "cluster_id": cid,
                "name": info["name"],
                "description": info["description"],
                "count": cnt,
                "percentage": pct,
                "color": CLUSTER_COLORS[int(cid) % len(CLUSTER_COLORS)]
            })

        eval_df = compute_elbow_and_silhouette(X_scaled, max_k=min(8, len(df) - 1))
        fig_eval = plot_elbow_and_silhouette(eval_df)
        
        fig_2d = plot_2d_clusters(
            df_clustered, selected_features[0], selected_features[1],
            "Cluster", centroids_df, personas
        )

        # Determine z-axis column: prefer a 3rd selected feature, then Exam_Score, then any unused numeric col
        if len(selected_features) > 2:
            z_col = selected_features[2]
        elif "Exam_Score" in df.columns and "Exam_Score" not in selected_features[:2]:
            z_col = "Exam_Score"
        else:
            # Pick any unused numeric column to avoid a flat/degenerate 3D chart
            unused_cols = [c for c in feature_candidates if c not in selected_features]
            z_col = unused_cols[0] if unused_cols else selected_features[1]

        fig_3d = plot_3d_clusters(
            df_clustered, selected_features[0], selected_features[1], z_col,
            "Cluster", personas
        )

        boxplots = plot_feature_boxplots(df_clustered, "Cluster", selected_features, personas)
        corr_cols = selected_features + ([c for c in ["Exam_Score"] if c in df.columns and c not in selected_features])
        corr_fig = plot_correlation_matrix(df, corr_cols)

        if eval_df.empty:
            best_k = n_clusters
            best_score = 0.0
        else:
            best_k = int(eval_df.loc[eval_df["Silhouette_Score"].idxmax()]["K"])
            best_score = float(eval_df["Silhouette_Score"].max())

        df_clustered["Persona_Name"] = df_clustered["Cluster"].map(lambda c: personas.get(c, {}).get("name", f"Cluster {c}"))
        sample_records = df_clustered.head(50).to_dict(orient="records")

        return jsonify({
            "success": True,
            "selected_features": selected_features,
            "n_clusters": n_clusters,
            "total_students": len(df),
            "personas": persona_cards,
            "centroids": centroids_df.to_dict(orient="records"),
            "best_k": best_k,
            "best_silhouette_score": round(best_score, 4),
            "eval_data": eval_df.to_dict(orient="records"),
            "fig_eval_json": json.loads(pio.to_json(fig_eval)),
            "fig_2d_json": json.loads(pio.to_json(fig_2d)),
            "fig_3d_json": json.loads(pio.to_json(fig_3d)),
            "boxplots_json": [json.loads(pio.to_json(bp)) for bp in boxplots],
            "corr_fig_json": json.loads(pio.to_json(corr_fig)),
            "clustered_sample": sample_records
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/predict", methods=["POST"])
def predict_student():
    try:
        data = request.json or {}
        use_custom = data.get("use_custom", False)
        selected_features = data.get("features", ["Study_Hours_Per_Week", "Attendance_Percentage"])
        student_input = data.get("student_data", {})
        n_clusters = int(data.get("n_clusters", 4))
        random_state = int(data.get("random_state", 42))

        df = get_current_data(use_custom=use_custom)
        selected_features = [f for f in selected_features if f in df.columns]

        X_scaled, scaler = preprocess_data(df, selected_features)
        model, _ = fit_kmeans(X_scaled, n_clusters=n_clusters, random_state=random_state)
        centroids_df = get_unscaled_centroids(model, scaler, selected_features)
        personas = assign_personas(centroids_df, selected_features)

        predicted_cid, persona_info = predict_single_student(
            model, scaler, selected_features, student_input, personas
        )

        study_hrs = float(student_input.get("Study_Hours_Per_Week", 0))
        attendance = float(student_input.get("Attendance_Percentage", 0))

        advice = []
        if attendance < 75:
            advice.append("⚠️ **Attendance Alert**: Class attendance is below the 75% threshold. Prioritize attending lectures to stay updated on critical coursework.")
        else:
            advice.append("✅ **Attendance Strengths**: Great attendance habit! Maintaining regular class presence supports active learning.")

        if study_hrs < 15:
            advice.append("📚 **Study Habit Growth**: Weekly study hours are on the lower side (< 15 hrs). Aim for structured 2-hour daily study blocks to boost comprehension.")
        elif study_hrs >= 25:
            advice.append("🌟 **Dedicated Scholar**: Outstanding dedication to self-study! Keep up the structured preparation and balance study with wellness rest.")
        else:
            advice.append("👍 **Balanced Study**: Good baseline study routine (15–25 hrs/week). Focus on high-yield study techniques like active recall and practice problems.")

        return jsonify({
            "success": True,
            "cluster_id": predicted_cid,
            "persona_name": persona_info["name"],
            "persona_description": persona_info["description"],
            "color": CLUSTER_COLORS[int(predicted_cid) % len(CLUSTER_COLORS)],
            "advice": advice
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    print("Starting Student Performance Clustering Web Server on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
