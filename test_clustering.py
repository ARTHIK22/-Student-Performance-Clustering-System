import sys
import os
import pandas as pd
import numpy as np

# Ensure root directory is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_generator import save_sample_dataset
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
    plot_correlation_matrix
)

def run_tests():
    print("--- 1. Testing Data Generator ---")
    csv_path, df = save_sample_dataset()
    assert os.path.exists(csv_path), "CSV file was not created!"
    assert len(df) == 300, f"Expected 300 rows, got {len(df)}"
    print(f"Data Generator OK: {len(df)} records generated.")

    print("\n--- 2. Testing Preprocessing & Scaling ---")
    feature_cols = ["Study_Hours_Per_Week", "Attendance_Percentage"]
    X_scaled, scaler = preprocess_data(df, feature_cols)
    assert X_scaled.shape == (300, 2), f"Unexpected shape {X_scaled.shape}"
    print("Preprocessing & Scaling OK.")

    print("\n--- 3. Testing Elbow & Silhouette Computation ---")
    eval_df = compute_elbow_and_silhouette(X_scaled, max_k=6)
    assert len(eval_df) == 5, f"Expected 5 rows for K=2 to 6, got {len(eval_df)}"
    print(f"Optimal K Evaluation OK:\n{eval_df}")

    print("\n--- 4. Testing K-Means Model Fitting ---")
    n_clusters = 4
    model, labels = fit_kmeans(X_scaled, n_clusters=n_clusters)
    df["Cluster"] = labels
    assert len(labels) == 300, "Labels count mismatch!"
    print(f"K-Means Fitting OK for K={n_clusters}.")

    print("\n--- 5. Testing Centroids & Persona Assignment ---")
    centroids_df = get_unscaled_centroids(model, scaler, feature_cols)
    assert len(centroids_df) == n_clusters, "Centroids count mismatch!"
    personas = assign_personas(centroids_df, feature_cols)
    print("Centroids & Personas:")
    for cid, info in personas.items():
        print(f"  Cluster {cid}: {info['name']} -> {info['description']}")

    print("\n--- 6. Testing Predict Single Student ---")
    sample_student = {"Study_Hours_Per_Week": 25.0, "Attendance_Percentage": 92.0}
    predicted_cid, persona_info = predict_single_student(model, scaler, feature_cols, sample_student, personas)
    print(f"Prediction for {sample_student}: Cluster {predicted_cid} ({persona_info['name']})")

    print("\n--- 7. Testing Plotly Visualizations ---")
    fig_eval = plot_elbow_and_silhouette(eval_df)
    fig_2d = plot_2d_clusters(df, "Study_Hours_Per_Week", "Attendance_Percentage", "Cluster", centroids_df, personas)
    fig_3d = plot_3d_clusters(df, "Study_Hours_Per_Week", "Attendance_Percentage", "Exam_Score", "Cluster", personas)
    boxplots = plot_feature_boxplots(df, "Cluster", feature_cols, personas)
    corr_fig = plot_correlation_matrix(df, feature_cols + ["Exam_Score"])
    
    assert fig_eval is not None, "Elbow plot failed!"
    assert fig_2d is not None, "2D plot failed!"
    assert fig_3d is not None, "3D plot failed!"
    assert len(boxplots) == len(feature_cols), "Boxplots failed!"
    assert corr_fig is not None, "Correlation matrix failed!"
    print("Visualizations OK.")

    print("\n✅ ALL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
