import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def preprocess_data(df, feature_cols):
    """
    Extracts features and normalizes them using StandardScaler.
    Returns:
        X_scaled (np.ndarray): Scaled feature matrix
        scaler (StandardScaler): Fitted scaler instance
    """
    X = df[feature_cols].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def compute_elbow_and_silhouette(X_scaled, max_k=10):
    """
    Computes WCSS (inertia) and Silhouette Scores for K from 2 to max_k.
    Returns:
        results_df (pd.DataFrame): Dataframe with K, WCSS, and Silhouette_Score
    """
    max_k = min(max_k, len(X_scaled) - 1)
    wcss = []
    silhouette_scores = []
    k_range = list(range(2, max_k + 1))

    # Safety guard: if dataset is too small to evaluate any K, return empty frame
    if not k_range:
        return pd.DataFrame(columns=["K", "WCSS", "Silhouette_Score"])
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        wcss.append(kmeans.inertia_)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)
        
    results_df = pd.DataFrame({
        "K": k_range,
        "WCSS": wcss,
        "Silhouette_Score": silhouette_scores
    })
    return results_df

def fit_kmeans(X_scaled, n_clusters, random_state=42):
    """
    Fits K-Means model on scaled data.
    Returns:
        model (KMeans): Trained K-Means model
        labels (np.ndarray): Cluster index for each sample
    """
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X_scaled)
    return model, labels

def get_unscaled_centroids(model, scaler, feature_cols):
    """
    Transforms cluster centroids back to original unit space.
    Returns:
        centroids_df (pd.DataFrame): Centroid values in original units
    """
    centroids_scaled = model.cluster_centers_
    centroids_orig = scaler.inverse_transform(centroids_scaled)
    
    df_centroids = pd.DataFrame(centroids_orig, columns=feature_cols)
    df_centroids.index.name = "Cluster"
    df_centroids.reset_index(inplace=True)
    return df_centroids

def assign_personas(centroids_df, feature_cols):
    """
    Assigns meaningful academic personas to clusters based on mean Study Hours and Attendance.
    """
    personas = {}
    
    # Calculate dataset medians/means for context if study hours & attendance exist
    study_col = next((c for c in feature_cols if "study" in c.lower()), None)
    attend_col = next((c for c in feature_cols if "attend" in c.lower()), None)
    
    if study_col and attend_col:
        study_mid = centroids_df[study_col].median()
        attend_mid = centroids_df[attend_col].median()
        
        for idx, row in centroids_df.iterrows():
            cluster_id = row["Cluster"]
            st = row[study_col]
            at = row[attend_col]
            
            if st >= study_mid and at >= attend_mid:
                persona = "🌟 High Performers (Top Tier)"
                desc = "High study dedication and high attendance. Excellent academic track record."
            elif st < study_mid and at < attend_mid:
                persona = "⚠️ At-Risk / Needs Intervention"
                desc = "Low study hours and low attendance. High risk of poor academic outcomes."
            elif st < study_mid and at >= attend_mid:
                persona = "📚 Passive / Attentive Learners"
                desc = "Attends classes regularly but puts in lower self-study hours."
            elif st >= study_mid and at < attend_mid:
                persona = "⚡ Independent / Self-Driven"
                desc = "High self-study hours, though class attendance is lower."
            else:
                persona = f"🎯 Group {cluster_id}"
                desc = "Balanced academic metrics."
            
            personas[cluster_id] = {
                "name": persona,
                "description": desc
            }
    else:
        for idx, row in centroids_df.iterrows():
            cluster_id = row["Cluster"]
            personas[cluster_id] = {
                "name": f"Cluster {cluster_id}",
                "description": f"Cluster with centroid: {dict(row.drop('Cluster'))}"
            }
            
    return personas

def predict_single_student(model, scaler, feature_cols, input_values, personas):
    """
    Predicts cluster for a single student based on input dict.
    Returns:
        cluster_id (int), persona_info (dict)
    """
    input_df = pd.DataFrame([input_values])[feature_cols]
    input_scaled = scaler.transform(input_df)
    cluster_id = int(model.predict(input_scaled)[0])
    persona_info = personas.get(cluster_id, {"name": f"Cluster {cluster_id}", "description": ""})
    return cluster_id, persona_info
