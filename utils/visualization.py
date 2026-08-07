import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set consistent color palette for clusters
CLUSTER_COLORS = [
    '#6366F1', '#10B981', '#F59E0B', '#EF4444', 
    '#8B5CF6', '#EC4899', '#06B6D4', '#84CC16'
]

def plot_elbow_and_silhouette(eval_df):
    """
    Creates a dual-panel Plotly interactive plot for Elbow Method and Silhouette Analysis.
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("WCSS (Elbow Method)", "Silhouette Score Analysis"),
        horizontal_spacing=0.12
    )

    # WCSS Plot
    fig.add_trace(
        go.Scatter(
            x=eval_df["K"],
            y=eval_df["WCSS"],
            mode="lines+markers",
            name="WCSS",
            line=dict(color="#6366F1", width=3),
            marker=dict(size=8, color="#4F46E5")
        ),
        row=1, col=1
    )

    # Silhouette Score Plot
    fig.add_trace(
        go.Scatter(
            x=eval_df["K"],
            y=eval_df["Silhouette_Score"],
            mode="lines+markers",
            name="Silhouette",
            line=dict(color="#10B981", width=3),
            marker=dict(size=8, color="#059669")
        ),
        row=1, col=2
    )

    # Optimal K highlight (max silhouette score)
    best_k_row = eval_df.loc[eval_df["Silhouette_Score"].idxmax()]
    fig.add_trace(
        go.Scatter(
            x=[best_k_row["K"]],
            y=[best_k_row["Silhouette_Score"]],
            mode="markers",
            name=f"Optimal K={int(best_k_row['K'])}",
            marker=dict(size=14, color="#EF4444", symbol="star")
        ),
        row=1, col=2
    )

    fig.update_xaxes(title_text="Number of Clusters (K)", row=1, col=1, dtick=1)
    fig.update_yaxes(title_text="Within-Cluster Sum of Squares (WCSS)", row=1, col=1)

    fig.update_xaxes(title_text="Number of Clusters (K)", row=1, col=2, dtick=1)
    fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)

    fig.update_layout(
        height=420,
        showlegend=True,
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def plot_2d_clusters(df, x_col, y_col, cluster_col, centroids_df=None, personas=None):
    """
    Creates an interactive 2D scatter plot of student clusters with centroids overlaid.
    """
    df_plot = df.copy()
    
    if personas:
        df_plot["Persona"] = df_plot[cluster_col].apply(
            lambda c: f"Cluster {c}: {personas.get(c, {}).get('name', personas.get(str(c), {}).get('name', f'Group {c}'))}"
        )
        color_col = "Persona"
    else:
        df_plot[cluster_col] = df_plot[cluster_col].astype(str)
        color_col = cluster_col

    hover_data = ["Student_ID", "Student_Name"] if "Student_ID" in df_plot.columns else []

    fig = px.scatter(
        df_plot,
        x=x_col,
        y=y_col,
        color=color_col,
        hover_name="Student_Name" if "Student_Name" in df_plot.columns else None,
        hover_data=hover_data + [x_col, y_col],
        color_discrete_sequence=CLUSTER_COLORS,
        title=f"Student Clusters: {x_col.replace('_', ' ')} vs {y_col.replace('_', ' ')}",
        opacity=0.85
    )

    fig.update_traces(marker=dict(size=9, line=dict(width=1, color='White')))

    # Overlay centroids if provided
    if centroids_df is not None and x_col in centroids_df.columns and y_col in centroids_df.columns:
        fig.add_trace(
            go.Scatter(
                x=centroids_df[x_col],
                y=centroids_df[y_col],
                mode="markers+text",
                name="Cluster Centroids",
                text=[f"Centroid {i}" for i in centroids_df["Cluster"]],
                textposition="top center",
                textfont=dict(size=12, color="#F59E0B"),
                marker=dict(
                    size=16,
                    color="#F59E0B",
                    symbol="diamond",
                    line=dict(width=2, color="#FFFFFF")
                )
            )
        )

    fig.update_layout(
        template="plotly_white",
        height=540,
        margin=dict(l=60, r=40, t=60, b=100),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
        xaxis_title=x_col.replace("_", " "),
        yaxis_title=y_col.replace("_", " ")
    )
    return fig

def plot_3d_clusters(df, x_col, y_col, z_col, cluster_col, personas=None):
    """
    Creates an interactive 3D scatter plot of student clusters.
    """
    df_plot = df.copy()
    if personas:
        df_plot["Persona"] = df_plot[cluster_col].apply(
            lambda c: personas.get(c, {}).get("name", personas.get(str(c), {}).get("name", f"Cluster {c}"))
        )
        color_col = "Persona"
    else:
        df_plot[cluster_col] = df_plot[cluster_col].astype(str)
        color_col = cluster_col

    hover_data = ["Student_ID", "Student_Name"] if "Student_ID" in df_plot.columns else []

    fig = px.scatter_3d(
        df_plot,
        x=x_col,
        y=y_col,
        z=z_col,
        color=color_col,
        hover_name="Student_Name" if "Student_Name" in df_plot.columns else None,
        hover_data=hover_data,
        color_discrete_sequence=CLUSTER_COLORS,
        title=f"3D Visualizer ({x_col.replace('_', ' ')} x {y_col.replace('_', ' ')} x {z_col.replace('_', ' ')})",
        opacity=0.9
    )

    fig.update_traces(marker=dict(size=7, line=dict(width=0.5, color='rgba(255,255,255,0.6)')))
    fig.update_layout(
        template="plotly_white",
        height=600,
        margin=dict(l=20, r=20, b=20, t=50),
        scene=dict(
            xaxis_title=x_col.replace("_", " "),
            yaxis_title=y_col.replace("_", " "),
            zaxis_title=z_col.replace("_", " ")
        )
    )
    return fig

def plot_feature_boxplots(df, cluster_col, feature_cols, personas=None):
    """
    Plots box plots comparing feature distributions across clusters.
    """
    df_plot = df.copy()
    if personas:
        df_plot["Cluster Persona"] = df_plot[cluster_col].apply(
            lambda c: personas.get(c, {}).get("name", personas.get(str(c), {}).get("name", f"Cluster {c}"))
        )
        color_col = "Cluster Persona"
    else:
        df_plot["Cluster Persona"] = df_plot[cluster_col].astype(str)
        color_col = "Cluster Persona"

    figs = []
    for col in feature_cols:
        fig = px.box(
            df_plot,
            x="Cluster Persona",
            y=col,
            color="Cluster Persona",
            color_discrete_sequence=CLUSTER_COLORS,
            title=f"Distribution of {col.replace('_', ' ')} across Clusters",
            points="all"
        )
        fig.update_layout(
            template="plotly_white",
            height=400,
            margin=dict(l=50, r=40, t=50, b=120),
            showlegend=False,
            yaxis_title=col.replace("_", " "),
            xaxis_title="Cluster Persona"
        )
        figs.append(fig)
    return figs

def plot_correlation_matrix(df, numeric_cols):
    """
    Plotly heatmap for feature correlations.
    """
    corr = df[numeric_cols].corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="Blues",
        title="Feature Correlation Matrix"
    )
    fig.update_layout(
        template="plotly_white",
        height=400,
        margin=dict(l=50, r=40, t=60, b=50)
    )
    return fig
