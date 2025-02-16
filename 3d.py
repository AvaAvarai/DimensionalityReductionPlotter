import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

# Function to compute 3D t-SNE and visualize it for a single dataset file
def plot_3d_tsne(file_path):
    # Load dataset
    df = pd.read_csv(file_path)

    # Find class column (case-insensitive)
    class_col = [col for col in df.columns if col.lower() == 'class']
    if not class_col:
        raise ValueError(f"No 'class' column found in the dataset {file_path}.")
    class_col = class_col[0]

    # Separate features and class labels
    X = df.drop(columns=[class_col])
    y = df[class_col].astype(str)  # Convert to string for categorical handling

    # Map class labels to numerical values
    class_labels = np.unique(y)
    y_num = np.array([np.where(class_labels == label)[0][0] for label in y])

    # Normalize numerical features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Compute 3D t-SNE with faster algorithm
    tsne = TSNE(n_components=3, method='barnes_hut', random_state=42)
    embedding = tsne.fit_transform(X_scaled)

    # Plot settings
    fig = go.Figure(data=[go.Scatter3d(
        x=embedding[:, 0],
        y=embedding[:, 1],
        z=embedding[:, 2],
        mode='markers',
        marker=dict(
            color=y_num,
            colorscale='Viridis',
            opacity=0.7,
            size=5  # Added size parameter for better visualization
        ),
        text=y,  # Added text parameter to show label on highlight
        hovertemplate='%{text}<br>x: %{x}<br>y: %{y}<br>z: %{z}<br><extra></extra>'
    )])
    fig.update_layout(
        title=f"3D t-SNE of {file_path.split('.')[0]}",
        scene=dict(
            xaxis_title='X-axis',
            yaxis_title='Y-axis',
            zaxis_title='Z-axis',
            camera_eye=dict(x=1.1, y=1.1, z=0.6)  # Adjusted camera position for better view
        )
    )
    fig.show()  # Display the plot in the browser

# Example usage
if __name__ == "__main__":
    file_path = "breast-cancer-wisconsin.csv"
    plot_3d_tsne(file_path)

# To install required packages, run: pip install pandas scikit-learn plotly
