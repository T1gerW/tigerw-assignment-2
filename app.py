import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-GUI rendering

from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import random

app = Flask(__name__)

# Global variables to store data and centroids
data = None
centroids = None
labels = None

def run_kmeans(data, initial_centroids, max_iter=300):
    """
    Custom implementation of the KMeans algorithm.
    Args:
        data (ndarray): The dataset of shape (num_samples, num_features).
        initial_centroids (ndarray): The initial centroids of shape (k, num_features).
        max_iter (int): The maximum number of iterations.

    Returns:
        final_centroids (ndarray): The final centroids of shape (k, num_features).
        labels (ndarray): The labels of each data point indicating the nearest centroid.
    """
    num_samples, num_features = data.shape
    k = len(initial_centroids)  # Number of clusters

    # Initialize centroids
    centroids = np.array(initial_centroids)
    prev_centroids = np.zeros_like(centroids)
    labels = np.zeros(num_samples)

    # Iterate until convergence or reaching max_iter
    for _ in range(max_iter):
        # Step 1: Assign points to the nearest centroid
        for i in range(num_samples):
            distances = np.linalg.norm(data[i] - centroids, axis=1)  # Calculate distance to each centroid
            labels[i] = np.argmin(distances)  # Assign label of closest centroid

        # Step 2: Update centroids based on the mean of assigned points
        prev_centroids = np.copy(centroids)
        for j in range(k):
            points_assigned = data[labels == j]
            if len(points_assigned) > 0:
                centroids[j] = np.mean(points_assigned, axis=0)

        # Step 3: Check for convergence (if centroids do not change)
        if np.allclose(centroids, prev_centroids):
            break

    return centroids, labels

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to generate a new random dataset without obvious clusters
@app.route('/generate_dataset', methods=['POST'])
def generate_dataset():
    global data, centroids, labels
    
    # Generate a completely random dataset uniformly distributed between -10 and 10
    num_samples = 200
    data = np.random.uniform(-10, 10, size=(num_samples, 2))
    
    # Reset centroids and labels
    centroids, labels = None, None

    # Generate the initial plot without centroids or labels
    plot_url = plot_clusters(data, centroids, labels)
    
    return jsonify({'data': data.tolist(), 'plot': plot_url})


# Function to plot the clusters with x and y axis labels and fixed axis limits
def plot_clusters(data, centroids, labels):
    plt.figure(figsize=(8, 6))

    # Plot the data points
    if labels is None:
        plt.scatter(data[:, 0], data[:, 1], s=50, cmap='viridis')
    else:
        plt.scatter(data[:, 0], data[:, 1], c=labels, s=50, cmap='viridis')

    # Plot the centroids
    if centroids is not None and len(centroids) > 0:
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.75)

    # Set axis labels and title
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.title("KMeans Clustering Visualization")

    # Add grid lines
    plt.grid(True)

    # Set fixed x and y axis limits from -10 to 10
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

    # Emphasize x and y axes passing through the origin
    plt.axhline(0, color='black', linestyle='-', linewidth=1)  # Thicker line for x-axis through origin
    plt.axvline(0, color='black', linestyle='-', linewidth=1)  # Thicker line for y-axis through origin

    # Optional: Set tick marks for better visibility
    plt.xticks(np.arange(-10, 11, 2))
    plt.yticks(np.arange(-10, 11, 2))

    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    return plot_url



@app.route('/initialize_centroids', methods=['POST'])
def initialize_centroids():
    global centroids, data, labels

    # Retrieve initialization method and centroids from the request
    method = request.json.get('method')
    k = request.json.get('k', 4)  # Default to 4 centroids if k is not provided

    if method == 'random':
        centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    elif method == 'farthest':
        centroids = farthest_first_initialization(data, k)
    elif method == 'kmeans++':
        centroids = kmeans_plus_plus_initialization(data, k)
    elif method == 'manual':
        # For manual initialization, use provided centroids from the request
        centroids = np.array(request.json.get('centroids'))
        if len(centroids) == 0:
            return jsonify({'error': 'No centroids provided for manual initialization'}), 400
    else:
        centroids = np.array([])

    # Reset labels whenever new centroids are initialized
    labels = None

    # Debug print to check centroids
    print(f"Centroids after initialization: {centroids}")

    # Generate the updated plot with the new centroids
    plot_url = plot_clusters(data, centroids, labels)
    return jsonify({'centroids': centroids.tolist(), 'plot': plot_url})





@app.route('/kmeans_step', methods=['POST'])
def kmeans_step():
    global centroids, labels, data

    if centroids is not None and len(centroids) > 0 and data is not None and len(data) > 0:
        try:
            # Debug print to check centroids and labels
            print(f"Centroids before step: {centroids}")
            print(f"Labels before step: {labels}")

            # Run a single step of KMeans using the custom function
            centroids, labels = run_kmeans(data, centroids, max_iter=1)  # One iteration step

            # Debug print to check centroids and labels after step
            print(f"Centroids after step: {centroids}")
            print(f"Labels after step: {labels}")

            # Generate the plot with the updated centroids and labels
            plot_url = plot_clusters(data, centroids, labels)
            return jsonify({'plot': plot_url, 'centroids': centroids.tolist(), 'labels': labels.tolist()})
        except Exception as e:
            print(f"Error during KMeans step: {e}")
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'Centroids not initialized or data is missing'}), 400

@app.route('/kmeans_converge', methods=['POST'])
def kmeans_converge():
    global centroids, labels, data

    if centroids is not None and len(centroids) > 0 and data is not None and len(data) > 0:
        try:
            # Debug print to check centroids and labels
            print(f"Centroids before convergence: {centroids}")
            print(f"Labels before convergence: {labels}")

            # Run KMeans until convergence using the custom function
            centroids, labels = run_kmeans(data, centroids, max_iter=300)  # Full convergence

            # Debug print to check centroids and labels after convergence
            print(f"Centroids after convergence: {centroids}")
            print(f"Labels after convergence: {labels}")

            # Generate the final plot after convergence
            plot_url = plot_clusters(data, centroids, labels)
            return jsonify({'plot': plot_url, 'centroids': centroids.tolist(), 'labels': labels.tolist()})
        except Exception as e:
            print(f"Error during KMeans convergence: {e}")
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'Centroids not initialized or data is missing'}), 400





# Helper function for farthest first initialization
def farthest_first_initialization(X, k):
    centroids = [X[np.random.choice(X.shape[0])]]
    while len(centroids) < k:
        dist = np.array([min([np.linalg.norm(x - c) for c in centroids]) for x in X])
        centroids.append(X[np.argmax(dist)])
    return np.array(centroids)

# Helper function for KMeans++ initialization
def kmeans_plus_plus_initialization(X, k):
    centroids = [X[np.random.choice(X.shape[0])]]
    for _ in range(1, k):
        dist_sq = np.array([min([np.inner(c-x, c-x) for c in centroids]) for x in X])
        probabilities = dist_sq / dist_sq.sum()
        cumulative_probabilities = probabilities.cumsum()
        r = random.random()
        for j, p in enumerate(cumulative_probabilities):
            if r < p:
                centroids.append(X[j])
                break
    return np.array(centroids)

@app.route('/reset', methods=['POST'])
def reset():
    global data, centroids, labels
    
    # Reset centroids and labels
    centroids, labels = None, None
    
    # Generate the initial plot with only the data points
    plot_url = plot_clusters(data, centroids, labels)
    
    # Return the plot to the front end along with a success message
    return jsonify({'message': 'Algorithm state reset successfully', 'plot': plot_url})



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)


