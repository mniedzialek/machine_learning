import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_image(filepath: str) -> np.ndarray:
    return np.array(Image.open(filepath).convert("RGB"), dtype=np.float32)

def extract_features(image: np.ndarray, feature_type: str = "exg") -> np.ndarray:
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    if feature_type == "exg":
        return 2.0 * G - R - B
    elif feature_type == "rgb":
        return image.copy()
    else:
        raise ValueError("Unsupported feature type.")

def fit_gaussian_1d(data: np.ndarray) -> tuple[float, float, float]:
    flat = data.flatten()
    N = len(flat)
    mu = float(np.sum(flat) / N)
    var = float(np.sum((flat - mu) ** 2) / N)
    std = float(np.sqrt(var))
    return mu, std, var

def gaussian_pdf_1d(x: np.ndarray | float, mu: float, std: float) -> np.ndarray:
    std = max(std, 1e-9)
    coef = 1.0 / (np.sqrt(2.0 * np.pi) * std)
    return coef * np.exp(-0.5 * ((x - mu) / std) ** 2)

def bayes_classify_1d(
    features: np.ndarray,
    mu_f: float,
    std_f: float,
    prior_f: float,
    mu_u: float,
    std_u: float,
    prior_u: float
) -> np.ndarray:
    p_f = gaussian_pdf_1d(features, mu_f, std_f) * prior_f
    p_u = gaussian_pdf_1d(features, mu_u, std_u) * prior_u
    return p_f > p_u

class KNNClassifier:
    def __init__(self, k: int = 3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.X_train = np.array(X, dtype=np.float32)
        self.y_train = np.array(y, dtype=np.int32)
        return self

    def predict(self, X: np.ndarray, batch_size: int = 5000) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_test = np.array(X, dtype=np.float32)
        M = X_test.shape[0]
        preds = np.zeros(M, dtype=np.int32)

        y2 = np.sum(self.X_train ** 2, axis=1, keepdims=True).T

        for i in range(0, M, batch_size):
            batch = X_test[i : i + batch_size]
            x2 = np.sum(batch ** 2, axis=1, keepdims=True)
            dist_sq = x2 + y2 - 2.0 * np.dot(batch, self.X_train.T)

            nearest_idx = np.argpartition(dist_sq, kth=self.k - 1, axis=1)[:, :self.k]
            nearest_labels = self.y_train[nearest_idx]
            preds[i : i + batch_size] = (np.mean(nearest_labels, axis=1) >= 0.5).astype(np.int32)

        return preds

def color_mask(mask: np.ndarray) -> np.ndarray:
    colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
    colored[mask] = [0, 100, 255]   # Blue for Forest
    colored[~mask] = [255, 50, 50]  # Red for Urban
    return colored

def plot_samples(img_forest: np.ndarray, img_urban: np.ndarray, img_test: np.ndarray):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_forest.astype(np.uint8))
    axes[0].set_title("Forest (Training)")
    axes[0].axis("off")

    axes[1].imshow(img_urban.astype(np.uint8))
    axes[1].set_title("Urban (Training)")
    axes[1].axis("off")

    axes[2].imshow(img_test.astype(np.uint8))
    axes[2].set_title("Test Image")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

def plot_histograms_and_pdfs(
    feat_forest: np.ndarray,
    feat_urban: np.ndarray,
    mu_f: float,
    std_f: float,
    mu_u: float,
    std_u: float
):
    f_flat = feat_forest.flatten()
    u_flat = feat_urban.flatten()
    x = np.linspace(min(f_flat.min(), u_flat.min()), max(f_flat.max(), u_flat.max()), 400)

    plt.figure(figsize=(9, 5))
    plt.hist(f_flat, bins=50, density=True, alpha=0.5, color="forestgreen", label="Forest")
    plt.hist(u_flat, bins=50, density=True, alpha=0.5, color="crimson", label="Urban")

    plt.plot(x, gaussian_pdf_1d(x, mu_f, std_f), color="darkgreen", lw=2, label=fr"Forest Fit ($\mu$={mu_f:.1f}, $\sigma$={std_f:.1f})")
    plt.plot(x, gaussian_pdf_1d(x, mu_u, std_u), color="darkred", lw=2, label=fr"Urban Fit ($\mu$={mu_u:.1f}, $\sigma$={std_u:.1f})")

    plt.title("Histograms and Fitted Gaussian PDFs")
    plt.xlabel("ExG Value")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_bayes_result(test_img: np.ndarray, bayes_mask: np.ndarray):
    colored = color_mask(bayes_mask)
    overlay = (0.5 * colored + 0.5 * test_img).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(test_img.astype(np.uint8))
    axes[0].set_title("Test Image")
    axes[0].axis("off")

    axes[1].imshow(colored)
    axes[1].set_title("Bayes (Blue: Forest, Red: Urban)")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

def plot_knn_results(test_img: np.ndarray, knn_dict: dict[int, np.ndarray]):
    fig, axes = plt.subplots(1, len(knn_dict) + 1, figsize=(14, 3.8))
    axes[0].imshow(test_img.astype(np.uint8))
    axes[0].set_title("Test Image")
    axes[0].axis("off")

    for i, (k, mask) in enumerate(knn_dict.items(), start=1):
        axes[i].imshow(color_mask(mask))
        axes[i].set_title(f"k-NN (k={k})\nBlue: Forest, Red: Urban")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()
