from dbscan import generate
from gen import make_random_data
from sklearn import metrics
from plot import make_plot

def output_details(labels, n_clusters_, n_noise_):
    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
    print(
        "Adjusted Mutual Information: %0.3f"
        % metrics.adjusted_mutual_info_score(labels_true, labels)
    )
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))


if __name__ == "__main__":
    [X, labels_true] = make_random_data()
    [labels, n_clusters_, n_noise_, core_samples_mask] = generate(X)

    output_details(labels, n_clusters_, n_noise_)

    make_plot(X, labels, core_samples_mask, n_clusters_)

    


