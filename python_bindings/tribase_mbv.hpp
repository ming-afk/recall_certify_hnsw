#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tribase_mbv {

template<typename tableint_t, typename dist_t>
struct Cache {
    std::vector<int32_t> cluster_label_by_internal;
    std::vector<dist_t> centroids_rowmajor;
    std::vector<dist_t> cluster_radius_by_cluster;
    std::vector<dist_t> residual_norm_by_internal;
    std::vector<tableint_t> subnn_l2_internal_ids;
    std::vector<dist_t> subnn_l2_dists;
    std::vector<tableint_t> subnn_angle_internal_ids;
    std::vector<dist_t> subnn_angle_vals;
    size_t node_count{0};
    size_t cluster_count{0};
    size_t dim{0};
    size_t keep{0};
    bool ready{false};

    void clear() {
        cluster_label_by_internal.clear();
        centroids_rowmajor.clear();
        cluster_radius_by_cluster.clear();
        residual_norm_by_internal.clear();
        subnn_l2_internal_ids.clear();
        subnn_l2_dists.clear();
        subnn_angle_internal_ids.clear();
        subnn_angle_vals.clear();
        node_count = 0;
        cluster_count = 0;
        dim = 0;
        keep = 0;
        ready = false;
    }

    size_t bytes() const {
        return
            cluster_label_by_internal.size() * sizeof(int32_t) +
            centroids_rowmajor.size() * sizeof(dist_t) +
            cluster_radius_by_cluster.size() * sizeof(dist_t) +
            residual_norm_by_internal.size() * sizeof(dist_t) +
            subnn_l2_internal_ids.size() * sizeof(tableint_t) +
            subnn_l2_dists.size() * sizeof(dist_t) +
            subnn_angle_internal_ids.size() * sizeof(tableint_t) +
            subnn_angle_vals.size() * sizeof(dist_t);
    }
};

}  // namespace tribase_mbv
