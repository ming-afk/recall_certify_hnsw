#include <iostream>
#include <fstream>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "hnswlib.h"
#include <thread>
#include <atomic>
#include <cstdint>
#include <stdlib.h>
#include <assert.h>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <algorithm>
#include <chrono>
#include <limits>
#include <random>
#include <numeric>
#include <string>
#include "tribase_mbv.hpp"
namespace py = pybind11;
using namespace pybind11::literals;  // needed to bring in _a literal

#ifndef HNSWLIB_MBV_CONFORMAL_EARLY_STOP
#define HNSWLIB_MBV_CONFORMAL_EARLY_STOP 1
#endif

/*
 * replacement for the openmp '#pragma omp parallel for' directive
 * only handles a subset of functionality (no reductions etc)
 * Process ids from start (inclusive) to end (EXCLUSIVE)
 *
 * The method is borrowed from nmslib
 */
template<class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if (id >= end) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        /*
                         * This will work even when current is the largest value that
                         * size_t can fit, because fetch_add returns the previous value
                         * before the increment (what will result in overflow
                         * and produce 0 instead of current + 1).
                         */
                        current = end;
                        break;
                    }
                }
            }));
        }
        for (auto &thread : threads) {
            thread.join();
        }
        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}


inline void assert_true(bool expr, const std::string & msg) {
    if (expr == false) throw std::runtime_error("Unpickle Error: " + msg);
    return;
}


class CustomFilterFunctor: public hnswlib::BaseFilterFunctor {
    std::function<bool(hnswlib::labeltype)> filter;

 public:
    explicit CustomFilterFunctor(const std::function<bool(hnswlib::labeltype)>& f) {
        filter = f;
    }

    bool operator()(hnswlib::labeltype id) {
        return filter(id);
    }
};


inline void get_input_array_shapes(const py::buffer_info& buffer, size_t* rows, size_t* features) {
    if (buffer.ndim != 2 && buffer.ndim != 1) {
        char msg[256];
        snprintf(msg, sizeof(msg),
            "Input vector data wrong shape. Number of dimensions %d. Data must be a 1D or 2D array.",
            buffer.ndim);
        throw std::runtime_error(msg);
    }
    if (buffer.ndim == 2) {
        *rows = buffer.shape[0];
        *features = buffer.shape[1];
    } else {
        *rows = 1;
        *features = buffer.shape[0];
    }
}


inline std::vector<size_t> get_input_ids_and_check_shapes(const py::object& ids_, size_t feature_rows) {
    std::vector<size_t> ids;
    if (!ids_.is_none()) {
        py::array_t < size_t, py::array::c_style | py::array::forcecast > items(ids_);
        auto ids_numpy = items.request();
        // check shapes
        if (!((ids_numpy.ndim == 1 && ids_numpy.shape[0] == feature_rows) ||
              (ids_numpy.ndim == 0 && feature_rows == 1))) {
            char msg[256];
            snprintf(msg, sizeof(msg),
                "The input label shape %d does not match the input data vector shape %d",
                ids_numpy.ndim, feature_rows);
            throw std::runtime_error(msg);
        }
        // extract data
        if (ids_numpy.ndim == 1) {
            std::vector<size_t> ids1(ids_numpy.shape[0]);
            for (size_t i = 0; i < ids1.size(); i++) {
                ids1[i] = items.data()[i];
            }
            ids.swap(ids1);
        } else if (ids_numpy.ndim == 0) {
            ids.push_back(*items.data());
        }
    }

    return ids;
}


template<typename dist_t, typename data_t = float>
class Index {
 public:
    static const int ser_version = 1;  // serialization version

    std::string space_name;
    int dim;
    size_t seed;
    size_t default_ef;

    bool index_inited;
    bool ep_added;
    bool normalize;
    int num_threads_default;
    hnswlib::labeltype cur_l;
    hnswlib::HierarchicalNSW<dist_t>* appr_alg;
    hnswlib::SpaceInterface<float>* l2space;

    // --- PCA prune data (optional, copied into C++ buffers) ---
    std::vector<float> pca_dirs_buf_;   // (m, dim) row-major
    std::vector<float> projX_buf_;      // (N, m) row-major, N indexed by external label
    size_t projX_rows_{0};
    size_t pca_m_{0};
    bool has_pca_{false};

    // --- Convex-hull landmark cache for MBV_hull_query_ex ---
    // We store:
    // 1) sampled hull-mark internal ids (size C),
    // 2) precomputed d(c_j, u) for every node u and every mark c_j (N x C, row-major).
    std::vector<hnswlib::tableint> hull_mark_internal_ids_;
    std::vector<hnswlib::labeltype> hull_mark_labels_;
    std::vector<float> hull_mark_node_dists_;  // row-major: [u * C + j]
    size_t hull_mark_count_{0};
    size_t hull_cache_node_count_{0};
    bool hull_cache_ready_{false};
    bool hull_cache_euclidean_{true};

    // --- Mechanism-A cache on level-1 nodes ---
    // alpha_k(v): distance from v (in V1) to its k-th NN in V0 (all active nodes at level-0 graph).
    std::vector<float> level1_alpha_k_by_internal_;  // size N, +inf for non-V1 / unavailable
    std::vector<hnswlib::tableint> level1_nodes_;
    size_t level1_alpha_k_k_{0};
    size_t level1_alpha_k_ef_{0};
    bool level1_alpha_k_ready_{false};
    bool level1_alpha_k_euclidean_{true};

    // --- Mechanism-C cache on level-0 nodes ---
    // beta(u): distance from u (in V0) to its nearest level-1 node.
    // nn1(u): internal id of that nearest level-1 node.
    std::vector<float> level0_beta_by_internal_;  // size N, +inf if unavailable
    std::vector<hnswlib::tableint> level0_nn1_internal_by_internal_;  // size N, -1 if unavailable
    size_t level0_beta_nodes_cached_{0};
    size_t level0_beta_level1_count_{0};
    size_t level0_beta_ef_{0};
    bool level0_beta_ready_{false};
    bool level0_beta_euclidean_{true};

    // --- Experimental MBV scoreboard cache ---
    // Row-major by internal node id: [u * keep + j].
    // Kept separate from existing MBV APIs so this experiment is easy to revert.
    struct MbvScoreboardAnnCacheHeader {
        char magic[32];
        uint64_t version;
        uint64_t keep;
        uint64_t node_count;
        uint64_t entries;
        uint64_t ef_search;
        uint64_t euclidean;
        uint64_t tableint_bytes;
        uint64_t float_bytes;
        uint64_t ids_size;
        uint64_t dists_size;
    };
    std::vector<hnswlib::tableint> mbv_score_cache_internal_;
    std::vector<float> mbv_score_cache_dists_;
    size_t mbv_score_cache_keep_{0};
    size_t mbv_score_cache_node_count_{0};
    size_t mbv_score_cache_entries_{0};
    size_t mbv_score_cache_ef_{0};
    bool mbv_score_cache_ready_{false};
    bool mbv_score_cache_euclidean_{true};

    // --- Tribase-style cluster + subNN cache for standalone MBV pruning ---
    // Stored by internal id so the online query path stays array-indexed.
    tribase_mbv::Cache<hnswlib::tableint, dist_t> tribase_cache_;
    bool mbv_fine_timers_enabled_{true};
    bool mbv_region_timers_enabled_{false};

    // --- Legacy level-0 edge-weight cache ---
    // Some older index binaries were saved before inline edge weights were added.
    // For those indexes, MBV and Dijkstra-based rectifiers would otherwise
    // recompute edge lengths on every query. We cache the level-0 edge weights
    // once in memory so legacy indexes can use the same fast path as weighted
    // indexes without changing the on-disk binary.
    std::vector<dist_t> l0_edge_weight_cache_;
    std::vector<size_t> l0_edge_weight_offsets_;
    size_t l0_edge_weight_cache_nodes_{0};
    bool l0_edge_weight_cache_ready_{false};
    bool l0_edge_weight_cache_is_euclidean_{false};

    inline const dist_t* get_level0_edge_weight_cache_ptr(hnswlib::tableint u, bool* already_euclidean = nullptr) const {
        if (!l0_edge_weight_cache_ready_) return nullptr;
        if ((size_t)u >= l0_edge_weight_cache_nodes_) return nullptr;
        if (l0_edge_weight_offsets_.empty()) return nullptr;
        const size_t begin = l0_edge_weight_offsets_[(size_t)u];
        const size_t end = l0_edge_weight_offsets_[(size_t)u + 1];
        if (end <= begin) return nullptr;
        if (already_euclidean) *already_euclidean = l0_edge_weight_cache_is_euclidean_;
        return l0_edge_weight_cache_.data() + begin;
    }

    inline bool can_use_direct_euclidean_l0_edge_cache(size_t node_count) const {
        return l0_edge_weight_cache_ready_
            && l0_edge_weight_cache_is_euclidean_
            && l0_edge_weight_cache_nodes_ >= node_count
            && l0_edge_weight_offsets_.size() > node_count
            && !l0_edge_weight_cache_.empty();
    }

    inline const dist_t* get_direct_euclidean_l0_edge_cache_ptr(hnswlib::tableint u) const {
        return l0_edge_weight_cache_.data() + l0_edge_weight_offsets_[(size_t)u];
    }



    inline const dist_t* get_level0_weights_ptr_or_cache(
        hnswlib::linklistsizeint* ll,
        hnswlib::tableint u,
        bool* already_euclidean = nullptr
    ) const {
        const dist_t* w_arr = appr_alg->get_linklist_weights_ptr(ll, 0);
        if (w_arr) {
            if (already_euclidean) *already_euclidean = false;
            return w_arr;
        }
        return get_level0_edge_weight_cache_ptr(u, already_euclidean);
    }

    py::dict ensure_level0_edge_weight_cache(bool force = false) {
        py::dict stats;
        if (!appr_alg) throw std::runtime_error("Index not initialized");

        const size_t N = (size_t)appr_alg->cur_element_count.load();
        stats["node_count"] = (long long)N;

        hnswlib::linklistsizeint* ll0 = (N > 0) ? appr_alg->get_linklist_at_level((hnswlib::tableint)0, 0) : nullptr;
        const bool has_inline_weights = (ll0 != nullptr && appr_alg->get_linklist_weights_ptr(ll0, 0) != nullptr);
        stats["has_inline_weights"] = has_inline_weights;

        if (has_inline_weights && !force) {
            l0_edge_weight_cache_.clear();
            l0_edge_weight_offsets_.clear();
            l0_edge_weight_cache_nodes_ = 0;
            l0_edge_weight_cache_ready_ = false;
            l0_edge_weight_cache_is_euclidean_ = false;
            stats["built_cache"] = false;
            stats["reason"] = py::str("inline_weights_present");
            stats["entries"] = (long long)0;
            stats["weights_are_euclidean"] = false;
            return stats;
        }

        if (l0_edge_weight_cache_ready_ && l0_edge_weight_cache_nodes_ == N && !force) {
            stats["built_cache"] = false;
            stats["reason"] = py::str("cache_already_ready");
            stats["entries"] = (long long)l0_edge_weight_cache_.size();
            stats["weights_are_euclidean"] = l0_edge_weight_cache_is_euclidean_;
            return stats;
        }

        auto t0 = std::chrono::steady_clock::now();
        std::vector<size_t> offsets(N + 1, 0);
        size_t total = 0;
        for (size_t u = 0; u < N; u++) {
            hnswlib::linklistsizeint* ll = appr_alg->get_linklist_at_level((hnswlib::tableint)u, 0);
            const size_t sz = appr_alg->getListCount(ll);
            offsets[u] = total;
            total += sz;
        }
        offsets[N] = total;

        std::vector<dist_t> weights(total);
        const bool store_euclidean = (space_name == "l2");
        for (size_t u = 0; u < N; u++) {
            hnswlib::linklistsizeint* ll = appr_alg->get_linklist_at_level((hnswlib::tableint)u, 0);
            const size_t sz = appr_alg->getListCount(ll);
            hnswlib::tableint* nbrs = (hnswlib::tableint*)(ll + 1);
            char* udata = appr_alg->getDataByInternalId((hnswlib::tableint)u);
            const size_t base = offsets[u];
            for (size_t i = 0; i < sz; i++) {
                char* vdata = appr_alg->getDataByInternalId(nbrs[i]);
                dist_t w = appr_alg->fstdistfunc_(udata, vdata, appr_alg->dist_func_param_);
                if (store_euclidean) {
                    if (w < (dist_t)0) w = (dist_t)0;
                    w = (dist_t)std::sqrt((double)w);
                }
                weights[base + i] = w;
            }
        }
        auto t1 = std::chrono::steady_clock::now();

        l0_edge_weight_cache_.swap(weights);
        l0_edge_weight_offsets_.swap(offsets);
        l0_edge_weight_cache_nodes_ = N;
        l0_edge_weight_cache_ready_ = true;
        l0_edge_weight_cache_is_euclidean_ = store_euclidean;

        stats["built_cache"] = true;
        stats["reason"] = py::str("built_from_graph");
        stats["entries"] = (long long)l0_edge_weight_cache_.size();
        stats["weights_are_euclidean"] = l0_edge_weight_cache_is_euclidean_;
        stats["build_time_ns"] = (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        return stats;
    }

    inline void invalidate_level0_edge_weight_cache() {
        l0_edge_weight_cache_.clear();
        l0_edge_weight_offsets_.clear();
        l0_edge_weight_cache_nodes_ = 0;
        l0_edge_weight_cache_ready_ = false;
        l0_edge_weight_cache_is_euclidean_ = false;
    }














    Index(const std::string &space_name, const int dim) : space_name(space_name), dim(dim) {
        normalize = false;
        if (space_name == "l2") {
            l2space = new hnswlib::L2Space(dim);
        } else if (space_name == "ip") {
            l2space = new hnswlib::InnerProductSpace(dim);
        } else if (space_name == "cosine") {
            l2space = new hnswlib::InnerProductSpace(dim);
            normalize = true;
        } else {
            throw std::runtime_error("Space name must be one of l2, ip, or cosine.");
        }
        appr_alg = NULL;
        ep_added = true;
        index_inited = false;
        num_threads_default = std::thread::hardware_concurrency();

        default_ef = 10;
    }


    ~Index() {
        delete l2space;
        if (appr_alg)
            delete appr_alg;
    }


    void init_new_index(
        size_t maxElements,
        size_t M,
        size_t efConstruction,
        size_t random_seed,
        bool allow_replace_deleted) {
        if (appr_alg) {
            throw std::runtime_error("The index is already initiated.");
        }
        cur_l = 0;
        appr_alg = new hnswlib::HierarchicalNSW<dist_t>(l2space, maxElements, M, efConstruction, random_seed, allow_replace_deleted);
        index_inited = true;
        ep_added = false;
        appr_alg->ef_ = default_ef;
        seed = random_seed;
    }


    void set_ef(size_t ef) {
      default_ef = ef;
      if (appr_alg)
          appr_alg->ef_ = ef;
    }


    void set_num_threads(int num_threads) {
        this->num_threads_default = num_threads;
    }


    void saveIndex(const std::string &path_to_index) {
        appr_alg->saveIndex(path_to_index);
    }


    void loadIndex(const std::string &path_to_index, size_t max_elements, bool allow_replace_deleted) {
      if (appr_alg) {
          std::cerr << "Warning: Calling load_index for an already inited index. Old index is being deallocated." << std::endl;
          delete appr_alg;
      }
      appr_alg = new hnswlib::HierarchicalNSW<dist_t>(l2space, path_to_index, false, max_elements, allow_replace_deleted);
      cur_l = appr_alg->cur_element_count;
      index_inited = true;
    }


    void normalize_vector(float* data, float* norm_array) {
        float norm = 0.0f;
        for (int i = 0; i < dim; i++)
            norm += data[i] * data[i];
        norm = 1.0f / (sqrtf(norm) + 1e-30f);
        for (int i = 0; i < dim; i++)
            norm_array[i] = data[i] * norm;
    }


    void addItems(py::object input, py::object ids_ = py::none(), int num_threads = -1, bool replace_deleted = false) {
        py::array_t < dist_t, py::array::c_style | py::array::forcecast > items(input);
        auto buffer = items.request();
        if (num_threads <= 0)
            num_threads = num_threads_default;

        size_t rows, features;
        get_input_array_shapes(buffer, &rows, &features);

        if (features != dim)
            throw std::runtime_error("Wrong dimensionality of the vectors");

        // avoid using threads when the number of additions is small:
        if (rows <= num_threads * 4) {
            num_threads = 1;
        }

        std::vector<size_t> ids = get_input_ids_and_check_shapes(ids_, rows);

        {
            int start = 0;
            if (!ep_added) {
                size_t id = ids.size() ? ids.at(0) : (cur_l);
                float* vector_data = (float*)items.data(0);
                std::vector<float> norm_array(dim);
                if (normalize) {
                    normalize_vector(vector_data, norm_array.data());
                    vector_data = norm_array.data();
                }
                appr_alg->addPoint((void*)vector_data, (size_t)id, replace_deleted);
                start = 1;
                ep_added = true;
            }

            py::gil_scoped_release l;
            if (normalize == false) {
                ParallelFor(start, rows, num_threads, [&](size_t row, size_t threadId) {
                    size_t id = ids.size() ? ids.at(row) : (cur_l + row);
                    appr_alg->addPoint((void*)items.data(row), (size_t)id, replace_deleted);
                    });
            } else {
                std::vector<float> norm_array(num_threads * dim);
                ParallelFor(start, rows, num_threads, [&](size_t row, size_t threadId) {
                    // normalize vector:
                    size_t start_idx = threadId * dim;
                    normalize_vector((float*)items.data(row), (norm_array.data() + start_idx));

                    size_t id = ids.size() ? ids.at(row) : (cur_l + row);
                    appr_alg->addPoint((void*)(norm_array.data() + start_idx), (size_t)id, replace_deleted);
                    });
            }
            cur_l += rows;
        }
    }





    // -----------------------------
    // Exporting graph edges
    // -----------------------------



    // -----------------------------
    // START: Post-hoc certification bindings (NEW)
    // -----------------------------







    // -----------------------------
    // END: Post-hoc certification bindings (NEW)
    // -----------------------------


    py::dict getAnnData() const { /* WARNING: Index::getAnnData is not thread-safe with Index::addItems */
        std::unique_lock <std::mutex> templock(appr_alg->global);

        size_t level0_npy_size = appr_alg->cur_element_count * appr_alg->size_data_per_element_;
        size_t link_npy_size = 0;
        std::vector<size_t> link_npy_offsets(appr_alg->cur_element_count);

        for (size_t i = 0; i < appr_alg->cur_element_count; i++) {
            size_t linkListSize = appr_alg->element_levels_[i] > 0 ? appr_alg->size_links_per_element_ * appr_alg->element_levels_[i] : 0;
            link_npy_offsets[i] = link_npy_size;
            if (linkListSize)
                link_npy_size += linkListSize;
        }

        char* data_level0_npy = (char*)malloc(level0_npy_size);
        char* link_list_npy = (char*)malloc(link_npy_size);
        int* element_levels_npy = (int*)malloc(appr_alg->element_levels_.size() * sizeof(int));

        hnswlib::labeltype* label_lookup_key_npy = (hnswlib::labeltype*)malloc(appr_alg->label_lookup_.size() * sizeof(hnswlib::labeltype));
        hnswlib::tableint* label_lookup_val_npy = (hnswlib::tableint*)malloc(appr_alg->label_lookup_.size() * sizeof(hnswlib::tableint));

        memset(label_lookup_key_npy, -1, appr_alg->label_lookup_.size() * sizeof(hnswlib::labeltype));
        memset(label_lookup_val_npy, -1, appr_alg->label_lookup_.size() * sizeof(hnswlib::tableint));

        size_t idx = 0;
        for (auto it = appr_alg->label_lookup_.begin(); it != appr_alg->label_lookup_.end(); ++it) {
            label_lookup_key_npy[idx] = it->first;
            label_lookup_val_npy[idx] = it->second;
            idx++;
        }

        memset(link_list_npy, 0, link_npy_size);

        memcpy(data_level0_npy, appr_alg->data_level0_memory_, level0_npy_size);
        memcpy(element_levels_npy, appr_alg->element_levels_.data(), appr_alg->element_levels_.size() * sizeof(int));

        for (size_t i = 0; i < appr_alg->cur_element_count; i++) {
            size_t linkListSize = appr_alg->element_levels_[i] > 0 ? appr_alg->size_links_per_element_ * appr_alg->element_levels_[i] : 0;
            if (linkListSize) {
                memcpy(link_list_npy + link_npy_offsets[i], appr_alg->linkLists_[i], linkListSize);
            }
        }

        py::capsule free_when_done_l0(data_level0_npy, [](void* f) {
            delete[] f;
            });
        py::capsule free_when_done_lvl(element_levels_npy, [](void* f) {
            delete[] f;
            });
        py::capsule free_when_done_lb(label_lookup_key_npy, [](void* f) {
            delete[] f;
            });
        py::capsule free_when_done_id(label_lookup_val_npy, [](void* f) {
            delete[] f;
            });
        py::capsule free_when_done_ll(link_list_npy, [](void* f) {
            delete[] f;
            });

        /*  TODO: serialize state of random generators appr_alg->level_generator_ and appr_alg->update_probability_generator_  */
        /*        for full reproducibility / to avoid re-initializing generators inside Index::createFromParams         */

        return py::dict(
            "offset_level0"_a = appr_alg->offsetLevel0_,
            "max_elements"_a = appr_alg->max_elements_,
            "cur_element_count"_a = (size_t)appr_alg->cur_element_count,
            "size_data_per_element"_a = appr_alg->size_data_per_element_,
            "label_offset"_a = appr_alg->label_offset_,
            "offset_data"_a = appr_alg->offsetData_,
            "max_level"_a = appr_alg->maxlevel_,
            "enterpoint_node"_a = appr_alg->enterpoint_node_,
            "max_M"_a = appr_alg->maxM_,
            "max_M0"_a = appr_alg->maxM0_,
            "M"_a = appr_alg->M_,
            "mult"_a = appr_alg->mult_,
            "ef_construction"_a = appr_alg->ef_construction_,
            "ef"_a = appr_alg->ef_,
            "has_deletions"_a = (bool)appr_alg->num_deleted_,
            "size_links_per_element"_a = appr_alg->size_links_per_element_,
            "allow_replace_deleted"_a = appr_alg->allow_replace_deleted_,

            "label_lookup_external"_a = py::array_t<hnswlib::labeltype>(
                { appr_alg->label_lookup_.size() },  // shape
                { sizeof(hnswlib::labeltype) },  // C-style contiguous strides for each index
                label_lookup_key_npy,  // the data pointer
                free_when_done_lb),

            "label_lookup_internal"_a = py::array_t<hnswlib::tableint>(
                { appr_alg->label_lookup_.size() },  // shape
                { sizeof(hnswlib::tableint) },  // C-style contiguous strides for each index
                label_lookup_val_npy,  // the data pointer
                free_when_done_id),

            "element_levels"_a = py::array_t<int>(
                { appr_alg->element_levels_.size() },  // shape
                { sizeof(int) },  // C-style contiguous strides for each index
                element_levels_npy,  // the data pointer
                free_when_done_lvl),

            // linkLists_,element_levels_,data_level0_memory_
            "data_level0"_a = py::array_t<char>(
                { level0_npy_size },  // shape
                { sizeof(char) },  // C-style contiguous strides for each index
                data_level0_npy,  // the data pointer
                free_when_done_l0),

            "link_lists"_a = py::array_t<char>(
                { link_npy_size },  // shape
                { sizeof(char) },  // C-style contiguous strides for each index
                link_list_npy,  // the data pointer
                free_when_done_ll));
    }


    py::dict getIndexParams() const { /* WARNING: Index::getAnnData is not thread-safe with Index::addItems */
        auto params = py::dict(
            "ser_version"_a = py::int_(Index<float>::ser_version),  // serialization version
            "space"_a = space_name,
            "dim"_a = dim,
            "index_inited"_a = index_inited,
            "ep_added"_a = ep_added,
            "normalize"_a = normalize,
            "num_threads"_a = num_threads_default,
            "seed"_a = seed);

        if (index_inited == false)
            return py::dict(**params, "ef"_a = default_ef);

        auto ann_params = getAnnData();

        return py::dict(**params, **ann_params);
    }


    static Index<float>* createFromParams(const py::dict d) {
        // check serialization version
        assert_true(((int)py::int_(Index<float>::ser_version)) >= d["ser_version"].cast<int>(), "Invalid serialization version!");

        auto space_name_ = d["space"].cast<std::string>();
        auto dim_ = d["dim"].cast<int>();
        auto index_inited_ = d["index_inited"].cast<bool>();

        Index<float>* new_index = new Index<float>(space_name_, dim_);

        /*  TODO: deserialize state of random generators into new_index->level_generator_ and new_index->update_probability_generator_  */
        /*        for full reproducibility / state of generators is serialized inside Index::getIndexParams                      */
        new_index->seed = d["seed"].cast<size_t>();

        if (index_inited_) {
            new_index->appr_alg = new hnswlib::HierarchicalNSW<dist_t>(
                new_index->l2space,
                d["max_elements"].cast<size_t>(),
                d["M"].cast<size_t>(),
                d["ef_construction"].cast<size_t>(),
                new_index->seed);
            new_index->cur_l = d["cur_element_count"].cast<size_t>();
        }

        new_index->index_inited = index_inited_;
        new_index->ep_added = d["ep_added"].cast<bool>();
        new_index->num_threads_default = d["num_threads"].cast<int>();
        new_index->default_ef = d["ef"].cast<size_t>();

        if (index_inited_)
            new_index->setAnnData(d);

        return new_index;
    }


    static Index<float> * createFromIndex(const Index<float> & index) {
        return createFromParams(index.getIndexParams());
    }


    void setAnnData(const py::dict d) { /* WARNING: Index::setAnnData is not thread-safe with Index::addItems */
        std::unique_lock <std::mutex> templock(appr_alg->global);

        assert_true(appr_alg->offsetLevel0_ == d["offset_level0"].cast<size_t>(), "Invalid value of offsetLevel0_ ");
        assert_true(appr_alg->max_elements_ == d["max_elements"].cast<size_t>(), "Invalid value of max_elements_ ");

        appr_alg->cur_element_count = d["cur_element_count"].cast<size_t>();

        assert_true(appr_alg->size_data_per_element_ == d["size_data_per_element"].cast<size_t>(), "Invalid value of size_data_per_element_ ");
        assert_true(appr_alg->label_offset_ == d["label_offset"].cast<size_t>(), "Invalid value of label_offset_ ");
        assert_true(appr_alg->offsetData_ == d["offset_data"].cast<size_t>(), "Invalid value of offsetData_ ");

        appr_alg->maxlevel_ = d["max_level"].cast<int>();
        appr_alg->enterpoint_node_ = d["enterpoint_node"].cast<hnswlib::tableint>();

        assert_true(appr_alg->maxM_ == d["max_M"].cast<size_t>(), "Invalid value of maxM_ ");
        assert_true(appr_alg->maxM0_ == d["max_M0"].cast<size_t>(), "Invalid value of maxM0_ ");
        assert_true(appr_alg->M_ == d["M"].cast<size_t>(), "Invalid value of M_ ");
        assert_true(appr_alg->mult_ == d["mult"].cast<double>(), "Invalid value of mult_ ");
        assert_true(appr_alg->ef_construction_ == d["ef_construction"].cast<size_t>(), "Invalid value of ef_construction_ ");

        appr_alg->ef_ = d["ef"].cast<size_t>();

        assert_true(appr_alg->size_links_per_element_ == d["size_links_per_element"].cast<size_t>(), "Invalid value of size_links_per_element_ ");

        auto label_lookup_key_npy = d["label_lookup_external"].cast<py::array_t < hnswlib::labeltype, py::array::c_style | py::array::forcecast > >();
        auto label_lookup_val_npy = d["label_lookup_internal"].cast<py::array_t < hnswlib::tableint, py::array::c_style | py::array::forcecast > >();
        auto element_levels_npy = d["element_levels"].cast<py::array_t < int, py::array::c_style | py::array::forcecast > >();
        auto data_level0_npy = d["data_level0"].cast<py::array_t < char, py::array::c_style | py::array::forcecast > >();
        auto link_list_npy = d["link_lists"].cast<py::array_t < char, py::array::c_style | py::array::forcecast > >();

        for (size_t i = 0; i < appr_alg->cur_element_count; i++) {
            if (label_lookup_val_npy.data()[i] < 0) {
                throw std::runtime_error("Internal id cannot be negative!");
            } else {
                appr_alg->label_lookup_.insert(std::make_pair(label_lookup_key_npy.data()[i], label_lookup_val_npy.data()[i]));
            }
        }

        memcpy(appr_alg->element_levels_.data(), element_levels_npy.data(), element_levels_npy.nbytes());

        size_t link_npy_size = 0;
        std::vector<size_t> link_npy_offsets(appr_alg->cur_element_count);

        for (size_t i = 0; i < appr_alg->cur_element_count; i++) {
            size_t linkListSize = appr_alg->element_levels_[i] > 0 ? appr_alg->size_links_per_element_ * appr_alg->element_levels_[i] : 0;
            link_npy_offsets[i] = link_npy_size;
            if (linkListSize)
                link_npy_size += linkListSize;
        }

        memcpy(appr_alg->data_level0_memory_, data_level0_npy.data(), data_level0_npy.nbytes());

        for (size_t i = 0; i < appr_alg->max_elements_; i++) {
            size_t linkListSize = appr_alg->element_levels_[i] > 0 ? appr_alg->size_links_per_element_ * appr_alg->element_levels_[i] : 0;
            if (linkListSize == 0) {
                appr_alg->linkLists_[i] = nullptr;
            } else {
                appr_alg->linkLists_[i] = (char*)malloc(linkListSize);
                if (appr_alg->linkLists_[i] == nullptr)
                    throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");

                memcpy(appr_alg->linkLists_[i], link_list_npy.data() + link_npy_offsets[i], linkListSize);
            }
        }

        // process deleted elements
        bool allow_replace_deleted = false;
        if (d.contains("allow_replace_deleted")) {
            allow_replace_deleted = d["allow_replace_deleted"].cast<bool>();
        }
        appr_alg->allow_replace_deleted_= allow_replace_deleted;

        appr_alg->num_deleted_ = 0;
        bool has_deletions = d["has_deletions"].cast<bool>();
        if (has_deletions) {
            for (size_t i = 0; i < appr_alg->cur_element_count; i++) {
                if (appr_alg->isMarkedDeleted(i)) {
                    appr_alg->num_deleted_ += 1;
                    if (allow_replace_deleted) appr_alg->deleted_elements.insert(i);
                }
            }
        }
    }


    py::object knnQuery_return_numpy(
        py::object input,
        size_t k = 1,
        int num_threads = -1,
        const std::function<bool(hnswlib::labeltype)>& filter = nullptr) {
        py::array_t < dist_t, py::array::c_style | py::array::forcecast > items(input);
        auto buffer = items.request();
        hnswlib::labeltype* data_numpy_l;
        dist_t* data_numpy_d;
        size_t rows, features;

        if (num_threads <= 0)
            num_threads = num_threads_default;

        {
            py::gil_scoped_release l;
            get_input_array_shapes(buffer, &rows, &features);

            // avoid using threads when the number of searches is small:
            if (rows <= num_threads * 4) {
                num_threads = 1;
            }

            data_numpy_l = new hnswlib::labeltype[rows * k];
            data_numpy_d = new dist_t[rows * k];

            // Warning: search with a filter works slow in python in multithreaded mode. For best performance set num_threads=1
            CustomFilterFunctor idFilter(filter);
            CustomFilterFunctor* p_idFilter = filter ? &idFilter : nullptr;

            if (normalize == false) {
                ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId) {
                    std::priority_queue<std::pair<dist_t, hnswlib::labeltype >> result = appr_alg->searchKnn(
                        (void*)items.data(row), k, p_idFilter);
                    if (result.size() != k)
                        throw std::runtime_error(
                            "Cannot return the results in a contiguous 2D array. Probably ef or M is too small");
                    for (int i = (int)k - 1; i >= 0; i--) {
                        auto& result_tuple = result.top();
                        data_numpy_d[row * k + i] = result_tuple.first;
                        data_numpy_l[row * k + i] = result_tuple.second;
                        result.pop();
                    }
                });
            } else {
                std::vector<float> norm_array(num_threads * features);
                ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId) {
                    size_t start_idx = threadId * dim;
                    normalize_vector((float*)items.data(row), (norm_array.data() + start_idx));

                    std::priority_queue<std::pair<dist_t, hnswlib::labeltype >> result = appr_alg->searchKnn(
                        (void*)(norm_array.data() + start_idx), k, p_idFilter);
                    if (result.size() != k)
                        throw std::runtime_error(
                            "Cannot return the results in a contiguous 2D array. Probably ef or M is too small");
                    for (int i = (int)k - 1; i >= 0; i--) {
                        auto& result_tuple = result.top();
                        data_numpy_d[row * k + i] = result_tuple.first;
                        data_numpy_l[row * k + i] = result_tuple.second;
                        result.pop();
                    }
                });
            }
        }
        py::capsule free_when_done_l(data_numpy_l, [](void* f) { delete[] f; });
        py::capsule free_when_done_d(data_numpy_d, [](void* f) { delete[] f; });

        return py::make_tuple(
            py::array_t<hnswlib::labeltype>(
                { rows, k },
                { k * sizeof(hnswlib::labeltype), sizeof(hnswlib::labeltype) },
                data_numpy_l,
                free_when_done_l),
            py::array_t<dist_t>(
                { rows, k },
                { k * sizeof(dist_t), sizeof(dist_t) },
                data_numpy_d,
                free_when_done_d));
    }









    size_t getMaxElements() const {
        return appr_alg->max_elements_;
    }


    size_t getCurrentCount() const {
        return appr_alg->cur_element_count;
    }


    // =============================
    // EXPERIMENT HOT-PATH (C++): Dijkstra + optional PCA prune + verification
    // =============================

    inline hnswlib::tableint label_to_internal_(hnswlib::labeltype label) const {
        if (!appr_alg) throw std::runtime_error("Index not initialized");
        std::unique_lock<std::mutex> lock_table(appr_alg->label_lookup_lock);
        auto it = appr_alg->label_lookup_.find(label);
        if (it == appr_alg->label_lookup_.end() || appr_alg->isMarkedDeleted(it->second)) {
            throw std::runtime_error("Label not found");
        }
        return it->second;
    }

    inline dist_t maybe_euclid_(dist_t d, bool euclidean) const {
        if (!euclidean) return d;
        if (space_name != "l2") return d;
        if (d < (dist_t)0) d = (dist_t)0;
        return (dist_t)std::sqrt(d);
    }

    struct Level0DirectedRemoveResult {
        bool existed = false;
        bool removed = false;
    };

    struct Level0BidirectionalRemoveResult {
        Level0DirectedRemoveResult uv;
        Level0DirectedRemoveResult vu;
    };

    struct Level0DirectedUpsertResult {
        bool existed = false;
        bool kept = false;
    };

    struct Level0BidirectionalUpsertResult {
        Level0DirectedUpsertResult uv;
        Level0DirectedUpsertResult vu;
        dist_t weight_raw = (dist_t)0;
    };

    long long label_to_internal(hnswlib::labeltype label) const {
        return (long long)label_to_internal_(label);
    }

    long long internal_to_label(hnswlib::tableint internal_id) const {
        if (!appr_alg) throw std::runtime_error("Index not initialized");
        const size_t n = (size_t)appr_alg->cur_element_count.load();
        if ((size_t)internal_id >= n) {
            throw std::runtime_error("internal id out of range");
        }
        if (appr_alg->isMarkedDeleted(internal_id)) {
            throw std::runtime_error("internal id is marked deleted");
        }
        return (long long)appr_alg->getExternalLabel(internal_id);
    }

    py::dict get_enterpoint_info() const {
        if (!appr_alg) throw std::runtime_error("Index not initialized");
        const hnswlib::tableint ep = appr_alg->enterpoint_node_;
        const size_t n = (size_t)appr_alg->cur_element_count.load();
        const bool valid = (ep != (hnswlib::tableint)-1) && ((size_t)ep < n);
        return py::dict(
            "enterpoint_label"_a = (valid ? (long long)appr_alg->getExternalLabel(ep) : -1),
            "enterpoint_internal"_a = (valid ? (long long)ep : -1),
            "maxlevel"_a = appr_alg->maxlevel_,
            "entrypoint_node_level"_a = (valid ? appr_alg->element_levels_[ep] : -1));
    }

    py::dict set_seeded_enterpoint_at_max_level(uint64_t random_seed = 0) {
        if (!appr_alg) throw std::runtime_error("Index not initialized");
        const size_t n = (size_t)appr_alg->cur_element_count.load();
        const int maxlevel = appr_alg->maxlevel_;
        if (n == 0 || maxlevel < 0) throw std::runtime_error("Index is empty");

        std::vector<hnswlib::tableint> candidates;
        candidates.reserve(16);
        for (size_t i = 0; i < n; i++) {
            if (appr_alg->element_levels_[i] == maxlevel && !appr_alg->isMarkedDeleted((hnswlib::tableint)i)) {
                candidates.push_back((hnswlib::tableint)i);
            }
        }
        if (candidates.empty()) throw std::runtime_error("No active nodes found at max level");

        std::mt19937_64 rng(random_seed);
        std::uniform_int_distribution<size_t> pick(0, candidates.size() - 1);
        const hnswlib::tableint selected = candidates[pick(rng)];

        std::unique_lock<std::mutex> global_lock(appr_alg->global);
        const hnswlib::tableint old_ep = appr_alg->enterpoint_node_;
        const bool old_valid = old_ep != (hnswlib::tableint)-1 && ((size_t)old_ep < n);
        appr_alg->enterpoint_node_ = selected;
        appr_alg->maxlevel_ = maxlevel;

        return py::dict(
            "seed"_a = random_seed,
            "old_enterpoint_label"_a = (old_valid ? (long long)appr_alg->getExternalLabel(old_ep) : -1),
            "old_enterpoint_internal"_a = (old_valid ? (long long)old_ep : -1),
            "enterpoint_label"_a = (long long)appr_alg->getExternalLabel(selected),
            "enterpoint_internal"_a = (long long)selected,
            "maxlevel"_a = maxlevel,
            "candidate_count"_a = candidates.size());
    }

    py::array_t<hnswlib::tableint> get_level0_neighbors_internal(hnswlib::tableint internal_id) const {
        if (!appr_alg) throw std::runtime_error("Index not initialized");
        const size_t n = (size_t)appr_alg->cur_element_count.load();
        if ((size_t)internal_id >= n) {
            throw std::runtime_error("internal id out of range");
        }

        std::unique_lock<std::mutex> lock_links(appr_alg->link_list_locks_[internal_id]);
        hnswlib::linklistsizeint* ll = appr_alg->get_linklist0(internal_id);
        const size_t sz = appr_alg->getListCount(ll);
        hnswlib::tableint* nbrs = (hnswlib::tableint*)(ll + 1);

        std::vector<hnswlib::tableint> out;
        out.reserve(sz);
        for (size_t i = 0; i < sz; i++) {
            const hnswlib::tableint nb = nbrs[i];
            if ((size_t)nb >= n) continue;
            if (appr_alg->isMarkedDeleted(nb)) continue;
            out.push_back(nb);
        }

        py::array_t<hnswlib::tableint> arr(out.size());
        if (!out.empty()) {
            std::memcpy(arr.mutable_data(), out.data(), out.size() * sizeof(hnswlib::tableint));
        }
        return arr;
    }

    std::vector<hnswlib::labeltype> get_links(hnswlib::labeltype label, int level = 0) {
        if (!appr_alg) throw std::runtime_error("Index not initialized");

        std::unique_lock<std::mutex> lock_label(appr_alg->getLabelOpMutex(label));
        const hnswlib::tableint internal_id = label_to_internal_(label);
        if (level < 0) throw std::runtime_error("level must be >= 0");
        if (level > appr_alg->element_levels_[internal_id]) {
            throw std::runtime_error("Requested level exceeds node level");
        }

        std::unique_lock<std::mutex> lock_links(appr_alg->link_list_locks_[internal_id]);
        hnswlib::linklistsizeint* ll = appr_alg->get_linklist_at_level(internal_id, level);
        const size_t sz = appr_alg->getListCount(ll);
        hnswlib::tableint* nbrs = (hnswlib::tableint*)(ll + 1);
        const size_t n = (size_t)appr_alg->cur_element_count.load();

        std::vector<hnswlib::labeltype> out;
        out.reserve(sz);
        for (size_t i = 0; i < sz; i++) {
            const hnswlib::tableint nb = nbrs[i];
            if ((size_t)nb >= n) continue;
            if (appr_alg->isMarkedDeleted(nb)) continue;
            out.push_back(appr_alg->getExternalLabel(nb));
        }
        return out;
    }

    py::tuple get_level0_neighbors_internal_with_weights(
        hnswlib::tableint internal_id,
        bool euclidean = true
    ) const {
        if (!appr_alg) throw std::runtime_error("Index not initialized");
        const size_t n = (size_t)appr_alg->cur_element_count.load();
        if ((size_t)internal_id >= n) {
            throw std::runtime_error("internal id out of range");
        }
        if (euclidean && space_name != "l2") {
            throw std::runtime_error("euclidean=True is only supported for space='l2'");
        }

        std::unique_lock<std::mutex> lock_links(appr_alg->link_list_locks_[internal_id]);
        hnswlib::linklistsizeint* ll = appr_alg->get_linklist0(internal_id);
        const size_t sz = appr_alg->getListCount(ll);
        hnswlib::tableint* nbrs = (hnswlib::tableint*)(ll + 1);
        bool weights_are_euclidean = false;
        const dist_t* w_arr = get_level0_weights_ptr_or_cache(ll, internal_id, &weights_are_euclidean);
        char* udata = (w_arr == nullptr) ? appr_alg->getDataByInternalId(internal_id) : nullptr;

        std::vector<hnswlib::tableint> out_ids;
        std::vector<dist_t> out_w;
        out_ids.reserve(sz);
        out_w.reserve(sz);
        for (size_t i = 0; i < sz; i++) {
            const hnswlib::tableint nb = nbrs[i];
            if ((size_t)nb >= n) continue;
            if (appr_alg->isMarkedDeleted(nb)) continue;

            dist_t d = (dist_t)0;
            if (w_arr != nullptr) {
                d = w_arr[i];
                if (euclidean && !weights_are_euclidean) {
                    d = maybe_euclid_(d, true);
                }
            } else {
                d = appr_alg->fstdistfunc_(udata, appr_alg->getDataByInternalId(nb), appr_alg->dist_func_param_);
                d = maybe_euclid_(d, euclidean);
            }

            out_ids.push_back(nb);
            out_w.push_back(d);
        }

        py::array_t<hnswlib::tableint> ids_arr(out_ids.size());
        py::array_t<dist_t> w_arr_py(out_w.size());
        if (!out_ids.empty()) {
            std::memcpy(ids_arr.mutable_data(), out_ids.data(), out_ids.size() * sizeof(hnswlib::tableint));
            std::memcpy(w_arr_py.mutable_data(), out_w.data(), out_w.size() * sizeof(dist_t));
        }
        return py::make_tuple(ids_arr, w_arr_py);
    }

    Level0DirectedUpsertResult upsert_level0_neighbor_locked(
        hnswlib::tableint src,
        hnswlib::tableint dst,
        dist_t weight_raw
    ) {
        Level0DirectedUpsertResult res;
        if (src == dst) return res;

        hnswlib::linklistsizeint* ll = appr_alg->get_linklist0(src);
        const size_t sz = appr_alg->getListCount(ll);
        hnswlib::tableint* ids = (hnswlib::tableint*)(ll + 1);
        dist_t* ws = appr_alg->get_linklist_weights_ptr(ll, 0);

        for (size_t j = 0; j < sz; j++) {
            if (ids[j] == dst) {
                res.existed = true;
                res.kept = true;
                if (ws) ws[j] = weight_raw;
                return res;
            }
        }

        const size_t Mcurmax = appr_alg->maxM0_;
        if (sz < Mcurmax) {
            ids[sz] = dst;
            if (ws) ws[sz] = weight_raw;
            appr_alg->setListCount(ll, (unsigned short)(sz + 1));
            res.kept = true;
            return res;
        }

        std::priority_queue<
            std::pair<dist_t, hnswlib::tableint>,
            std::vector<std::pair<dist_t, hnswlib::tableint>>,
            typename hnswlib::HierarchicalNSW<dist_t>::CompareByFirst
        > candidates;
        candidates.emplace(weight_raw, dst);

        char* src_data = ws ? nullptr : appr_alg->getDataByInternalId(src);
        for (size_t j = 0; j < sz; j++) {
            const hnswlib::tableint nb = ids[j];
            const dist_t d = ws
                ? ws[j]
                : appr_alg->fstdistfunc_(src_data, appr_alg->getDataByInternalId(nb), appr_alg->dist_func_param_);
            candidates.emplace(d, nb);
        }

        appr_alg->getNeighborsByHeuristic2(candidates, Mcurmax);

        size_t out = 0;
        bool kept = false;
        while (!candidates.empty()) {
            ids[out] = candidates.top().second;
            if (ws) ws[out] = candidates.top().first;
            if (candidates.top().second == dst) kept = true;
            candidates.pop();
            out++;
        }
        appr_alg->setListCount(ll, (unsigned short)out);
        res.kept = kept;
        return res;
    }

    Level0BidirectionalUpsertResult upsert_level0_edge_pair_internal(
        hnswlib::tableint u,
        hnswlib::tableint v
    ) {
        Level0BidirectionalUpsertResult res;
        if (u == v) return res;

        char* udata = appr_alg->getDataByInternalId(u);
        char* vdata = appr_alg->getDataByInternalId(v);
        res.weight_raw = appr_alg->fstdistfunc_(udata, vdata, appr_alg->dist_func_param_);

        const hnswlib::tableint lo = std::min(u, v);
        const hnswlib::tableint hi = std::max(u, v);
        std::unique_lock<std::mutex> lock_lo(appr_alg->link_list_locks_[lo], std::defer_lock);
        std::unique_lock<std::mutex> lock_hi(appr_alg->link_list_locks_[hi], std::defer_lock);
        std::lock(lock_lo, lock_hi);
        res.uv = upsert_level0_neighbor_locked(u, v, res.weight_raw);
        res.vu = upsert_level0_neighbor_locked(v, u, res.weight_raw);
        return res;
    }

    Level0DirectedRemoveResult remove_level0_neighbor_locked(
        hnswlib::tableint src,
        hnswlib::tableint dst
    ) {
        Level0DirectedRemoveResult res;
        if (src == dst) return res;

        hnswlib::linklistsizeint* ll = appr_alg->get_linklist0(src);
        const size_t sz = appr_alg->getListCount(ll);
        hnswlib::tableint* ids = (hnswlib::tableint*)(ll + 1);
        dist_t* ws = appr_alg->get_linklist_weights_ptr(ll, 0);

        for (size_t j = 0; j < sz; j++) {
            if (ids[j] != dst) continue;
            res.existed = true;
            const size_t last = sz - 1;
            if (j != last) {
                ids[j] = ids[last];
                if (ws) ws[j] = ws[last];
            }
            appr_alg->setListCount(ll, (unsigned short)last);
            res.removed = true;
            return res;
        }
        return res;
    }

    Level0BidirectionalRemoveResult remove_level0_edge_pair_internal(
        hnswlib::tableint u,
        hnswlib::tableint v
    ) {
        Level0BidirectionalRemoveResult res;
        if (u == v) return res;

        const hnswlib::tableint lo = std::min(u, v);
        const hnswlib::tableint hi = std::max(u, v);
        std::unique_lock<std::mutex> lock_lo(appr_alg->link_list_locks_[lo], std::defer_lock);
        std::unique_lock<std::mutex> lock_hi(appr_alg->link_list_locks_[hi], std::defer_lock);
        std::lock(lock_lo, lock_hi);
        res.uv = remove_level0_neighbor_locked(u, v);
        res.vu = remove_level0_neighbor_locked(v, u);
        return res;
    }

    py::dict add_level0_edge_bidirectional(hnswlib::labeltype u_label, hnswlib::labeltype v_label, bool euclidean = true) {
        if (!appr_alg) throw std::runtime_error("Index not initialized");
        if (euclidean && space_name != "l2") {
            throw std::runtime_error("euclidean=True is only supported for space='l2'");
        }

        const hnswlib::tableint u = label_to_internal_(u_label);
        const hnswlib::tableint v = label_to_internal_(v_label);
        auto res = upsert_level0_edge_pair_internal(u, v);
        invalidate_level0_edge_weight_cache();

        py::dict out;
        out["u_label"] = u_label;
        out["v_label"] = v_label;
        out["same_node"] = (u == v);
        out["uv_existed"] = res.uv.existed;
        out["vu_existed"] = res.vu.existed;
        out["uv_kept"] = res.uv.kept;
        out["vu_kept"] = res.vu.kept;
        out["weight"] = (double)maybe_euclid_(res.weight_raw, euclidean);
        return out;
    }

    py::dict remove_level0_edge_bidirectional(hnswlib::labeltype u_label, hnswlib::labeltype v_label) {
        if (!appr_alg) throw std::runtime_error("Index not initialized");

        const hnswlib::tableint u = label_to_internal_(u_label);
        const hnswlib::tableint v = label_to_internal_(v_label);
        auto res = remove_level0_edge_pair_internal(u, v);
        invalidate_level0_edge_weight_cache();

        py::dict out;
        out["u_label"] = u_label;
        out["v_label"] = v_label;
        out["same_node"] = (u == v);
        out["uv_existed"] = res.uv.existed;
        out["vu_existed"] = res.vu.existed;
        out["uv_removed"] = res.uv.removed;
        out["vu_removed"] = res.vu.removed;
        return out;
    }

    py::dict remove_level0_edge_internal(hnswlib::tableint src, hnswlib::tableint dst) {
        if (!appr_alg) throw std::runtime_error("Index not initialized");
        const size_t n = (size_t)appr_alg->cur_element_count.load();
        if ((size_t)src >= n || (size_t)dst >= n) {
            throw std::runtime_error("internal id out of range");
        }

        const hnswlib::tableint only = src;
        std::unique_lock<std::mutex> lock(appr_alg->link_list_locks_[only]);
        auto res = remove_level0_neighbor_locked(src, dst);
        lock.unlock();
        invalidate_level0_edge_weight_cache();

        py::dict out;
        out["src_internal"] = (long long)src;
        out["dst_internal"] = (long long)dst;
        out["same_node"] = (src == dst);
        out["existed"] = res.existed;
        out["removed"] = res.removed;
        return out;
    }

    py::dict remove_level0_edge_bidirectional_internal(hnswlib::tableint u, hnswlib::tableint v) {
        if (!appr_alg) throw std::runtime_error("Index not initialized");
        const size_t n = (size_t)appr_alg->cur_element_count.load();
        if ((size_t)u >= n || (size_t)v >= n) {
            throw std::runtime_error("internal id out of range");
        }
        auto res = remove_level0_edge_pair_internal(u, v);
        invalidate_level0_edge_weight_cache();

        py::dict out;
        out["u_internal"] = (long long)u;
        out["v_internal"] = (long long)v;
        out["same_node"] = (u == v);
        out["uv_existed"] = res.uv.existed;
        out["vu_existed"] = res.vu.existed;
        out["uv_removed"] = res.uv.removed;
        out["vu_removed"] = res.vu.removed;
        return out;
    }

















    void set_mbv_fine_timers_enabled(bool enabled) {
        mbv_fine_timers_enabled_ = enabled;
    }

    bool get_mbv_fine_timers_enabled() const {
        return mbv_fine_timers_enabled_;
    }

    void set_mbv_region_timers_enabled(bool enabled) {
        mbv_region_timers_enabled_ = enabled;
    }

    bool get_mbv_region_timers_enabled() const {
        return mbv_region_timers_enabled_;
    }



    // Mechanism C precompute:
    // For each u in V0, compute/store:
    // - beta(u): distance to nearest level-1 node,
    // - NN1(u): internal id of that nearest level-1 node.
    // Uses one filtered 1-NN query per active u against label-set(V1).

    // Mechanism A precompute:
    // For each v in V1 (nodes with level >= 1), compute/store alpha_k(v):
    // distance from v to its k-th nearest neighbor in V0 using one ANN kNN query.

    // Precompute a convex-hull landmark cache for MBV_hull_query_ex:
    // 1) Build a hull-mark pool via directional extreme-point search over a sampled node set.
    // 2) Sample hull_sample_size marks from that pool.
    // 3) Precompute d(c_j, u) for every node u and every mark c_j.

    py::tuple get_last_query_computed_dists(bool euclidean = true) const {
        if (!appr_alg) throw std::runtime_error("Index not initialized");

        std::vector<hnswlib::tableint> ids_internal;
        std::vector<dist_t> dists_sq;
        appr_alg->getLastQueryComputedDists(ids_internal, dists_sq);

        std::vector<hnswlib::labeltype> labels;
        labels.reserve(ids_internal.size());

        for (size_t i = 0; i < ids_internal.size(); i++) {
            labels.push_back(appr_alg->getExternalLabel(ids_internal[i]));
            dists_sq[i] = maybe_euclid_(dists_sq[i], euclidean);
        }

        py::array_t<hnswlib::labeltype> labels_arr(labels.size());
        py::array_t<dist_t> dists_arr(dists_sq.size());

        std::memcpy(labels_arr.mutable_data(), labels.data(), labels.size() * sizeof(hnswlib::labeltype));
        std::memcpy(dists_arr.mutable_data(), dists_sq.data(), dists_sq.size() * sizeof(dist_t));

        return py::make_tuple(labels_arr, dists_arr);
    }

    py::tuple get_last_popped_candidate_dists(bool euclidean = true) const {
        if (!appr_alg) throw std::runtime_error("Index not initialized");

        std::vector<hnswlib::tableint> ids_internal;
        std::vector<dist_t> dists_sq;
        appr_alg->getLastPoppedCandidateDists(ids_internal, dists_sq);

        std::vector<hnswlib::labeltype> labels;
        labels.reserve(ids_internal.size());

        for (size_t i = 0; i < ids_internal.size(); i++) {
            labels.push_back(appr_alg->getExternalLabel(ids_internal[i]));
            dists_sq[i] = maybe_euclid_(dists_sq[i], euclidean);
        }

        py::array_t<hnswlib::labeltype> labels_arr(labels.size());
        py::array_t<dist_t> dists_arr(dists_sq.size());

        std::memcpy(labels_arr.mutable_data(), labels.data(), labels.size() * sizeof(hnswlib::labeltype));
        std::memcpy(dists_arr.mutable_data(), dists_sq.data(), dists_sq.size() * sizeof(dist_t));

        return py::make_tuple(labels_arr, dists_arr);
    }

    py::array_t<dist_t> get_last_candidate_queue_dists(bool euclidean = true) const {
        if (!appr_alg) throw std::runtime_error("Index not initialized");
        std::vector<dist_t> dists;
        appr_alg->getLastCandidateQueueDists(dists);
        for (size_t i = 0; i < dists.size(); i++) {
            dists[i] = maybe_euclid_(dists[i], euclidean);
        }
        py::array_t<dist_t> arr(dists.size());
        std::memcpy(arr.mutable_data(), dists.data(), dists.size() * sizeof(dist_t));
        return arr;
    }

    void set_store_candidate_queue_dists(bool enabled) {
        if (!appr_alg) throw std::runtime_error("Index not initialized");
        appr_alg->setCollectLastCandidateQueue(enabled);
    }

    void set_collect_traversal_counts(bool enabled) {
        if (!appr_alg) throw std::runtime_error("Index not initialized");
        appr_alg->setCollectTraversalCounts(enabled);
    }

    void reset_traversal_counts() {
        if (!appr_alg) throw std::runtime_error("Index not initialized");
        appr_alg->resetTraversalCounts();
    }

    py::dict get_traversal_counts() const {
        if (!appr_alg) throw std::runtime_error("Index not initialized");
        std::vector<uint64_t> node_counts;
        std::vector<hnswlib::tableint> edge_src_internal;
        std::vector<hnswlib::tableint> edge_dst_internal;
        std::vector<uint64_t> edge_counts;
        uint64_t query_count = 0;
        appr_alg->getTraversalCounts(
            node_counts,
            edge_src_internal,
            edge_dst_internal,
            edge_counts,
            query_count);

        py::array_t<uint64_t> node_counts_arr(node_counts.size());
        py::array_t<hnswlib::tableint> edge_src_arr(edge_src_internal.size());
        py::array_t<hnswlib::tableint> edge_dst_arr(edge_dst_internal.size());
        py::array_t<uint64_t> edge_counts_arr(edge_counts.size());
        if (!node_counts.empty()) {
            std::memcpy(node_counts_arr.mutable_data(), node_counts.data(), node_counts.size() * sizeof(uint64_t));
        }
        if (!edge_src_internal.empty()) {
            std::memcpy(edge_src_arr.mutable_data(), edge_src_internal.data(), edge_src_internal.size() * sizeof(hnswlib::tableint));
            std::memcpy(edge_dst_arr.mutable_data(), edge_dst_internal.data(), edge_dst_internal.size() * sizeof(hnswlib::tableint));
            std::memcpy(edge_counts_arr.mutable_data(), edge_counts.data(), edge_counts.size() * sizeof(uint64_t));
        }
        py::dict out;
        out["query_count"] = py::int_(query_count);
        out["node_counts"] = std::move(node_counts_arr);
        out["edge_src_internal"] = std::move(edge_src_arr);
        out["edge_dst_internal"] = std::move(edge_dst_arr);
        out["edge_counts"] = std::move(edge_counts_arr);
        return out;
    }

    py::dict get_traversal_counts_split() const {
        if (!appr_alg) throw std::runtime_error("Index not initialized");
        std::vector<uint64_t> upper_node_counts;
        std::vector<hnswlib::tableint> upper_edge_src_internal;
        std::vector<hnswlib::tableint> upper_edge_dst_internal;
        std::vector<uint64_t> upper_edge_counts;
        std::vector<uint64_t> l0_node_counts;
        std::vector<hnswlib::tableint> l0_edge_src_internal;
        std::vector<hnswlib::tableint> l0_edge_dst_internal;
        std::vector<uint64_t> l0_edge_counts;
        uint64_t query_count = 0;
        appr_alg->getTraversalCountsSplit(
            upper_node_counts,
            upper_edge_src_internal,
            upper_edge_dst_internal,
            upper_edge_counts,
            l0_node_counts,
            l0_edge_src_internal,
            l0_edge_dst_internal,
            l0_edge_counts,
            query_count);

        py::array_t<uint64_t> upper_node_counts_arr(upper_node_counts.size());
        py::array_t<hnswlib::tableint> upper_edge_src_arr(upper_edge_src_internal.size());
        py::array_t<hnswlib::tableint> upper_edge_dst_arr(upper_edge_dst_internal.size());
        py::array_t<uint64_t> upper_edge_counts_arr(upper_edge_counts.size());
        py::array_t<uint64_t> l0_node_counts_arr(l0_node_counts.size());
        py::array_t<hnswlib::tableint> l0_edge_src_arr(l0_edge_src_internal.size());
        py::array_t<hnswlib::tableint> l0_edge_dst_arr(l0_edge_dst_internal.size());
        py::array_t<uint64_t> l0_edge_counts_arr(l0_edge_counts.size());
        if (!upper_node_counts.empty()) {
            std::memcpy(upper_node_counts_arr.mutable_data(), upper_node_counts.data(), upper_node_counts.size() * sizeof(uint64_t));
        }
        if (!upper_edge_src_internal.empty()) {
            std::memcpy(upper_edge_src_arr.mutable_data(), upper_edge_src_internal.data(), upper_edge_src_internal.size() * sizeof(hnswlib::tableint));
            std::memcpy(upper_edge_dst_arr.mutable_data(), upper_edge_dst_internal.data(), upper_edge_dst_internal.size() * sizeof(hnswlib::tableint));
            std::memcpy(upper_edge_counts_arr.mutable_data(), upper_edge_counts.data(), upper_edge_counts.size() * sizeof(uint64_t));
        }
        if (!l0_node_counts.empty()) {
            std::memcpy(l0_node_counts_arr.mutable_data(), l0_node_counts.data(), l0_node_counts.size() * sizeof(uint64_t));
        }
        if (!l0_edge_src_internal.empty()) {
            std::memcpy(l0_edge_src_arr.mutable_data(), l0_edge_src_internal.data(), l0_edge_src_internal.size() * sizeof(hnswlib::tableint));
            std::memcpy(l0_edge_dst_arr.mutable_data(), l0_edge_dst_internal.data(), l0_edge_dst_internal.size() * sizeof(hnswlib::tableint));
            std::memcpy(l0_edge_counts_arr.mutable_data(), l0_edge_counts.data(), l0_edge_counts.size() * sizeof(uint64_t));
        }

        py::dict out;
        out["query_count"] = py::int_(query_count);
        out["upper_node_counts"] = std::move(upper_node_counts_arr);
        out["upper_edge_src_internal"] = std::move(upper_edge_src_arr);
        out["upper_edge_dst_internal"] = std::move(upper_edge_dst_arr);
        out["upper_edge_counts"] = std::move(upper_edge_counts_arr);
        out["l0_node_counts"] = std::move(l0_node_counts_arr);
        out["l0_edge_src_internal"] = std::move(l0_edge_src_arr);
        out["l0_edge_dst_internal"] = std::move(l0_edge_dst_arr);
        out["l0_edge_counts"] = std::move(l0_edge_counts_arr);
        return out;
    }

    long long get_traversal_query_count() const {
        if (!appr_alg) throw std::runtime_error("Index not initialized");
        return (long long)appr_alg->getTraversalQueryCount();
    }

    long long get_last_ef_explored() const {
        if (!appr_alg) throw std::runtime_error("Index not initialized");
        return appr_alg->getLastEfExplored();
    }

    long long get_last_best_found_expansion() const {
        if (!appr_alg) throw std::runtime_error("Index not initialized");
        return appr_alg->getLastBestFoundExpansion();
    }

    long long get_last_turn_back_count() const {
        if (!appr_alg) throw std::runtime_error("Index not initialized");
        return appr_alg->getLastTurnBackCount();
    }

    double get_last_turn_back_total_delta(bool euclidean = true) const {
        if (!appr_alg) throw std::runtime_error("Index not initialized");
        return maybe_euclid_(appr_alg->getLastTurnBackTotalDelta(), euclidean);
    }

    // Standalone MBV path (Algorithm 3 style): ANN -> MBV.
    // This is intentionally separate from the older certify/rectify stage pipelines.
    py::tuple mbv_query_ex(
        py::array_t<dist_t, py::array::c_style | py::array::forcecast> query,
        size_t k = 100,
        float t = 3.91f,
        bool euclidean = true,
        bool use_ann_cache = true,
        bool use_ellipse_prune = true,
        bool collect_node_trace = false,
        long long oracle_stop_pop = -1,
        size_t mbv_seed_count = 0,
        size_t mbv_seed_start = 0
#if HNSWLIB_MBV_CONFORMAL_EARLY_STOP
        , py::object conformal_stop_trees = py::none(),
        py::object conformal_stop_feature_names = py::none(),
        double conformal_stop_qhat = 0.0,
        long long conformal_stop_allowed_missing = -1,
        double conformal_stop_base_score = 0.0,
        py::object conformal_stop_static_features = py::none(),
        double conformal_stop_binary_threshold = -1.0,
        long long conformal_stop_min_pop = 0,
        py::object conformal_stop_isotonic_x = py::none(),
        py::object conformal_stop_isotonic_y = py::none()
#endif
    ) {
        if (!appr_alg) throw std::runtime_error("Index not initialized");
        if (query.ndim() != 1) throw std::runtime_error("query must be 1D");
        if ((size_t)query.shape(0) != (size_t)dim) throw std::runtime_error("query dim mismatch");
        if (k == 0) throw std::runtime_error("k must be > 0");

#if HNSWLIB_MBV_CONFORMAL_EARLY_STOP
        struct ConformalTreeNode {
            bool leaf = true;
            int feature = -1;
            double split = 0.0;
            double value = 0.0;
            int yes = -1;
            int no = -1;
            int missing = -1;
        };
        std::vector<std::string> conformal_feature_names;
        std::unordered_map<std::string, int> conformal_feature_index;
        if (!conformal_stop_feature_names.is_none()) {
            int idx = 0;
            for (py::handle h : py::reinterpret_borrow<py::iterable>(conformal_stop_feature_names)) {
                std::string name = py::cast<std::string>(h);
                conformal_feature_names.push_back(name);
                conformal_feature_index[name] = idx;
                conformal_feature_index["f" + std::to_string(idx)] = idx;
                idx++;
            }
        }
        auto conformal_feature_id = [&](const std::string& name) -> int {
            auto it = conformal_feature_index.find(name);
            if (it != conformal_feature_index.end()) return it->second;
            if (name.size() > 1 && name[0] == 'f') return std::atoi(name.c_str() + 1);
            return -1;
        };
        std::function<void(py::dict, std::vector<ConformalTreeNode>&)> parse_conformal_node;
        parse_conformal_node = [&](py::dict d, std::vector<ConformalTreeNode>& tree) {
            const int node_id = py::cast<int>(d["nodeid"]);
            if ((int)tree.size() <= node_id) tree.resize((size_t)node_id + 1);
            ConformalTreeNode node;
            if (d.contains("leaf")) {
                node.leaf = true;
                node.value = py::cast<double>(d["leaf"]);
            } else {
                node.leaf = false;
                node.feature = conformal_feature_id(py::cast<std::string>(d["split"]));
                node.split = py::cast<double>(d["split_condition"]);
                node.yes = py::cast<int>(d["yes"]);
                node.no = py::cast<int>(d["no"]);
                node.missing = py::cast<int>(d["missing"]);
                for (py::handle child : py::reinterpret_borrow<py::iterable>(d["children"])) {
                    parse_conformal_node(py::reinterpret_borrow<py::dict>(child), tree);
                }
            }
            tree[(size_t)node_id] = node;
        };
        std::vector<std::vector<ConformalTreeNode>> conformal_trees;
        if (!conformal_stop_trees.is_none()) {
            py::object json_loads = py::module_::import("json").attr("loads");
            for (py::handle root : py::reinterpret_borrow<py::iterable>(conformal_stop_trees)) {
                py::object root_obj = py::reinterpret_borrow<py::object>(root);
                if (py::isinstance<py::str>(root_obj)) {
                    root_obj = json_loads(root_obj);
                }
                std::vector<ConformalTreeNode> tree;
                parse_conformal_node(py::reinterpret_borrow<py::dict>(root_obj), tree);
                conformal_trees.push_back(std::move(tree));
            }
        }
        std::vector<double> conformal_static_features(conformal_feature_names.size(), 0.0);
        std::vector<uint8_t> conformal_static_feature_present(conformal_feature_names.size(), 0);
        if (!conformal_stop_static_features.is_none()) {
            py::dict sf = py::reinterpret_borrow<py::dict>(conformal_stop_static_features);
            for (const auto& item : sf) {
                const std::string name = py::cast<std::string>(item.first);
                auto it = conformal_feature_index.find(name);
                if (it != conformal_feature_index.end()) {
                    conformal_static_features[(size_t)it->second] = py::cast<double>(item.second);
                    conformal_static_feature_present[(size_t)it->second] = 1;
                }
            }
        }
        std::vector<double> conformal_iso_x;
        std::vector<double> conformal_iso_y;
        if (!conformal_stop_isotonic_x.is_none() && !conformal_stop_isotonic_y.is_none()) {
            for (py::handle h : py::reinterpret_borrow<py::iterable>(conformal_stop_isotonic_x)) {
                conformal_iso_x.push_back(py::cast<double>(h));
            }
            for (py::handle h : py::reinterpret_borrow<py::iterable>(conformal_stop_isotonic_y)) {
                conformal_iso_y.push_back(py::cast<double>(h));
            }
            if (conformal_iso_x.size() != conformal_iso_y.size()) {
                conformal_iso_x.clear();
                conformal_iso_y.clear();
            }
        }
#endif

        const dist_t* qptr = query.data();
        const bool apply_l2_sqrt = euclidean && space_name == "l2";

        std::vector<std::pair<hnswlib::labeltype, dist_t>> out;
        out.reserve(k);

        long long ann_time_ns = 0;
        long long mbv_time_ns = 0;
        long long total_time_ns = 0;
        long long expanded_nodes = 0;
        long long ann_cache_hits = 0;
        long long exact_dist_computes = 0;
        long long edge_dist_computes = 0;
        long long dijkstra_loop_time_ns = 0;
        long long pq_pop_time_ns = 0;
        long long pq_push_time_ns = 0;
        long long exact_dist_time_ns = 0;
        long long neighbor_loop_time_ns = 0;
        const bool fine_timers = mbv_fine_timers_enabled_;
        const bool region_timers = mbv_region_timers_enabled_;
        long long pq_pop_count = 0;
        long long pq_push_count = 0;
        long long pq_size_before_push_sum = 0;
        long long neighbor_expanded_count = 0;
        long long edge_neighbor_visits = 0;
        long long lb_pruned = 0;
        long long ellipse_pruned = 0;
        long long r_prune_shrink_count = 0;
        long long settled_skips = 0;
        long long deleted_skips = 0;
        dist_t expanded_graph_dist_sum = (dist_t)0;
        dist_t expanded_graph_dist_max = (dist_t)0;
        dist_t computed_l2_dist_sum = (dist_t)0;
        dist_t computed_l2_dist_max = (dist_t)0;
        long long region_setup_seed_ns = 0;
        long long region_pop_ns = 0;
        long long region_lb_prune_ns = 0;
        long long region_exact_eval_ns = 0;
        long long region_topk_update_ns = 0;
        long long region_ellipse_check_ns = 0;
        long long region_neighbor_expand_ns = 0;
        long long region_edge_weight_lookup_ns = 0;
        long long region_neighbor_for_loop_ns = 0;
        long long region_neighbor_for_loop_count = 0;
        long long neighbor_iter_count = 0;
#if HNSWLIB_MBV_CONFORMAL_EARLY_STOP
        const bool conformal_stop_enabled = !conformal_trees.empty() && conformal_stop_allowed_missing >= 0;
        bool conformal_stopped = false;
        long long conformal_stop_pop = -1;
        long long conformal_stop_update_idx = -1;
        long long conformal_stop_calls = 0;
        long long conformal_stop_time_ns = 0;
        long long conformal_feature_time_ns = 0;
        long long conformal_inference_time_ns = 0;
        double conformal_pred_missing = (double)k;
        double conformal_stop_prob = 0.0;
        bool conformal_have_prev_update = false;
        dist_t conformal_prev_dk = (dist_t)0;
        long long conformal_prev_queue_size = 0;
#endif
        long long neighbor_iter_settled_skip_count = 0;
        long long neighbor_iter_rprune_skip_count = 0;
        long long neighbor_iter_no_improve_skip_count = 0;
        long long neighbor_iter_relax_count = 0;
        long long neighbor_iter_settled_check_ns = 0;
        long long neighbor_iter_weight_nd_ns = 0;
        long long neighbor_iter_guard_ns = 0;
        long long neighbor_iter_relax_write_ns = 0;
        long long neighbor_iter_heap_push_ns = 0;
        long long neighbor_iter_trace_ns = 0;
        bool oracle_stopped = false;
        dist_t dk_final = (dist_t)0;
        dist_t r_prune_final = (dist_t)0;
        size_t k_effective = 0;
        std::vector<hnswlib::tableint> lb_pruned_nodes;
        std::vector<hnswlib::tableint> ellipse_pruned_nodes;
        std::vector<hnswlib::tableint> popped_nodes;
        std::vector<hnswlib::tableint> exact_dist_nodes;
        std::vector<hnswlib::tableint> skipped_exact_nodes;
        std::vector<hnswlib::tableint> neighbor_expanded_nodes;
        std::vector<long long> neighbor_expanded_degrees;
        std::vector<long long> dk_update_pop_counts;
        std::vector<hnswlib::tableint> dk_update_nodes;
        std::vector<dist_t> dk_update_old_values;
        std::vector<dist_t> dk_update_new_values;
        std::vector<dist_t> dk_update_node_dists;
        std::vector<long long> topk_update_pop_counts;
        std::vector<hnswlib::tableint> topk_update_nodes;
        std::vector<long long> topk_update_push_counts;
        std::vector<long long> topk_update_expanded_nodes;
        std::vector<long long> topk_update_exact_dist_computes;
        std::vector<long long> topk_update_edge_visits;
        std::vector<long long> topk_update_lb_pruned;
        std::vector<long long> topk_update_ellipse_pruned;
        std::vector<long long> topk_update_ann_cache_hits;
        std::vector<long long> topk_update_queue_sizes;
        std::vector<long long> topk_update_pops_since_prev;
        std::vector<dist_t> topk_update_queue_min_dists;
        std::vector<dist_t> topk_update_du_graph;
        std::vector<dist_t> topk_update_duq;
        std::vector<dist_t> topk_update_old_dk;
        std::vector<dist_t> topk_update_new_dk;
        std::vector<dist_t> topk_update_r_prune;
        std::vector<dist_t> topk_update_d1;
        std::vector<dist_t> topk_update_mean_dist;
        std::vector<dist_t> topk_update_median_dist;
        std::vector<dist_t> topk_update_std_dist;
        std::vector<dist_t> topk_update_q25_dist;
        std::vector<dist_t> topk_update_q75_dist;
        std::vector<dist_t> topk_update_gap_before_k;
        std::vector<dist_t> topk_update_gap_before_k5;
        std::vector<dist_t> topk_update_gap_before_k10;
        std::vector<long long> topk_update_within_1pct_dk;
        std::vector<long long> topk_update_within_5pct_dk;
        std::vector<long long> topk_update_within_10pct_dk;
        std::vector<dist_t> topk_update_expanded_graph_dist_mean;
        std::vector<dist_t> topk_update_expanded_graph_dist_max;
        std::vector<dist_t> topk_update_computed_l2_dist_mean;
        std::vector<dist_t> topk_update_computed_l2_dist_max;
        std::vector<std::vector<hnswlib::labeltype>> topk_update_labels;
        std::vector<std::vector<dist_t>> topk_update_dists;
        std::vector<hnswlib::tableint> neighbor_push_parent_nodes;
        std::vector<hnswlib::tableint> neighbor_push_child_nodes;

        struct MBVWorkspace {
            std::vector<dist_t> G, state_val, parent_w;
            std::vector<hnswlib::tableint> parent;
            std::vector<uint32_t> epoch_seen, epoch_known, epoch_exact, epoch_settled;
            std::vector<hnswlib::tableint> touched;
            uint32_t cur_epoch = 1;

            void ensure_size(size_t N) {
                if (G.size() >= N) return;
                G.resize(N);
                state_val.resize(N);
                parent_w.resize(N);
                parent.resize(N);
                epoch_seen.resize(N, 0);
                epoch_known.resize(N, 0);
                epoch_exact.resize(N, 0);
                epoch_settled.resize(N, 0);
            }

            void begin_query(size_t N) {
                ensure_size(N);
                cur_epoch++;
                if (cur_epoch == 0) {
                    std::fill(epoch_seen.begin(), epoch_seen.end(), 0);
                    std::fill(epoch_known.begin(), epoch_known.end(), 0);
                    std::fill(epoch_exact.begin(), epoch_exact.end(), 0);
                    std::fill(epoch_settled.begin(), epoch_settled.end(), 0);
                    cur_epoch = 1;
                }
                touched.clear();
            }

            bool seen(hnswlib::tableint u) const {
                return epoch_seen[(size_t)u] == cur_epoch;
            }

            dist_t graph_dist(hnswlib::tableint u, dist_t inf) const {
                return seen(u) ? G[(size_t)u] : inf;
            }

            void set_graph(hnswlib::tableint u, dist_t g, hnswlib::tableint p, dist_t w) {
                const size_t idx = (size_t)u;
                if (epoch_seen[idx] != cur_epoch) {
                    epoch_seen[idx] = cur_epoch;
                    touched.push_back(u);
                }
                G[idx] = g;
                parent[idx] = p;
                parent_w[idx] = w;
            }

            bool known(hnswlib::tableint u) const {
                return epoch_known[(size_t)u] == cur_epoch;
            }

            bool exact(hnswlib::tableint u) const {
                return epoch_exact[(size_t)u] == cur_epoch;
            }

            void set_exact(hnswlib::tableint u, dist_t v) {
                const size_t idx = (size_t)u;
                epoch_known[idx] = cur_epoch;
                epoch_exact[idx] = cur_epoch;
                state_val[idx] = v;
            }

            void set_bound(hnswlib::tableint u, dist_t v) {
                const size_t idx = (size_t)u;
                epoch_known[idx] = cur_epoch;
                epoch_exact[idx] = 0;
                state_val[idx] = v;
            }

            bool settled(hnswlib::tableint u) const {
                return epoch_settled[(size_t)u] == cur_epoch;
            }

            void mark_settled(hnswlib::tableint u) {
                epoch_settled[(size_t)u] = cur_epoch;
            }
        };

        {
            py::gil_scoped_release release;
            auto t_all0 = std::chrono::steady_clock::now();

            const size_t N = (size_t)appr_alg->cur_element_count.load();
            const bool direct_euclidean_l0_cache = can_use_direct_euclidean_l0_edge_cache(N);
            const dist_t INF = std::numeric_limits<dist_t>::infinity();
            thread_local MBVWorkspace mbv_ws;
            mbv_ws.begin_query(N);
            const bool has_deletions = (appr_alg->num_deleted_ > 0);
            if (!apply_l2_sqrt || !direct_euclidean_l0_cache || has_deletions) {
                throw std::runtime_error("mbv_query_ex hot-loop build assumes euclidean L2, direct euclidean level-0 edge cache, and no deletions");
            }
            lb_pruned_nodes.reserve(std::min<size_t>(N, k * (size_t)32));
            ellipse_pruned_nodes.reserve(std::min<size_t>(N, k * (size_t)1024));
            if (collect_node_trace) {
                popped_nodes.reserve(std::min<size_t>(N, k * (size_t)4096));
                exact_dist_nodes.reserve(std::min<size_t>(N, k * (size_t)1024));
                skipped_exact_nodes.reserve(std::min<size_t>(N, k * (size_t)1024));
                neighbor_expanded_nodes.reserve(std::min<size_t>(N, k * (size_t)4096));
                neighbor_expanded_degrees.reserve(std::min<size_t>(N, k * (size_t)4096));
                dk_update_pop_counts.reserve(k * (size_t)16);
                dk_update_nodes.reserve(k * (size_t)16);
                dk_update_old_values.reserve(k * (size_t)16);
                dk_update_new_values.reserve(k * (size_t)16);
                dk_update_node_dists.reserve(k * (size_t)16);
                topk_update_pop_counts.reserve(k * (size_t)16);
                topk_update_nodes.reserve(k * (size_t)16);
                topk_update_push_counts.reserve(k * (size_t)16);
                topk_update_expanded_nodes.reserve(k * (size_t)16);
                topk_update_exact_dist_computes.reserve(k * (size_t)16);
                topk_update_edge_visits.reserve(k * (size_t)16);
                topk_update_lb_pruned.reserve(k * (size_t)16);
                topk_update_ellipse_pruned.reserve(k * (size_t)16);
                topk_update_ann_cache_hits.reserve(k * (size_t)16);
                topk_update_queue_sizes.reserve(k * (size_t)16);
                topk_update_pops_since_prev.reserve(k * (size_t)16);
                topk_update_queue_min_dists.reserve(k * (size_t)16);
                topk_update_du_graph.reserve(k * (size_t)16);
                topk_update_duq.reserve(k * (size_t)16);
                topk_update_old_dk.reserve(k * (size_t)16);
                topk_update_new_dk.reserve(k * (size_t)16);
                topk_update_r_prune.reserve(k * (size_t)16);
                topk_update_d1.reserve(k * (size_t)16);
                topk_update_mean_dist.reserve(k * (size_t)16);
                topk_update_median_dist.reserve(k * (size_t)16);
                topk_update_std_dist.reserve(k * (size_t)16);
                topk_update_q25_dist.reserve(k * (size_t)16);
                topk_update_q75_dist.reserve(k * (size_t)16);
                topk_update_gap_before_k.reserve(k * (size_t)16);
                topk_update_gap_before_k5.reserve(k * (size_t)16);
                topk_update_gap_before_k10.reserve(k * (size_t)16);
                topk_update_within_1pct_dk.reserve(k * (size_t)16);
                topk_update_within_5pct_dk.reserve(k * (size_t)16);
                topk_update_within_10pct_dk.reserve(k * (size_t)16);
                topk_update_expanded_graph_dist_mean.reserve(k * (size_t)16);
                topk_update_expanded_graph_dist_max.reserve(k * (size_t)16);
                topk_update_computed_l2_dist_mean.reserve(k * (size_t)16);
                topk_update_computed_l2_dist_max.reserve(k * (size_t)16);
                topk_update_labels.reserve(k * (size_t)16);
                topk_update_dists.reserve(k * (size_t)16);
                neighbor_push_parent_nodes.reserve(std::min<size_t>(N, k * (size_t)4096));
                neighbor_push_child_nodes.reserve(std::min<size_t>(N, k * (size_t)4096));
            }

            // ---- Stage A: ANN ----
            auto t_ann0 = std::chrono::steady_clock::now();
#if HNSWLIB_MBV_CONFORMAL_EARLY_STOP
            if (conformal_stop_enabled) {
                // Match the Python feature collector's ANN side-channel features.
                appr_alg->setCollectLastCandidateQueue(true);
            }
#endif
            auto top = appr_alg->searchKnn((void*)qptr, k, nullptr);
            auto t_ann1 = std::chrono::steady_clock::now();
            ann_time_ns = (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(t_ann1 - t_ann0).count();
#if HNSWLIB_MBV_CONFORMAL_EARLY_STOP
            double conformal_crc_unchecked_count = 0.0;
            double conformal_crc_ann_reached = 0.0;
            double conformal_crc_turn_back_count = 0.0;
            double conformal_crc_turn_back_total_delta = 0.0;
            double conformal_crc_unchecked_min = 0.0;
            double conformal_crc_unchecked_q25 = 0.0;
            double conformal_crc_unchecked_median = 0.0;
            double conformal_crc_unchecked_q75 = 0.0;
            double conformal_crc_unchecked_max = 0.0;
            double conformal_crc_unchecked_mean = 0.0;
            double conformal_crc_unchecked_std = 0.0;
            if (conformal_stop_enabled) {
                std::vector<dist_t> last_candidate_queue;
                std::vector<hnswlib::tableint> last_ann_computed_ids;
                std::vector<dist_t> last_ann_computed_dists;
                appr_alg->getLastCandidateQueueDists(last_candidate_queue);
                appr_alg->getLastQueryComputedDists(last_ann_computed_ids, last_ann_computed_dists);
                conformal_crc_unchecked_count = (double)last_candidate_queue.size();
                conformal_crc_ann_reached = (double)last_ann_computed_ids.size();
                conformal_crc_turn_back_count = (double)appr_alg->getLastTurnBackCount();
                conformal_crc_turn_back_total_delta = (double)appr_alg->getLastTurnBackTotalDelta();
                if (!last_candidate_queue.empty()) {
                    std::sort(last_candidate_queue.begin(), last_candidate_queue.end());
                    const size_t m = last_candidate_queue.size();
                    auto qv = [&](double q) -> double {
                        const double pos = q * (double)(m - 1);
                        const size_t lo = (size_t)std::floor(pos);
                        const size_t hi = std::min(m - 1, lo + (size_t)1);
                        const double frac = pos - (double)lo;
                        return (double)last_candidate_queue[lo] * (1.0 - frac) + (double)last_candidate_queue[hi] * frac;
                    };
                    double sum = 0.0;
                    double sum_sq = 0.0;
                    for (dist_t x : last_candidate_queue) {
                        const double xd = (double)x;
                        sum += xd;
                        sum_sq += xd * xd;
                    }
                    conformal_crc_unchecked_min = (double)last_candidate_queue.front();
                    conformal_crc_unchecked_q25 = qv(0.25);
                    conformal_crc_unchecked_median = qv(0.50);
                    conformal_crc_unchecked_q75 = qv(0.75);
                    conformal_crc_unchecked_max = (double)last_candidate_queue.back();
                    conformal_crc_unchecked_mean = sum / (double)m;
                    const double var = std::max(0.0, sum_sq / (double)m - conformal_crc_unchecked_mean * conformal_crc_unchecked_mean);
                    conformal_crc_unchecked_std = std::sqrt(var);
                }
            }
#endif

            k_effective = top.size();
            if (k_effective == 0) throw std::runtime_error("ANN returned empty candidate set");
            std::vector<hnswlib::labeltype> ann_labels(k_effective);
            std::vector<dist_t> ann_dists(k_effective);
            std::unordered_set<hnswlib::labeltype> ann_label_set;
            ann_label_set.reserve(k_effective * 2 + 16);
            for (size_t i = k_effective; i-- > 0;) {
                auto p = top.top();
                top.pop();
                ann_labels[i] = p.second;
                ann_label_set.insert(p.second);
                dist_t d = p.first;
                if (apply_l2_sqrt) {
                    if (d < (dist_t)0) d = (dist_t)0;
                    d = (dist_t)std::sqrt((double)d);
                }
                ann_dists[i] = d;
            }

            // label -> internal id
            std::vector<hnswlib::tableint> ann_internal(k_effective, (hnswlib::tableint)-1);
            {
                std::unique_lock<std::mutex> lock_table(appr_alg->label_lookup_lock);
                for (size_t i = 0; i < k_effective; i++) {
                    auto it = appr_alg->label_lookup_.find(ann_labels[i]);
                    if (it != appr_alg->label_lookup_.end() && !appr_alg->isMarkedDeleted(it->second)) {
                        ann_internal[i] = it->second;
                    }
                }
            }

            // Query-local MBV state is stored in a reusable epoch workspace.
            // This avoids clearing several full-N arrays on every query.
            MBVWorkspace& ws = mbv_ws;

            // Optional cache from ANN internal beam computations.
            if (use_ann_cache) {
                std::vector<hnswlib::tableint> ids_internal;
                std::vector<dist_t> dists_sq;
                appr_alg->getLastQueryComputedDists(ids_internal, dists_sq);
                const size_t m = std::min(ids_internal.size(), dists_sq.size());
                for (size_t i = 0; i < m; i++) {
                    const hnswlib::tableint u = ids_internal[i];
                    if ((size_t)u >= N) continue;
                    dist_t duq = dists_sq[i];
                    if (apply_l2_sqrt) {
                        if (duq < (dist_t)0) duq = (dist_t)0;
                        duq = (dist_t)std::sqrt((double)duq);
                    }
                    if (!ws.known(u)) {
                        ws.set_exact(u, duq);
                    } else if (duq < ws.state_val[(size_t)u]) {
                        ws.set_exact(u, duq);
                    }
                }
            }

            // Result set seeded by ANN H_found (unique by label).
            using DistLabel = std::pair<dist_t, hnswlib::labeltype>;
            struct DistLabelCmp {
                bool operator()(const DistLabel& a, const DistLabel& b) const {
                    if (a.first < b.first) return true;
                    if (a.first > b.first) return false;
                    return a.second < b.second;
                }
            };
            std::set<DistLabel, DistLabelCmp> topk_set;
            std::unordered_map<hnswlib::labeltype, typename std::set<DistLabel, DistLabelCmp>::iterator> topk_pos;
            topk_pos.reserve(k_effective * 2 + 16);

            auto insert_or_update_topk = [&](hnswlib::labeltype lab, dist_t d) -> bool {
                auto pit = topk_pos.find(lab);
                if (pit != topk_pos.end()) {
                    auto sit = pit->second;
                    if (d >= sit->first) return false;
                    topk_set.erase(sit);
                    auto nit = topk_set.insert({d, lab}).first;
                    pit->second = nit;
                    return true;
                }
                if (topk_set.size() < k_effective) {
                    auto nit = topk_set.insert({d, lab}).first;
                    topk_pos[lab] = nit;
                    return true;
                }
                auto worst_it = std::prev(topk_set.end());
                if (d >= worst_it->first) return false;
                topk_pos.erase(worst_it->second);
                topk_set.erase(worst_it);
                auto nit = topk_set.insert({d, lab}).first;
                topk_pos[lab] = nit;
                return true;
            };

            for (size_t i = 0; i < k_effective; i++) {
                insert_or_update_topk(ann_labels[i], ann_dists[i]);
                const hnswlib::tableint s = ann_internal[i];
                if ((size_t)s < N) {
                    ws.set_exact(s, ann_dists[i]);
                }
            }
            if (topk_set.empty()) throw std::runtime_error("MBV top-k seed is empty");
            dist_t dk = std::prev(topk_set.end())->first;
            dist_t r_prune = (dist_t)t * dk;
            long long last_topk_update_pop_count = 0;
            long long topk_update_count = 0;

            if (oracle_stop_pop == 0) {
                auto t_mbv0 = std::chrono::steady_clock::now();
                auto t_mbv1 = std::chrono::steady_clock::now();
                oracle_stopped = true;
                mbv_time_ns = (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(t_mbv1 - t_mbv0).count();
                total_time_ns = (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(t_mbv1 - t_all0).count();
                dk_final = dk;
                r_prune_final = r_prune;
                out.clear();
                out.reserve(topk_set.size());
                for (const auto& p : topk_set) out.push_back({p.second, p.first});
            } else {
            // ---- Stage B: MBV (Dijkstra + bound verification + exact eval) ----
            auto t_mbv0 = std::chrono::steady_clock::now();

            struct QItem { dist_t d; hnswlib::tableint v; };
            struct QCmp { bool operator()(const QItem& a, const QItem& b) const { return a.d > b.d; } };
            std::vector<QItem> pq_storage;
            pq_storage.reserve(65536);
            std::priority_queue<QItem, std::vector<QItem>, QCmp> pq(QCmp{}, std::move(pq_storage));

            // Virtual-query seeding via ANN results.
            const size_t seed_begin = (mbv_seed_count == 0) ? 0 : std::min(mbv_seed_start, k_effective);
            const size_t seed_end = (mbv_seed_count == 0)
                ? k_effective
                : std::min(k_effective, seed_begin + mbv_seed_count);
            for (size_t i = seed_begin; i < seed_end; i++) {
                const hnswlib::tableint s = ann_internal[i];
                if ((size_t)s >= N) continue;
                const dist_t ds = ann_dists[i];
                if (ds < ws.graph_dist(s, INF)) {
                    ws.set_graph(s, ds, (hnswlib::tableint)-1, (dist_t)0);
                    pq_size_before_push_sum += (long long)pq.size();
                    pq.push(QItem{ds, s});
                }
            }

            auto t_setup1 = std::chrono::steady_clock::now();
            if (region_timers) {
                region_setup_seed_ns = (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(t_setup1 - t_mbv0).count();
            }
            auto t_loop0 = std::chrono::steady_clock::now();
            while (!pq.empty()) {
                std::chrono::steady_clock::time_point t_region_pop0;
                if (region_timers) t_region_pop0 = std::chrono::steady_clock::now();
                std::chrono::steady_clock::time_point t_pop0;
                if (fine_timers) t_pop0 = std::chrono::steady_clock::now();
                auto cur = pq.top();
                pq.pop();
                if (region_timers) {
                    auto t_region_pop1 = std::chrono::steady_clock::now();
                    region_pop_ns += (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(t_region_pop1 - t_region_pop0).count();
                }
                if (fine_timers) {
                    auto t_pop1 = std::chrono::steady_clock::now();
                    pq_pop_time_ns += (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(t_pop1 - t_pop0).count();
                }
                pq_pop_count++;
                const dist_t du = cur.d;
                const hnswlib::tableint u = cur.v;
                const bool oracle_stop_after_pop = (oracle_stop_pop > 0 && pq_pop_count >= oracle_stop_pop);
                if (collect_node_trace && u >= 0) popped_nodes.push_back(u);

                if (du > r_prune) break;
                if ((size_t)u >= N) {
                    if (oracle_stop_after_pop) { oracle_stopped = true; break; }
                    continue;
                }
                if (ws.settled(u)) {
                    settled_skips++;
                    if (oracle_stop_after_pop) { oracle_stopped = true; break; }
                    continue;
                }
                if (has_deletions && appr_alg->isMarkedDeleted(u)) {
                    deleted_skips++;
                    if (oracle_stop_after_pop) { oracle_stopped = true; break; }
                    continue;
                }
                ws.mark_settled(u);
                expanded_nodes++;
                expanded_graph_dist_sum += du;
                if (expanded_nodes == 1 || du > expanded_graph_dist_max) expanded_graph_dist_max = du;

                std::chrono::steady_clock::time_point t_region_lb0;
                if (region_timers) t_region_lb0 = std::chrono::steady_clock::now();
                // MBV lower bound propagation from parent.
                dist_t lb_u = -INF;
                const hnswlib::tableint p = ws.parent[(size_t)u];
                if (p >= 0 && (size_t)p < N && ws.known(p)) {
                    lb_u = ws.state_val[(size_t)p] - ws.parent_w[(size_t)u];
                }

                // Prune by lower bound (skip expensive metric distance).
                if (lb_u > dk) {
                    ws.set_bound(u, lb_u);
                    lb_pruned++;
                    lb_pruned_nodes.push_back(u);
                    if (collect_node_trace) skipped_exact_nodes.push_back(u);
                    if (region_timers) {
                        auto t_region_lb1 = std::chrono::steady_clock::now();
                        region_lb_prune_ns += (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(t_region_lb1 - t_region_lb0).count();
                    }
                } else {
                    if (region_timers) {
                        auto t_region_lb1 = std::chrono::steady_clock::now();
                        region_lb_prune_ns += (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(t_region_lb1 - t_region_lb0).count();
                    }
                    // Evaluate d(q,u) if not already exact in state cache.
                    dist_t duq = (dist_t)0;
                    if (ws.known(u) && ws.exact(u)) {
                        duq = ws.state_val[(size_t)u];
                        ann_cache_hits++;
                    } else {
                        std::chrono::steady_clock::time_point t_region_exact0;
                        if (region_timers) t_region_exact0 = std::chrono::steady_clock::now();
                        std::chrono::steady_clock::time_point t_exact0;
                        if (fine_timers) t_exact0 = std::chrono::steady_clock::now();
                        char* udata_q = appr_alg->getDataByInternalId(u);
                        dist_t duq_sq = appr_alg->fstdistfunc_((void*)qptr, udata_q, appr_alg->dist_func_param_);
                        duq = duq_sq;
                        if (apply_l2_sqrt) {
                            if (duq < (dist_t)0) duq = (dist_t)0;
                            duq = (dist_t)std::sqrt((double)duq);
                        }
                        if (fine_timers) {
                            auto t_exact1 = std::chrono::steady_clock::now();
                            exact_dist_time_ns += (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(t_exact1 - t_exact0).count();
                        }
                        ws.set_exact(u, duq);
                        exact_dist_computes++;
                        computed_l2_dist_sum += duq;
                        if (exact_dist_computes == 1 || duq > computed_l2_dist_max) computed_l2_dist_max = duq;
                        if (collect_node_trace) exact_dist_nodes.push_back(u);
                        if (region_timers) {
                            auto t_region_exact1 = std::chrono::steady_clock::now();
                            region_exact_eval_ns += (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(t_region_exact1 - t_region_exact0).count();
                        }
                    }

                    // Avoid set/hash churn when duq cannot improve a full top-k.
                    std::chrono::steady_clock::time_point t_region_topk0;
                    if (region_timers) t_region_topk0 = std::chrono::steady_clock::now();
                    if (topk_set.size() < k_effective || duq < dk) {
                        if (insert_or_update_topk(appr_alg->getExternalLabel(u), duq)) {
                            const dist_t old_r_prune = r_prune;
                            const dist_t old_dk = dk;
                            dk = std::prev(topk_set.end())->first;
                            r_prune = (dist_t)t * dk;
                            if (collect_node_trace
#if HNSWLIB_MBV_CONFORMAL_EARLY_STOP
                                || conformal_stop_enabled
#endif
                            ) {
                                dist_t topk_sum = (dist_t)0;
                                dist_t topk_sum_sq = (dist_t)0;
                                std::vector<dist_t> current_dists;
                                std::vector<hnswlib::labeltype> current_labels;
                                current_dists.reserve(topk_set.size());
                                current_labels.reserve(topk_set.size());
                                for (const auto& tp : topk_set) {
                                    topk_sum += tp.first;
                                    topk_sum_sq += tp.first * tp.first;
                                    current_dists.push_back(tp.first);
                                    current_labels.push_back(tp.second);
                                }
                                const size_t cur_topk_n = current_dists.size();
                                const dist_t cur_d1 = cur_topk_n ? current_dists.front() : (dist_t)0;
                                const dist_t cur_median = cur_topk_n ? current_dists[cur_topk_n / 2] : (dist_t)0;
                                const dist_t cur_q25 = cur_topk_n ? current_dists[cur_topk_n / 4] : (dist_t)0;
                                const dist_t cur_q75 = cur_topk_n ? current_dists[(cur_topk_n * (size_t)3) / (size_t)4] : (dist_t)0;
                                const dist_t cur_iqr = cur_q75 - cur_q25;
                                const dist_t cur_gap_before_k = cur_topk_n >= 2 ? current_dists[cur_topk_n - 1] - current_dists[cur_topk_n - 2] : (dist_t)0;
                                const dist_t cur_gap_before_k5 = cur_topk_n >= 6 ? current_dists[cur_topk_n - 1] - current_dists[cur_topk_n - 6] : (dist_t)0;
                                const dist_t cur_gap_before_k10 = cur_topk_n >= 11 ? current_dists[cur_topk_n - 1] - current_dists[cur_topk_n - 11] : (dist_t)0;
                                const dist_t cur_mean = cur_topk_n ? topk_sum / (dist_t)cur_topk_n : (dist_t)0;
                                dist_t cur_std = (dist_t)0;
                                if (cur_topk_n) {
                                    const dist_t cur_var = std::max((dist_t)0, topk_sum_sq / (dist_t)cur_topk_n - cur_mean * cur_mean);
                                    cur_std = (dist_t)std::sqrt((double)cur_var);
                                }
                                long long within_1pct = 0;
                                long long within_5pct = 0;
                                long long within_10pct = 0;
                                const dist_t one_pct = dk * (dist_t)0.01;
                                const dist_t five_pct = dk * (dist_t)0.05;
                                const dist_t ten_pct = dk * (dist_t)0.10;
                                for (dist_t td : current_dists) {
                                    const dist_t tail_gap = dk - td;
                                    if (tail_gap <= one_pct) within_1pct++;
                                    if (tail_gap <= five_pct) within_5pct++;
                                    if (tail_gap <= ten_pct) within_10pct++;
                                }
                                const dist_t queue_min = pq.empty() ? INF : pq.top().d;
                                const dist_t expanded_graph_mean = expanded_nodes > 0 ? expanded_graph_dist_sum / (dist_t)expanded_nodes : (dist_t)0;
                                const dist_t computed_l2_mean = exact_dist_computes > 0 ? computed_l2_dist_sum / (dist_t)exact_dist_computes : (dist_t)0;
                                long long ann_overlap_count = 0;
                                for (hnswlib::labeltype lab : current_labels) {
                                    if (ann_label_set.find(lab) != ann_label_set.end()) ann_overlap_count++;
                                }
#if HNSWLIB_MBV_CONFORMAL_EARLY_STOP
                                if (conformal_stop_enabled) {
                                    // Native conformal stopper: evaluate the tree ensemble in C++ at each top-k update.
                                    auto t_stop0 = std::chrono::steady_clock::now();
                                    conformal_stop_calls++;
                                    std::vector<double> features = conformal_static_features;
                                    auto safe_div_local = [](double a, double b) -> double {
                                        return b != 0.0 ? a / b : 0.0;
                                    };
                                    auto ann_quantile = [&](double q) -> double {
                                        if (ann_dists.empty()) return 0.0;
                                        const double pos = q * (double)(ann_dists.size() - 1);
                                        const size_t lo = (size_t)std::floor(pos);
                                        const size_t hi = std::min(ann_dists.size() - 1, lo + (size_t)1);
                                        const double frac = pos - (double)lo;
                                        return (double)ann_dists[lo] * (1.0 - frac) + (double)ann_dists[hi] * frac;
                                    };
                                    auto isotonic_transform = [&](double p) -> double {
                                        if (conformal_iso_x.empty()) return p;
                                        if (p <= conformal_iso_x.front()) return conformal_iso_y.front();
                                        if (p >= conformal_iso_x.back()) return conformal_iso_y.back();
                                        for (size_t ii = 1; ii < conformal_iso_x.size(); ii++) {
                                            if (p <= conformal_iso_x[ii]) {
                                                const double x0 = conformal_iso_x[ii - 1];
                                                const double x1 = conformal_iso_x[ii];
                                                const double y0 = conformal_iso_y[ii - 1];
                                                const double y1 = conformal_iso_y[ii];
                                                const double frac = x1 != x0 ? (p - x0) / (x1 - x0) : 0.0;
                                                return y0 + frac * (y1 - y0);
                                            }
                                        }
                                        return p;
                                    };
                                    auto set_feature = [&](const std::string& name, double value, bool preserve_static = false) {
                                        auto it = conformal_feature_index.find(name);
                                        if (it != conformal_feature_index.end()) {
                                            const size_t idx = (size_t)it->second;
                                            if (!preserve_static || !conformal_static_feature_present[idx]) {
                                                features[idx] = value;
                                            }
                                        }
                                    };
                                    set_feature("computed_l2_mean_over_dk", safe_div_local((double)computed_l2_mean, (double)dk));
                                    set_feature("expanded_per_pop", safe_div_local((double)expanded_nodes, (double)pq_pop_count));
                                    set_feature("crc_unchecked_count", conformal_crc_unchecked_count, true);
                                    set_feature("crc_turn_back_count", conformal_crc_turn_back_count, true);
                                    set_feature("crc_ann_reached", conformal_crc_ann_reached, true);
                                    set_feature("ann_dist_q10", ann_quantile(0.10), true);
                                    set_feature("crc_turn_back_total_delta", conformal_crc_turn_back_total_delta, true);
                                    set_feature("crc_unchecked_min", conformal_crc_unchecked_min, true);
                                    set_feature("crc_unchecked_q25", conformal_crc_unchecked_q25, true);
                                    set_feature("crc_unchecked_median", conformal_crc_unchecked_median, true);
                                    set_feature("crc_unchecked_q75", conformal_crc_unchecked_q75, true);
                                    set_feature("crc_unchecked_max", conformal_crc_unchecked_max, true);
                                    set_feature("crc_unchecked_mean", conformal_crc_unchecked_mean, true);
                                    set_feature("crc_unchecked_std", conformal_crc_unchecked_std, true);
                                    set_feature("queue_size", (double)pq.size());
                                    set_feature("topk_within_5pct_dk", (double)within_5pct);
                                    set_feature("pop_count", (double)pq_pop_count);
                                    set_feature("push_per_pop", safe_div_local((double)pq_push_count, (double)pq_pop_count));
                                    set_feature("edges_per_expanded", safe_div_local((double)edge_neighbor_visits, (double)expanded_nodes));
                                    set_feature("ann_overlap_frac", safe_div_local((double)ann_overlap_count, (double)k));
                                    set_feature("topk_iqr", (double)cur_iqr);
                                    set_feature("queue_size_over_pop", safe_div_local((double)pq.size(), (double)pq_pop_count));
                                    set_feature("queue_min_over_dk", safe_div_local((double)queue_min, (double)dk));
                                    set_feature("dk_over_d1", safe_div_local((double)dk, (double)cur_d1));
                                    set_feature("topk_gap_before_k", (double)cur_gap_before_k);
                                    set_feature("topk_gap_before_k5", (double)cur_gap_before_k5);
                                    set_feature("topk_gap_before_k10", (double)cur_gap_before_k10);
                                    set_feature("within_5pct_frac", safe_div_local((double)within_5pct, (double)k));
                                    set_feature("within_10pct_frac", safe_div_local((double)within_10pct, (double)k));
                                    set_feature("dk_delta_prev", conformal_have_prev_update ? (double)(dk - conformal_prev_dk) : 0.0);
                                    set_feature("queue_size_delta_prev", conformal_have_prev_update ? (double)((long long)pq.size() - conformal_prev_queue_size) : 0.0);
                                    set_feature("prev_missing_input", conformal_pred_missing);
                                    set_feature("prev_recall_input", 1.0 - conformal_pred_missing / (double)k);
                                    set_feature("prev_source_pred", 1.0);
                                    for (size_t fi = 0; fi < conformal_feature_names.size(); fi++) {
                                        const std::string& name = conformal_feature_names[fi];
                                        if (name.rfind("topk_dist_", 0) == 0 && name.rfind("topk_dist_over_dk_", 0) != 0) {
                                            const int di = std::atoi(name.c_str() + 10);
                                            features[fi] = (di >= 0 && (size_t)di < current_dists.size()) ? (double)current_dists[(size_t)di] : (double)dk;
                                        }
                                    }
                                    auto t_features1 = std::chrono::steady_clock::now();
                                    conformal_feature_time_ns += (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(t_features1 - t_stop0).count();
                                    auto t_infer0 = t_features1;
                                    conformal_pred_missing = conformal_stop_base_score;
                                    for (const auto& tree : conformal_trees) {
                                        int node_id = 0;
                                        for (int depth_guard = 0; depth_guard < 128; depth_guard++) {
                                            const ConformalTreeNode& node = tree[(size_t)node_id];
                                            if (node.leaf) {
                                                conformal_pred_missing += node.value;
                                                break;
                                            }
                                            const double value = (node.feature >= 0 && (size_t)node.feature < features.size()) ? features[(size_t)node.feature] : 0.0;
                                            node_id = std::isfinite(value) ? (value < node.split ? node.yes : node.no) : node.missing;
                                        }
                                    }
                                    if (conformal_stop_binary_threshold >= 0.0) {
                                        const double raw_prob = 1.0 / (1.0 + std::exp(-conformal_pred_missing));
                                        conformal_stop_prob = isotonic_transform(raw_prob);
                                    } else if (conformal_pred_missing < 0.0) {
                                        conformal_pred_missing = 0.0;
                                    }
                                    const bool conformal_pop_gate_ok = pq_pop_count >= conformal_stop_min_pop;
                                    if (conformal_pop_gate_ok &&
                                        ((conformal_stop_binary_threshold >= 0.0 && conformal_stop_prob >= conformal_stop_binary_threshold) ||
                                         (conformal_stop_binary_threshold < 0.0 && conformal_pred_missing + conformal_stop_qhat <= (double)conformal_stop_allowed_missing))) {
                                        conformal_stopped = true;
                                        conformal_stop_pop = pq_pop_count;
                                        conformal_stop_update_idx = topk_update_count;
                                    }
                                    conformal_have_prev_update = true;
                                    conformal_prev_dk = dk;
                                    conformal_prev_queue_size = (long long)pq.size();
                                    auto t_stop1 = std::chrono::steady_clock::now();
                                    conformal_inference_time_ns += (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(t_stop1 - t_infer0).count();
                                    conformal_stop_time_ns += (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(t_stop1 - t_stop0).count();
                                }
#endif
                                if (collect_node_trace) {
                                    topk_update_pop_counts.push_back(pq_pop_count);
                                    topk_update_nodes.push_back(u);
                                    topk_update_push_counts.push_back(pq_push_count);
                                    topk_update_expanded_nodes.push_back(expanded_nodes);
                                    topk_update_exact_dist_computes.push_back(exact_dist_computes);
                                    topk_update_edge_visits.push_back(edge_neighbor_visits);
                                    topk_update_lb_pruned.push_back(lb_pruned);
                                    topk_update_ellipse_pruned.push_back(ellipse_pruned);
                                    topk_update_ann_cache_hits.push_back(ann_cache_hits);
                                    topk_update_queue_sizes.push_back((long long)pq.size());
                                    topk_update_pops_since_prev.push_back(pq_pop_count - last_topk_update_pop_count);
                                    topk_update_queue_min_dists.push_back(queue_min);
                                    topk_update_du_graph.push_back(du);
                                    topk_update_duq.push_back(duq);
                                    topk_update_old_dk.push_back(old_dk);
                                    topk_update_new_dk.push_back(dk);
                                    topk_update_r_prune.push_back(r_prune);
                                    topk_update_d1.push_back(cur_d1);
                                    topk_update_mean_dist.push_back(cur_mean);
                                    topk_update_median_dist.push_back(cur_median);
                                    topk_update_std_dist.push_back(cur_std);
                                    topk_update_q25_dist.push_back(cur_q25);
                                    topk_update_q75_dist.push_back(cur_q75);
                                    topk_update_gap_before_k.push_back(cur_gap_before_k);
                                    topk_update_gap_before_k5.push_back(cur_gap_before_k5);
                                    topk_update_gap_before_k10.push_back(cur_gap_before_k10);
                                    topk_update_within_1pct_dk.push_back(within_1pct);
                                    topk_update_within_5pct_dk.push_back(within_5pct);
                                    topk_update_within_10pct_dk.push_back(within_10pct);
                                    topk_update_expanded_graph_dist_mean.push_back(expanded_graph_mean);
                                    topk_update_expanded_graph_dist_max.push_back(expanded_graph_dist_max);
                                    topk_update_computed_l2_dist_mean.push_back(computed_l2_mean);
                                    topk_update_computed_l2_dist_max.push_back(computed_l2_dist_max);
                                    topk_update_labels.push_back(std::move(current_labels));
                                    topk_update_dists.push_back(std::move(current_dists));
                                }
                                last_topk_update_pop_count = pq_pop_count;
                                topk_update_count++;
                            }
                            if (r_prune < old_r_prune) {
                                r_prune_shrink_count++;
                            }
                            if (collect_node_trace && dk < old_dk) {
                                dk_update_pop_counts.push_back(pq_pop_count);
                                dk_update_nodes.push_back(u);
                                dk_update_old_values.push_back(old_dk);
                                dk_update_new_values.push_back(dk);
                                dk_update_node_dists.push_back(duq);
                            }
                        }
                    }
                    if (region_timers) {
                        auto t_region_topk1 = std::chrono::steady_clock::now();
                        region_topk_update_ns += (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(t_region_topk1 - t_region_topk0).count();
                    }
#if HNSWLIB_MBV_CONFORMAL_EARLY_STOP
                    if (conformal_stopped) break;
#endif

                    // Elliptical look-ahead pruning.
                    std::chrono::steady_clock::time_point t_region_ellipse0;
                    if (region_timers) t_region_ellipse0 = std::chrono::steady_clock::now();
                    if (use_ellipse_prune && du + duq > r_prune + dk) {
                        ellipse_pruned++;
                        ellipse_pruned_nodes.push_back(u);
                        if (region_timers) {
                            auto t_region_ellipse1 = std::chrono::steady_clock::now();
                            region_ellipse_check_ns += (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(t_region_ellipse1 - t_region_ellipse0).count();
                        }
                        if (oracle_stop_after_pop) { oracle_stopped = true; break; }
                        continue;
                    }
                    if (region_timers) {
                        auto t_region_ellipse1 = std::chrono::steady_clock::now();
                        region_ellipse_check_ns += (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(t_region_ellipse1 - t_region_ellipse0).count();
                    }
                }

                // Expand neighbors.
                std::chrono::steady_clock::time_point t_region_neighbor0;
                if (region_timers) t_region_neighbor0 = std::chrono::steady_clock::now();
                std::chrono::steady_clock::time_point t_nbr0;
                if (fine_timers) t_nbr0 = std::chrono::steady_clock::now();
                neighbor_expanded_count++;
                if (collect_node_trace) neighbor_expanded_nodes.push_back(u);
                hnswlib::linklistsizeint* ll = appr_alg->get_linklist_at_level(u, 0);
                const size_t sz = appr_alg->getListCount(ll);
                if (collect_node_trace) neighbor_expanded_degrees.push_back((long long)sz);
                edge_neighbor_visits += (long long)sz;
                hnswlib::tableint* nbrs = (hnswlib::tableint*)(ll + 1);
                std::chrono::steady_clock::time_point t_region_weight_lookup0;
                if (region_timers) t_region_weight_lookup0 = std::chrono::steady_clock::now();
                const dist_t* w_arr = get_direct_euclidean_l0_edge_cache_ptr(u);
                if (region_timers) {
                    auto t_region_weight_lookup1 = std::chrono::steady_clock::now();
                    region_edge_weight_lookup_ns += (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(t_region_weight_lookup1 - t_region_weight_lookup0).count();
                }

                std::chrono::steady_clock::time_point t_region_for_loop0;
                if (region_timers) t_region_for_loop0 = std::chrono::steady_clock::now();
                if (region_timers) {
                    for (size_t i = 0; i < sz; i++) {
                        neighbor_iter_count++;

                        auto t_iter0 = std::chrono::steady_clock::now();
                        const hnswlib::tableint v = nbrs[i];
                        auto t_iter1 = std::chrono::steady_clock::now();
                        neighbor_iter_settled_check_ns += (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(t_iter1 - t_iter0).count();

                        const dist_t w = w_arr[i];
                        const dist_t nd = du + w;
                        auto t_iter2 = std::chrono::steady_clock::now();
                        neighbor_iter_weight_nd_ns += (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(t_iter2 - t_iter1).count();

                        const bool skip_rprune = nd > r_prune;
                        const bool skip_no_improve = (!skip_rprune && nd >= ws.graph_dist(v, INF));
                        auto t_iter3 = std::chrono::steady_clock::now();
                        neighbor_iter_guard_ns += (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(t_iter3 - t_iter2).count();
                        if (skip_rprune) {
                            neighbor_iter_rprune_skip_count++;
                            continue;
                        }
                        if (skip_no_improve) {
                            neighbor_iter_no_improve_skip_count++;
                            continue;
                        }

                        ws.set_graph(v, nd, u, w);
                        auto t_iter4 = std::chrono::steady_clock::now();
                        neighbor_iter_relax_write_ns += (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(t_iter4 - t_iter3).count();

                        pq_size_before_push_sum += (long long)pq.size();
                        pq.push(QItem{nd, v});
                        auto t_iter5 = std::chrono::steady_clock::now();
                        neighbor_iter_heap_push_ns += (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(t_iter5 - t_iter4).count();
                        if (collect_node_trace) {
                            neighbor_push_parent_nodes.push_back(u);
                            neighbor_push_child_nodes.push_back(v);
                        }
                        auto t_iter6 = std::chrono::steady_clock::now();
                        neighbor_iter_trace_ns += (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(t_iter6 - t_iter5).count();
                        neighbor_iter_relax_count++;
                        pq_push_count++;
                    }
                } else {
                    for (size_t i = 0; i < sz; i++) {
                        const hnswlib::tableint v = nbrs[i];

                        const dist_t w = w_arr[i];
                        const dist_t nd = du + w;
                        if (nd > r_prune || nd >= ws.graph_dist(v, INF)) continue;

                        ws.set_graph(v, nd, u, w);
                        std::chrono::steady_clock::time_point t_push0;
                        if (fine_timers) t_push0 = std::chrono::steady_clock::now();
                        pq_size_before_push_sum += (long long)pq.size();
                        pq.push(QItem{nd, v});
                        if (collect_node_trace) {
                            neighbor_push_parent_nodes.push_back(u);
                            neighbor_push_child_nodes.push_back(v);
                        }
                        if (fine_timers) {
                            auto t_push1 = std::chrono::steady_clock::now();
                            pq_push_time_ns += (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(t_push1 - t_push0).count();
                        }
                        pq_push_count++;
                    }
                }
                if (region_timers) {
                    auto t_region_for_loop1 = std::chrono::steady_clock::now();
                    region_neighbor_for_loop_ns += (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(t_region_for_loop1 - t_region_for_loop0).count();
                    region_neighbor_for_loop_count++;
                }
                if (fine_timers) {
                    auto t_nbr1 = std::chrono::steady_clock::now();
                    neighbor_loop_time_ns += (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(t_nbr1 - t_nbr0).count();
                }
                if (region_timers) {
                    auto t_region_neighbor1 = std::chrono::steady_clock::now();
                    region_neighbor_expand_ns += (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(t_region_neighbor1 - t_region_neighbor0).count();
                }
                if (oracle_stop_after_pop) { oracle_stopped = true; break; }
            }
            auto t_loop1 = std::chrono::steady_clock::now();
            dijkstra_loop_time_ns = (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(t_loop1 - t_loop0).count();

            auto t_mbv1 = std::chrono::steady_clock::now();
            mbv_time_ns = (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(t_mbv1 - t_mbv0).count();
            total_time_ns = (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(t_mbv1 - t_all0).count();

            dk_final = dk;
            r_prune_final = r_prune;

            out.clear();
            out.reserve(topk_set.size());
            for (const auto& p : topk_set) out.push_back({p.second, p.first});
            }
        }

        const size_t k_out = out.size();
        py::array_t<hnswlib::labeltype> out_labels(k_out);
        py::array_t<dist_t> out_dists(k_out);
        for (size_t i = 0; i < out.size(); i++) {
            out_labels.mutable_at(i) = out[i].first;
            out_dists.mutable_at(i) = out[i].second;
        }

        py::dict stats;
        stats["k_requested"] = (long long)k;
        stats["k_effective"] = (long long)k_effective;
        stats["k_output"] = (long long)k_out;
        stats["t"] = (double)t;
        stats["ann_time_ns"] = ann_time_ns;
        stats["mbv_time_ns"] = mbv_time_ns;
        stats["total_time_ns"] = total_time_ns;
        stats["expanded_nodes"] = expanded_nodes;
        stats["ann_cache_hits"] = ann_cache_hits;
        stats["exact_dist_computes"] = exact_dist_computes;
        stats["edge_dist_computes"] = edge_dist_computes;
        stats["dijkstra_loop_time_ns"] = dijkstra_loop_time_ns;
        stats["pq_pop_time_ns"] = pq_pop_time_ns;
        stats["pq_push_time_ns"] = pq_push_time_ns;
        stats["exact_dist_time_ns"] = exact_dist_time_ns;
        stats["neighbor_loop_time_ns"] = neighbor_loop_time_ns;
        stats["pq_pop_count"] = pq_pop_count;
        stats["pq_push_count"] = pq_push_count;
        stats["pq_total_ops"] = pq_pop_count + pq_push_count;
        stats["pq_size_before_push_sum"] = pq_size_before_push_sum;
        stats["avg_queue_size_at_push"] = pq_push_count > 0 ? (double)pq_size_before_push_sum / (double)pq_push_count : 0.0;
        stats["region_timers_enabled"] = region_timers;
        stats["region_setup_seed_ns"] = region_setup_seed_ns;
        stats["region_pop_ns"] = region_pop_ns;
        stats["region_lb_prune_ns"] = region_lb_prune_ns;
        stats["region_exact_eval_ns"] = region_exact_eval_ns;
        stats["region_topk_update_ns"] = region_topk_update_ns;
        stats["region_ellipse_check_ns"] = region_ellipse_check_ns;
        stats["region_neighbor_expand_ns"] = region_neighbor_expand_ns;
        stats["region_edge_weight_lookup_ns"] = region_edge_weight_lookup_ns;
        stats["region_neighbor_for_loop_ns"] = region_neighbor_for_loop_ns;
        stats["region_neighbor_for_loop_count"] = region_neighbor_for_loop_count;
        stats["neighbor_iter_count"] = neighbor_iter_count;
        stats["neighbor_iter_settled_skip_count"] = neighbor_iter_settled_skip_count;
        stats["neighbor_iter_rprune_skip_count"] = neighbor_iter_rprune_skip_count;
        stats["neighbor_iter_no_improve_skip_count"] = neighbor_iter_no_improve_skip_count;
        stats["neighbor_iter_relax_count"] = neighbor_iter_relax_count;
        stats["neighbor_iter_settled_check_ns"] = neighbor_iter_settled_check_ns;
        stats["neighbor_iter_weight_nd_ns"] = neighbor_iter_weight_nd_ns;
        stats["neighbor_iter_guard_ns"] = neighbor_iter_guard_ns;
        stats["neighbor_iter_relax_write_ns"] = neighbor_iter_relax_write_ns;
        stats["neighbor_iter_heap_push_ns"] = neighbor_iter_heap_push_ns;
        stats["neighbor_iter_trace_ns"] = neighbor_iter_trace_ns;
        stats["neighbor_expanded_count"] = neighbor_expanded_count;
        stats["neighbor_expansion_skipped_count"] = pq_pop_count - neighbor_expanded_count;
        stats["edge_neighbor_visits"] = edge_neighbor_visits;
        stats["num_dist_compute"] = (long long)(exact_dist_computes + edge_dist_computes);
        stats["num_pruned_by_lb"] = lb_pruned;
        stats["num_pruned_by_ellipse"] = ellipse_pruned;
        stats["r_prune_shrink_count"] = r_prune_shrink_count;
        stats["lb_pruned_nodes"] = py::cast(lb_pruned_nodes);
        stats["ellipse_pruned_nodes"] = py::cast(ellipse_pruned_nodes);
        stats["settled_skips"] = settled_skips;
        stats["deleted_skips"] = deleted_skips;
        stats["dk_final"] = (double)dk_final;
        stats["r_prune_final"] = (double)r_prune_final;
        stats["use_ann_cache"] = use_ann_cache;
        stats["use_ellipse_prune"] = use_ellipse_prune;
        stats["collect_node_trace"] = collect_node_trace;
        stats["oracle_stop_pop"] = oracle_stop_pop;
        stats["mbv_seed_count"] = mbv_seed_count;
        stats["mbv_seed_start"] = mbv_seed_start;
        stats["oracle_stopped"] = oracle_stopped;
#if HNSWLIB_MBV_CONFORMAL_EARLY_STOP
        stats["conformal_stop_enabled"] = conformal_stop_enabled;
        stats["conformal_stopped"] = conformal_stopped;
        stats["conformal_stop_pop"] = conformal_stop_pop;
        stats["conformal_stop_update_idx"] = conformal_stop_update_idx;
        stats["conformal_stop_calls"] = conformal_stop_calls;
        stats["conformal_stop_time_ns"] = conformal_stop_time_ns;
        stats["conformal_feature_time_ns"] = conformal_feature_time_ns;
        stats["conformal_inference_time_ns"] = conformal_inference_time_ns;
        stats["conformal_stop_qhat"] = conformal_stop_qhat;
        stats["conformal_stop_allowed_missing"] = conformal_stop_allowed_missing;
        stats["conformal_stop_min_pop"] = conformal_stop_min_pop;
        stats["conformal_pred_missing_final"] = conformal_pred_missing;
        stats["conformal_stop_prob_final"] = conformal_stop_prob;
#endif
        if (collect_node_trace) {
            stats["popped_nodes"] = py::cast(popped_nodes);
            stats["exact_dist_nodes"] = py::cast(exact_dist_nodes);
            stats["skipped_exact_nodes"] = py::cast(skipped_exact_nodes);
            stats["neighbor_expanded_nodes"] = py::cast(neighbor_expanded_nodes);
            stats["neighbor_expanded_degrees"] = py::cast(neighbor_expanded_degrees);
            stats["dk_update_pop_counts"] = py::cast(dk_update_pop_counts);
            stats["dk_update_nodes"] = py::cast(dk_update_nodes);
            stats["dk_update_old_values"] = py::cast(dk_update_old_values);
            stats["dk_update_new_values"] = py::cast(dk_update_new_values);
            stats["dk_update_node_dists"] = py::cast(dk_update_node_dists);
            stats["topk_update_pop_counts"] = py::cast(topk_update_pop_counts);
            stats["topk_update_nodes"] = py::cast(topk_update_nodes);
            stats["topk_update_push_counts"] = py::cast(topk_update_push_counts);
            stats["topk_update_expanded_nodes"] = py::cast(topk_update_expanded_nodes);
            stats["topk_update_exact_dist_computes"] = py::cast(topk_update_exact_dist_computes);
            stats["topk_update_edge_visits"] = py::cast(topk_update_edge_visits);
            stats["topk_update_lb_pruned"] = py::cast(topk_update_lb_pruned);
            stats["topk_update_ellipse_pruned"] = py::cast(topk_update_ellipse_pruned);
            stats["topk_update_ann_cache_hits"] = py::cast(topk_update_ann_cache_hits);
            stats["topk_update_queue_sizes"] = py::cast(topk_update_queue_sizes);
            stats["topk_update_pops_since_prev"] = py::cast(topk_update_pops_since_prev);
            stats["topk_update_queue_min_dists"] = py::cast(topk_update_queue_min_dists);
            stats["topk_update_du_graph"] = py::cast(topk_update_du_graph);
            stats["topk_update_duq"] = py::cast(topk_update_duq);
            stats["topk_update_old_dk"] = py::cast(topk_update_old_dk);
            stats["topk_update_new_dk"] = py::cast(topk_update_new_dk);
            stats["topk_update_r_prune"] = py::cast(topk_update_r_prune);
            stats["topk_update_d1"] = py::cast(topk_update_d1);
            stats["topk_update_mean_dist"] = py::cast(topk_update_mean_dist);
            stats["topk_update_median_dist"] = py::cast(topk_update_median_dist);
            stats["topk_update_std_dist"] = py::cast(topk_update_std_dist);
            stats["topk_update_q25_dist"] = py::cast(topk_update_q25_dist);
            stats["topk_update_q75_dist"] = py::cast(topk_update_q75_dist);
            stats["topk_update_gap_before_k"] = py::cast(topk_update_gap_before_k);
            stats["topk_update_gap_before_k5"] = py::cast(topk_update_gap_before_k5);
            stats["topk_update_gap_before_k10"] = py::cast(topk_update_gap_before_k10);
            stats["topk_update_within_1pct_dk"] = py::cast(topk_update_within_1pct_dk);
            stats["topk_update_within_5pct_dk"] = py::cast(topk_update_within_5pct_dk);
            stats["topk_update_within_10pct_dk"] = py::cast(topk_update_within_10pct_dk);
            stats["topk_update_expanded_graph_dist_mean"] = py::cast(topk_update_expanded_graph_dist_mean);
            stats["topk_update_expanded_graph_dist_max"] = py::cast(topk_update_expanded_graph_dist_max);
            stats["topk_update_computed_l2_dist_mean"] = py::cast(topk_update_computed_l2_dist_mean);
            stats["topk_update_computed_l2_dist_max"] = py::cast(topk_update_computed_l2_dist_max);
            stats["topk_update_labels"] = py::cast(topk_update_labels);
            stats["topk_update_dists"] = py::cast(topk_update_dists);
            stats["neighbor_push_parent_nodes"] = py::cast(neighbor_push_parent_nodes);
            stats["neighbor_push_child_nodes"] = py::cast(neighbor_push_child_nodes);
        }

        return py::make_tuple(out_labels, out_dists, stats);
    }

};

PYBIND11_PLUGIN(hnswlib) {
        py::module m("hnswlib");

        // Slim binding surface for ANN + CRC feature collection + native MBV early stopping.
        // The previous experimental binding surface is preserved in bindings_full_backup.cpp.
        py::class_<Index<float>>(m, "Index")
        .def(py::init<const std::string &, const int>(), py::arg("space"), py::arg("dim"))
        .def("init_index",
            &Index<float>::init_new_index,
            py::arg("max_elements"),
            py::arg("M") = 16,
            py::arg("ef_construction") = 200,
            py::arg("random_seed") = 100,
            py::arg("allow_replace_deleted") = false)
        .def("add_items",
            &Index<float>::addItems,
            py::arg("data"),
            py::arg("ids") = py::none(),
            py::arg("num_threads") = -1,
            py::arg("replace_deleted") = false)
        .def("knn_query",
            &Index<float>::knnQuery_return_numpy,
            py::arg("data"),
            py::arg("k") = 1,
            py::arg("num_threads") = -1,
            py::arg("filter") = py::none())
        .def("load_index",
            &Index<float>::loadIndex,
            py::arg("path_to_index"),
            py::arg("max_elements") = 0,
            py::arg("allow_replace_deleted") = false)
        .def("save_index", &Index<float>::saveIndex, py::arg("path_to_index"))
        .def("set_ef", &Index<float>::set_ef, py::arg("ef"))
        .def("set_num_threads", &Index<float>::set_num_threads, py::arg("num_threads"))
        .def("get_current_count", &Index<float>::getCurrentCount)
        .def("get_max_elements", &Index<float>::getMaxElements)
        .def("set_mbv_fine_timers_enabled",
            &Index<float>::set_mbv_fine_timers_enabled,
            py::arg("enabled"))
        .def("get_mbv_fine_timers_enabled",
            &Index<float>::get_mbv_fine_timers_enabled)
        .def("set_mbv_region_timers_enabled",
            &Index<float>::set_mbv_region_timers_enabled,
            py::arg("enabled"))
        .def("get_mbv_region_timers_enabled",
            &Index<float>::get_mbv_region_timers_enabled)
        .def("set_store_candidate_queue_dists",
            &Index<float>::set_store_candidate_queue_dists,
            py::arg("enabled"))
        .def("set_collect_traversal_counts",
            &Index<float>::set_collect_traversal_counts,
            py::arg("enabled"))
        .def("reset_traversal_counts",
            &Index<float>::reset_traversal_counts)
        .def("get_traversal_counts",
            &Index<float>::get_traversal_counts)
        .def("get_traversal_counts_split",
            &Index<float>::get_traversal_counts_split)
        .def("get_last_query_computed_dists",
            &Index<float>::get_last_query_computed_dists,
            py::arg("euclidean") = true)
        .def("get_last_popped_candidate_dists",
            &Index<float>::get_last_popped_candidate_dists,
            py::arg("euclidean") = true)
        .def("get_last_candidate_queue_dists",
            &Index<float>::get_last_candidate_queue_dists,
            py::arg("euclidean") = true)
        .def("get_last_ef_explored",
            &Index<float>::get_last_ef_explored)
        .def("get_last_best_found_expansion",
            &Index<float>::get_last_best_found_expansion)
        .def("get_last_turn_back_count",
            &Index<float>::get_last_turn_back_count)
        .def("get_last_turn_back_total_delta",
            &Index<float>::get_last_turn_back_total_delta,
            py::arg("euclidean") = true)
        .def("label_to_internal",
            &Index<float>::label_to_internal,
            py::arg("label"))
        .def("internal_to_label",
            &Index<float>::internal_to_label,
            py::arg("internal_id"))
        .def("get_enterpoint_info",
            &Index<float>::get_enterpoint_info)
        .def("set_seeded_enterpoint_at_max_level",
            &Index<float>::set_seeded_enterpoint_at_max_level,
            py::arg("random_seed") = 0)
        .def("get_links",
            &Index<float>::get_links,
            py::arg("label"),
            py::arg("level") = 0)
        .def("get_level0_neighbors_internal",
            &Index<float>::get_level0_neighbors_internal,
            py::arg("internal_id"))
        .def("get_level0_neighbors_internal_with_weights",
            &Index<float>::get_level0_neighbors_internal_with_weights,
            py::arg("internal_id"),
            py::arg("euclidean") = true)
        .def("add_level0_edge_bidirectional",
            &Index<float>::add_level0_edge_bidirectional,
            py::arg("u_label"),
            py::arg("v_label"),
            py::arg("euclidean") = true)
        .def("remove_level0_edge_bidirectional",
            &Index<float>::remove_level0_edge_bidirectional,
            py::arg("u_label"),
            py::arg("v_label"))
        .def("remove_level0_edge_internal",
            &Index<float>::remove_level0_edge_internal,
            py::arg("src"),
            py::arg("dst"))
        .def("remove_level0_edge_bidirectional_internal",
            &Index<float>::remove_level0_edge_bidirectional_internal,
            py::arg("u"),
            py::arg("v"))
        .def("ensure_level0_edge_weight_cache",
            &Index<float>::ensure_level0_edge_weight_cache,
            py::arg("force") = false)
        .def("mbv_query_ex",
            &Index<float>::mbv_query_ex,
            py::arg("query"),
            py::arg("k") = 100,
            py::arg("t") = 3.91f,
            py::arg("euclidean") = true,
            py::arg("use_ann_cache") = true,
            py::arg("use_ellipse_prune") = true,
            py::arg("collect_node_trace") = false,
            py::arg("oracle_stop_pop") = -1,
            py::arg("mbv_seed_count") = 0,
            py::arg("mbv_seed_start") = 0
#if HNSWLIB_MBV_CONFORMAL_EARLY_STOP
            , py::arg("conformal_stop_trees") = py::none(),
            py::arg("conformal_stop_feature_names") = py::none(),
            py::arg("conformal_stop_qhat") = 0.0,
            py::arg("conformal_stop_allowed_missing") = -1,
            py::arg("conformal_stop_base_score") = 0.0,
            py::arg("conformal_stop_static_features") = py::none(),
            py::arg("conformal_stop_binary_threshold") = -1.0,
            py::arg("conformal_stop_min_pop") = 0,
            py::arg("conformal_stop_isotonic_x") = py::none(),
            py::arg("conformal_stop_isotonic_y") = py::none()
#endif
            )
        .def_readonly("space", &Index<float>::space_name)
        .def_readonly("dim", &Index<float>::dim)
        .def_readwrite("num_threads", &Index<float>::num_threads_default)
        .def_property("ef",
          [](const Index<float> & index) {
            return index.index_inited ? index.appr_alg->ef_ : index.default_ef;
          },
          [](Index<float> & index, const size_t ef_) {
            index.default_ef = ef_;
            if (index.appr_alg)
              index.appr_alg->ef_ = ef_;
        })
        .def_property_readonly("max_elements", [](const Index<float> & index) {
            return index.index_inited ? index.appr_alg->max_elements_ : 0;
        })
        .def_property_readonly("element_count", [](const Index<float> & index) {
            return index.index_inited ? (size_t)index.appr_alg->cur_element_count : 0;
        })
        .def_property_readonly("ef_construction", [](const Index<float> & index) {
          return index.index_inited ? index.appr_alg->ef_construction_ : 0;
        })
        .def_property_readonly("M",  [](const Index<float> & index) {
          return index.index_inited ? index.appr_alg->M_ : 0;
        })
        .def("__repr__", [](const Index<float> &a) {
            return "<hnswlib.Index(space='" + a.space_name + "', dim="+std::to_string(a.dim)+")>";
        });

        return m.ptr();
}
