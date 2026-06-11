#pragma once

#include "visited_list_pool.h"
#include "hnswlib.h"
#include <atomic>
#include <random>
#include <stdlib.h>
#include <assert.h>
#include <unordered_set>
#include <list>
#include <memory>

#include <chrono>
#include <cmath>
#include <queue>
#include <unordered_map>
#include <algorithm>

#include <mutex>
#include <fstream>
#include <iostream>
#include <limits>
#include <cstring>
#include <vector>
#include <cstdint>

namespace hnswlib
{
    typedef unsigned int tableint;
    typedef unsigned int linklistsizeint;

    template <typename dist_t>
    class HierarchicalNSW : public AlgorithmInterface<dist_t>
    {
    public:
        static const tableint MAX_LABEL_OPERATION_LOCKS = 65536;
        static const unsigned char DELETE_MARK = 0x01;

        size_t max_elements_{0};
        mutable std::atomic<size_t> cur_element_count{0}; // current number of elements
        size_t size_data_per_element_{0};
        size_t size_links_per_element_{0};
        mutable std::atomic<size_t> num_deleted_{0}; // number of deleted elements
        size_t M_{0};
        size_t maxM_{0};
        size_t maxM0_{0};
        size_t ef_construction_{0};
        size_t ef_{0};

        double mult_{0.0}, revSize_{0.0};
        int maxlevel_{0};

        std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};

        // Locks operations with element by label value
        mutable std::vector<std::mutex> label_op_locks_;

        std::mutex global;
        std::vector<std::mutex> link_list_locks_;

        tableint enterpoint_node_{0};

        size_t size_links_level0_{0};
        size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{0};

        char *data_level0_memory_{nullptr};
        char **linkLists_{nullptr};
        std::vector<int> element_levels_; // keeps level of each element

        size_t data_size_{0};

        DISTFUNC<dist_t> fstdistfunc_;
        void *dist_func_param_{nullptr};

        mutable std::mutex label_lookup_lock; // lock for label_lookup_
        std::unordered_map<labeltype, tableint> label_lookup_;

        std::default_random_engine level_generator_;
        std::default_random_engine update_probability_generator_;

        mutable std::atomic<long> metric_distance_computations{0};
        mutable std::atomic<long> metric_hops{0};
        mutable std::atomic<long long> last_ef_explored_{-1};
        mutable std::atomic<long long> last_best_found_expansion_{-1};
        mutable std::atomic<long long> last_turn_back_count_{0};
        mutable std::atomic<double> last_turn_back_total_delta_{0.0};

        bool allow_replace_deleted_ = false; // flag to replace deleted elements (marked as deleted) during insertions

        std::mutex deleted_elements_lock;              // lock for deleted_elements
        std::unordered_set<tableint> deleted_elements; // contains internal ids of deleted elements

        // CHANGE k
        // ---- NEW: last-query "all computed distances" cache ----
        mutable std::mutex last_qd_mutex_;
        mutable std::vector<tableint> last_qd_ids_;
        mutable std::vector<dist_t>   last_qd_dists_;
        mutable std::mutex last_popped_mutex_;
        mutable std::vector<tableint> last_popped_ids_;
        mutable std::vector<dist_t>   last_popped_dists_;
        mutable std::mutex last_cand_mutex_;
        mutable std::vector<dist_t>   last_cand_dists_;
        bool collect_last_candidate_queue_{false};
        bool store_edge_weights_{true};  // enable/disable storage
        mutable std::mutex traversal_counts_mutex_;
        mutable std::vector<uint64_t> traversal_node_counts_;
        mutable std::unordered_map<uint64_t, uint64_t> traversal_edge_counts_;
        mutable std::vector<uint64_t> traversal_upper_node_counts_;
        mutable std::unordered_map<uint64_t, uint64_t> traversal_upper_edge_counts_;
        mutable std::vector<uint64_t> traversal_l0_node_counts_;
        mutable std::unordered_map<uint64_t, uint64_t> traversal_l0_edge_counts_;
        mutable std::atomic<uint64_t> traversal_query_count_{0};
        bool collect_traversal_counts_{false};
        
        void getLastQueryComputedDists(std::vector<tableint>& ids, std::vector<dist_t>& dists) const {
            std::lock_guard<std::mutex> g(last_qd_mutex_);
            ids = last_qd_ids_;
            dists = last_qd_dists_;
        }
        void getLastPoppedCandidateDists(std::vector<tableint>& ids, std::vector<dist_t>& dists) const {
            std::lock_guard<std::mutex> g(last_popped_mutex_);
            ids = last_popped_ids_;
            dists = last_popped_dists_;
        }
        void getLastCandidateQueueDists(std::vector<dist_t>& dists) const {
            std::lock_guard<std::mutex> g(last_cand_mutex_);
            dists = last_cand_dists_;
        }
        void setCollectLastCandidateQueue(bool enabled) {
            collect_last_candidate_queue_ = enabled;
            if (!enabled) {
                std::lock_guard<std::mutex> g(last_cand_mutex_);
                last_cand_dists_.clear();
            }
        }
        void setCollectTraversalCounts(bool enabled) {
            collect_traversal_counts_ = enabled;
        }
        void resetTraversalCounts() {
            std::lock_guard<std::mutex> g(traversal_counts_mutex_);
            traversal_node_counts_.assign(cur_element_count.load(), (uint64_t)0);
            traversal_edge_counts_.clear();
            traversal_upper_node_counts_.assign(cur_element_count.load(), (uint64_t)0);
            traversal_upper_edge_counts_.clear();
            traversal_l0_node_counts_.assign(cur_element_count.load(), (uint64_t)0);
            traversal_l0_edge_counts_.clear();
            traversal_query_count_.store((uint64_t)0, std::memory_order_relaxed);
        }
        void getTraversalCounts(
            std::vector<uint64_t>& node_counts,
            std::vector<tableint>& edge_src,
            std::vector<tableint>& edge_dst,
            std::vector<uint64_t>& edge_counts,
            uint64_t& query_count) const {
            std::lock_guard<std::mutex> g(traversal_counts_mutex_);
            node_counts = traversal_node_counts_;
            edge_src.clear();
            edge_dst.clear();
            edge_counts.clear();
            edge_src.reserve(traversal_edge_counts_.size());
            edge_dst.reserve(traversal_edge_counts_.size());
            edge_counts.reserve(traversal_edge_counts_.size());
            for (const auto& kv : traversal_edge_counts_) {
                edge_src.push_back((tableint)(kv.first >> 32));
                edge_dst.push_back((tableint)(kv.first & 0xffffffffULL));
                edge_counts.push_back(kv.second);
            }
            query_count = traversal_query_count_.load(std::memory_order_relaxed);
        }
        void getTraversalCountsSplit(
            std::vector<uint64_t>& upper_node_counts,
            std::vector<tableint>& upper_edge_src,
            std::vector<tableint>& upper_edge_dst,
            std::vector<uint64_t>& upper_edge_counts,
            std::vector<uint64_t>& l0_node_counts,
            std::vector<tableint>& l0_edge_src,
            std::vector<tableint>& l0_edge_dst,
            std::vector<uint64_t>& l0_edge_counts,
            uint64_t& query_count) const {
            std::lock_guard<std::mutex> g(traversal_counts_mutex_);
            upper_node_counts = traversal_upper_node_counts_;
            upper_edge_src.clear();
            upper_edge_dst.clear();
            upper_edge_counts.clear();
            upper_edge_src.reserve(traversal_upper_edge_counts_.size());
            upper_edge_dst.reserve(traversal_upper_edge_counts_.size());
            upper_edge_counts.reserve(traversal_upper_edge_counts_.size());
            for (const auto& kv : traversal_upper_edge_counts_) {
                upper_edge_src.push_back((tableint)(kv.first >> 32));
                upper_edge_dst.push_back((tableint)(kv.first & 0xffffffffULL));
                upper_edge_counts.push_back(kv.second);
            }

            l0_node_counts = traversal_l0_node_counts_;
            l0_edge_src.clear();
            l0_edge_dst.clear();
            l0_edge_counts.clear();
            l0_edge_src.reserve(traversal_l0_edge_counts_.size());
            l0_edge_dst.reserve(traversal_l0_edge_counts_.size());
            l0_edge_counts.reserve(traversal_l0_edge_counts_.size());
            for (const auto& kv : traversal_l0_edge_counts_) {
                l0_edge_src.push_back((tableint)(kv.first >> 32));
                l0_edge_dst.push_back((tableint)(kv.first & 0xffffffffULL));
                l0_edge_counts.push_back(kv.second);
            }
            query_count = traversal_query_count_.load(std::memory_order_relaxed);
        }
        uint64_t getTraversalQueryCount() const {
            return traversal_query_count_.load(std::memory_order_relaxed);
        }
        static inline uint64_t packTraversalEdge(tableint src, tableint dst) {
            return (uint64_t(src) << 32) | uint64_t(dst);
        }

        long long getLastEfExplored() const {
            return last_ef_explored_.load(std::memory_order_relaxed);
        }
        long long getLastBestFoundExpansion() const {
            return last_best_found_expansion_.load(std::memory_order_relaxed);
        }
        long long getLastTurnBackCount() const {
            return last_turn_back_count_.load(std::memory_order_relaxed);
        }
        double getLastTurnBackTotalDelta() const {
            return last_turn_back_total_delta_.load(std::memory_order_relaxed);
        }
        
        // IDs live at (ll + 1). We store weights in a parallel block after the *capacity* IDs.
        inline dist_t* get_linklist_weights_ptr(linklistsizeint* ll, int level) const {
            if (!store_edge_weights_) return nullptr;
            const size_t cap = (level == 0) ? maxM0_ : maxM_;
            char* ids_base = (char*)(ll + 1); // start of tableint ids
            return (dist_t*)(ids_base + cap * sizeof(tableint));
        }

        inline const dist_t* get_linklist_weights_ptr(const linklistsizeint* ll, int level) const {
            return const_cast<HierarchicalNSW*>(this)->get_linklist_weights_ptr(const_cast<linklistsizeint*>(ll), level);
        }

        // CHANGE 5
        /******************* START: Post-hoc certification (4 checks) *******************/

        // Timing helper (used by stats)
        static inline uint64_t now_ns()
        {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(
                       std::chrono::steady_clock::now().time_since_epoch())
                .count();
        }

        // NOTE: Keep name EarlyStopConfig to avoid changing bindings.
        // Semantics: enabled == "run post-hoc certification checks", NOT early-stop.
        struct EarlyStopConfig
        {
            bool enabled = true; // TRUE => run certification after search completes
            float epsilon = 0.0f;

            size_t K = 10;             // certify top-K (often min(ef, K))
            size_t delta_graph = 1;    // δ_graph for Check 3
            size_t delta_landmark = 4; // δ_landmark for Check 4
            size_t landmark_k = 2;     // |L| for Check 4

            int n_clusters = 0; // 0 => disable Check 2
        };

        // Stored configuration (set from Python)
        EarlyStopConfig early_;

        // Stats from post-hoc certification
        struct EarlyStopStats
        {
            uint64_t check1_fail = 0;
            uint64_t check2_fail = 0;
            uint64_t check3_fail = 0;
            uint64_t check4_fail = 0;

            uint64_t checks_executed = 0;
            uint64_t certified_count = 0;

            uint64_t dist_comps_check1 = 0;
            uint64_t dist_comps_check2 = 0;
            uint64_t dist_comps_check3 = 0;
            uint64_t dist_comps_check4 = 0;

            uint64_t bfs_nodes_check3 = 0;
            uint64_t bfs_nodes_check4 = 0;

            uint64_t time_check1_ns = 0;
            uint64_t time_check2_ns = 0;
            uint64_t time_check3_ns = 0;
            uint64_t time_check4_ns = 0;

            uint64_t bfs_check3_ns = 0;
            uint64_t bfs_check4_ns = 0;

            uint64_t comparisons_check1 = 0;
            uint64_t comparisons_check2 = 0;
            uint64_t comparisons_check3 = 0;
            uint64_t comparisons_check4 = 0;

            void reset() { *this = EarlyStopStats{}; }
        };

        mutable EarlyStopStats early_stats_;

        // --------- Check 2 data (cluster bounds) ---------
        std::vector<int> cluster_labels_;    // size >= cur_element_count
        std::vector<float> cluster_centers_; // size = n_clusters * dim
        std::vector<float> cluster_radii_;   // size = n_clusters

        // --------- Check 3/4 data (graph bounds) ---------
        std::vector<dist_t> d_max_; // size >= cur_element_count (or max_elements_)

        // ---------------- Setters (called from Python) ----------------

        // Enable/disable certification and set epsilon
        void setEarlyStopParams(bool enabled, float epsilon)
        {
            early_.enabled = enabled;
            early_.epsilon = epsilon;
        }

        // Set certify parameters (K/deltas/landmarks)
        void setCertifyParams(size_t K, size_t delta_graph, size_t delta_landmark, size_t landmark_k)
        {
            early_.K = K;
            early_.delta_graph = delta_graph;
            early_.delta_landmark = delta_landmark;
            early_.landmark_k = landmark_k;
        }

        // Provide cluster info (Check 2)
        void setClusterData(int n_clusters,
                            const std::vector<int> &labels,
                            const std::vector<float> &centers,
                            const std::vector<float> &radii)
        {
            early_.n_clusters = n_clusters;
            cluster_labels_ = labels;
            cluster_centers_ = centers;
            cluster_radii_ = radii;
        }

        // Compute d_max(v) from level-0 graph (called after build)
        void computeDmaxLevel0()
        {
            d_max_.assign(max_elements_, (dist_t)0);

            const size_t n = cur_element_count.load();
            for (tableint v = 0; v < (tableint)n; v++)
            {
                linklistsizeint *ll = get_linklist0(v);
                size_t sz = getListCount(ll);

                char *vdata = getDataByInternalId(v);
                tableint *nbrs = (tableint *)(ll + 1);

                dist_t mx = (dist_t)0;
                for (size_t j = 0; j < sz; j++)
                {
                    tableint nb = nbrs[j];
                    if ((size_t)nb >= n)
                        continue;
                    char *nbdata = getDataByInternalId(nb);

                    dist_t d = fstdistfunc_(vdata, nbdata, dist_func_param_);
                    if (d > mx)
                        mx = d;
                }
                d_max_[v] = mx;
            }
        }

        // Stats accessors
        EarlyStopStats getEarlyStopStats() const { return early_stats_; }
        void resetEarlyStopStats() { early_stats_.reset(); }

        // Helper: vector dimension (for cluster center layout)
        inline size_t get_dim() const { return data_size_ / sizeof(float); }

        /******************* END: Post-hoc certification (4 checks) *******************/

        // CHANGE 5

        HierarchicalNSW(SpaceInterface<dist_t> *s)
        {
        }

        HierarchicalNSW(
            SpaceInterface<dist_t> *s,
            const std::string &location,
            bool nmslib = false,
            size_t max_elements = 0,
            bool allow_replace_deleted = false)
            : allow_replace_deleted_(allow_replace_deleted)
        {
            loadIndex(location, s, max_elements);
        }

        HierarchicalNSW(
            SpaceInterface<dist_t> *s,
            size_t max_elements,
            size_t M = 16,
            size_t ef_construction = 200,
            size_t random_seed = 100,
            bool allow_replace_deleted = false)
            : label_op_locks_(MAX_LABEL_OPERATION_LOCKS),
              link_list_locks_(max_elements),
              element_levels_(max_elements),
              allow_replace_deleted_(allow_replace_deleted)
        {
            max_elements_ = max_elements;
            num_deleted_ = 0;
            data_size_ = s->get_data_size();
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();
            if (M <= 10000)
            {
                M_ = M;
            }
            else
            {
                HNSWERR << "warning: M parameter exceeds 10000 which may lead to adverse effects." << std::endl;
                HNSWERR << "         Cap to 10000 will be applied for the rest of the processing." << std::endl;
                M_ = 10000;
            }
            maxM_ = M_;
            maxM0_ = M_ * 2;
            ef_construction_ = std::max(ef_construction, M_);
            ef_ = 10;

            level_generator_.seed(random_seed);
            update_probability_generator_.seed(random_seed + 1);

            // CHANGE K:
            // size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
            size_links_level0_ = sizeof(linklistsizeint) + maxM0_ * sizeof(tableint)
                   + (store_edge_weights_ ? maxM0_ * sizeof(dist_t) : 0);

            size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);
            offsetData_ = size_links_level0_;
            label_offset_ = size_links_level0_ + data_size_;
            offsetLevel0_ = 0;
            // 

            size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);
            offsetData_ = size_links_level0_;
            label_offset_ = size_links_level0_ + data_size_;
            offsetLevel0_ = 0;

            data_level0_memory_ = (char *)malloc(max_elements_ * size_data_per_element_);
            if (data_level0_memory_ == nullptr)
                throw std::runtime_error("Not enough memory");

            cur_element_count = 0;

            visited_list_pool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, max_elements));

            // initializations for special treatment of the first node
            enterpoint_node_ = -1;
            maxlevel_ = -1;

            linkLists_ = (char **)malloc(sizeof(void *) * max_elements_);
            if (linkLists_ == nullptr)
                throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");
            // CHANGE K:
            // size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
            size_links_per_element_ = sizeof(linklistsizeint) + maxM_ * sizeof(tableint)
                        + (store_edge_weights_ ? maxM_ * sizeof(dist_t) : 0);
            //

            mult_ = 1 / log(1.0 * M_);
            revSize_ = 1.0 / mult_;
        }

        ~HierarchicalNSW()
        {
            clear();
        }

        void clear()
        {
            free(data_level0_memory_);
            data_level0_memory_ = nullptr;
            for (tableint i = 0; i < cur_element_count; i++)
            {
                if (element_levels_[i] > 0)
                    free(linkLists_[i]);
            }
            free(linkLists_);
            linkLists_ = nullptr;
            cur_element_count = 0;
            visited_list_pool_.reset(nullptr);
        }

        struct CompareByFirst
        {
            constexpr bool operator()(std::pair<dist_t, tableint> const &a,
                                      std::pair<dist_t, tableint> const &b) const noexcept
            {
                return a.first < b.first;
            }
        };

        void setEf(size_t ef)
        {
            ef_ = ef;
        }

        inline std::mutex &getLabelOpMutex(labeltype label) const
        {
            // calculate hash
            size_t lock_id = label & (MAX_LABEL_OPERATION_LOCKS - 1);
            return label_op_locks_[lock_id];
        }

        inline labeltype getExternalLabel(tableint internal_id) const
        {
            labeltype return_label;
            memcpy(&return_label, (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), sizeof(labeltype));
            return return_label;
        }

        inline void setExternalLabel(tableint internal_id, labeltype label) const
        {
            memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), &label, sizeof(labeltype));
        }

        inline labeltype *getExternalLabeLp(tableint internal_id) const
        {
            return (labeltype *)(data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_);
        }

        inline char *getDataByInternalId(tableint internal_id) const
        {
            return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
        }

        int getRandomLevel(double reverse_size)
        {
            std::uniform_real_distribution<double> distribution(0.0, 1.0);
            double r = -log(distribution(level_generator_)) * reverse_size;
            return (int)r;
        }

        size_t getMaxElements()
        {
            return max_elements_;
        }

        size_t getCurrentElementCount()
        {
            return cur_element_count;
        }

        size_t getDeletedCount()
        {
            return num_deleted_;
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        searchBaseLayer(tableint ep_id, const void *data_point, int layer)
        {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

            dist_t lowerBound;
            if (!isMarkedDeleted(ep_id))
            {
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
                top_candidates.emplace(dist, ep_id);
                lowerBound = dist;
                candidateSet.emplace(-dist, ep_id);
            }
            else
            {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidateSet.emplace(-lowerBound, ep_id);
            }
            visited_array[ep_id] = visited_array_tag;

            while (!candidateSet.empty())
            {
                std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
                if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_)
                {
                    break;
                }
                candidateSet.pop();

                tableint curNodeNum = curr_el_pair.second;

                std::unique_lock<std::mutex> lock(link_list_locks_[curNodeNum]);

                int *data; // = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
                if (layer == 0)
                {
                    data = (int *)get_linklist0(curNodeNum);
                }
                else
                {
                    data = (int *)get_linklist(curNodeNum, layer);
                    //                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
                }
                size_t size = getListCount((linklistsizeint *)data);
                tableint *datal = (tableint *)(data + 1);
#ifdef USE_SSE
                _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

                for (size_t j = 0; j < size; j++)
                {
                    tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                    _mm_prefetch((char *)(visited_array + *(datal + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                    if (visited_array[candidate_id] == visited_array_tag)
                        continue;
                    visited_array[candidate_id] = visited_array_tag;
                    char *currObj1 = (getDataByInternalId(candidate_id));

                    dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                    if (top_candidates.size() < ef_construction_ || lowerBound > dist1)
                    {
                        candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                        if (!isMarkedDeleted(candidate_id))
                            top_candidates.emplace(dist1, candidate_id);

                        if (top_candidates.size() > ef_construction_)
                            top_candidates.pop();

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
            visited_list_pool_->releaseVisitedList(vl);

            return top_candidates;
        }

        // bare_bone_search means there is no check for deletions and stop condition is ignored in return of extra performance
        struct TraversalLocalCounts {
            std::unordered_map<tableint, uint32_t> upper_node_counts;
            std::unordered_map<uint64_t, uint32_t> upper_edge_counts;
            std::unordered_map<tableint, uint32_t> l0_node_counts;
            std::unordered_map<uint64_t, uint32_t> l0_edge_counts;
        };

        template <bool bare_bone_search = true, bool collect_metrics = false>
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        searchBaseLayerST(
            tableint ep_id,
            const void *data_point,
            size_t ef,
            BaseFilterFunctor *isIdAllowed = nullptr,
            BaseSearchStopCondition<dist_t> *stop_condition = nullptr,
            TraversalLocalCounts *traversal_local = nullptr) const
        {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            size_t explored_count = 0;

            // CHANGE 1
            const bool collect_cert = early_.enabled;
            std::vector<tableint> last_visited_;
            std::vector<dist_t> last_distances_;
            if (collect_cert) {
                last_visited_.reserve(ef * 4);      // heuristic reserve
                last_distances_.reserve(ef * 4);
            }
            // CHANGE 1
            const bool collect_qd = early_.enabled; // reuse your existing flag toggle
            
            // CHANGE 6
            std::vector<tableint> touched;
            std::vector<dist_t> best; // size = cur_element_count_ at query time
            std::vector<tableint> popped_ids;
            std::vector<dist_t> popped_dists;

            if (collect_qd) {
                touched.reserve(ef * 8); // heuristic
                best.assign(cur_element_count.load(), std::numeric_limits<dist_t>::infinity());
                popped_ids.reserve(ef * 2);
                popped_dists.reserve(ef * 2);
            }

            auto record_qd = [&](tableint id, dist_t d) {
                if (!collect_qd) return;
                if (id < 0 || (size_t)id >= best.size()) return;

                if (std::isinf(best[id])) touched.push_back(id);
                if (d < best[id]) best[id] = d;
            };
            // CHANGE 6

            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

            // Query-trace stats for additional CRC/LTT features.
            double prev_expansion_dist = 0.0;
            double prev_delta = 0.0;
            size_t turn_back_count = 0;
            double turn_back_total_delta = 0.0;
            dist_t best_result_dist = std::numeric_limits<dist_t>::infinity();
            long long best_result_expansion = -1;

            dist_t lowerBound;
            if (bare_bone_search ||
                (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id)))))
            {
                char *ep_data = getDataByInternalId(ep_id);
                dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_);
                record_qd(ep_id, dist);

                // change 8
                if (collect_cert) {
                    last_visited_.push_back(ep_id);
                    last_distances_.push_back(dist);
                }
                // change 8

                lowerBound = dist;
                top_candidates.emplace(dist, ep_id);
                best_result_dist = dist;
                best_result_expansion = 0; // entry-point is known before any base-level expansion pop.
                if (!bare_bone_search && stop_condition)
                {
                    stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
                }
                candidate_set.emplace(-dist, ep_id);
            }
            else
            {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidate_set.emplace(-lowerBound, ep_id);
            }

            visited_array[ep_id] = visited_array_tag;

            while (!candidate_set.empty())
            {
                std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
                dist_t candidate_dist = -current_node_pair.first;

                bool flag_stop_search;
                if (bare_bone_search)
                {
                    flag_stop_search = candidate_dist > lowerBound;
                }
                else
                {
                    if (stop_condition)
                    {
                        flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
                    }
                    else
                    {
                        flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
                    }
                }
                if (flag_stop_search)
                {
                    break;
                }
                candidate_set.pop();
                explored_count++;

                // Track turn-backs on base-level expansion distance trace.
                if (explored_count == 1) {
                    // first expansion point
                } else if (explored_count == 2) {
                    prev_delta = (double)candidate_dist - prev_expansion_dist;
                } else {
                    double curr_delta = (double)candidate_dist - prev_expansion_dist;
                    if ((prev_delta > 0.0 && curr_delta < 0.0) || (prev_delta < 0.0 && curr_delta > 0.0)) {
                        turn_back_count++;
                        turn_back_total_delta += std::fabs(curr_delta);
                    }
                    prev_delta = curr_delta;
                }
                prev_expansion_dist = (double)candidate_dist;

                tableint current_node_id = current_node_pair.second;
                if (collect_qd) {
                    popped_ids.push_back(current_node_id);
                    popped_dists.push_back(candidate_dist);
                }
                if (traversal_local) {
                    traversal_local->l0_node_counts[current_node_id]++;
                }

                if (collect_cert) {
                    last_visited_.push_back(current_node_id);
                    last_distances_.push_back(candidate_dist);
                }

                int *data = (int *)get_linklist0(current_node_id);
                size_t size = getListCount((linklistsizeint *)data);
                //                bool cur_node_deleted = isMarkedDeleted(current_node_id);
                if (collect_metrics)
                {
                    metric_hops++;
                    metric_distance_computations += size;
                }

#ifdef USE_SSE
                _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
                _mm_prefetch((char *)(data + 2), _MM_HINT_T0);
#endif

                for (size_t j = 1; j <= size; j++)
                {
                    // change 6
                        tableint candidate_id = (tableint)*(data + j);
                        if ((size_t)candidate_id >= (size_t)cur_element_count.load()) continue;
                        if (traversal_local) {
                            traversal_local->l0_edge_counts[packTraversalEdge(current_node_id, candidate_id)]++;
                        }
                    // change 6
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                    _mm_prefetch((char *)(visited_array + *(data + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                 _MM_HINT_T0); ////////////
#endif
                    if (!(visited_array[candidate_id] == visited_array_tag))
                    {
                        visited_array[candidate_id] = visited_array_tag;

                        char *currObj1 = (getDataByInternalId(candidate_id));
                        dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);
                        record_qd(candidate_id, dist);
                        
                        // CHANGE 2
                        if (collect_cert) {
                            last_visited_.push_back(candidate_id);
                            last_distances_.push_back(dist);
                        }
                        // CHANGE 2

                        bool flag_consider_candidate;
                        if (!bare_bone_search && stop_condition)
                        {
                            flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
                        }
                        else
                        {
                            flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                        }

                        if (flag_consider_candidate)
                        {
                            candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                            _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                             offsetLevel0_, ///////////
                                         _MM_HINT_T0);      ////////////////////////
#endif

                            if (bare_bone_search ||
                                (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id)))))
                            {
                                top_candidates.emplace(dist, candidate_id);
                                if (dist < best_result_dist) {
                                    best_result_dist = dist;
                                    best_result_expansion = explored_count;
                                }
                                if (!bare_bone_search && stop_condition)
                                {
                                    stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
                                }
                            }

                            bool flag_remove_extra = false;
                            if (!bare_bone_search && stop_condition)
                            {
                                flag_remove_extra = stop_condition->should_remove_extra();
                            }
                            else
                            {
                                flag_remove_extra = top_candidates.size() > ef;
                            }
                            while (flag_remove_extra)
                            {
                                tableint id = top_candidates.top().second;
                                top_candidates.pop();
                                if (!bare_bone_search && stop_condition)
                                {
                                    stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
                                    flag_remove_extra = stop_condition->should_remove_extra();
                                }
                                else
                                {
                                    flag_remove_extra = top_candidates.size() > ef;
                                }
                            }

                            if (!top_candidates.empty())
                                lowerBound = top_candidates.top().first;
                        }
                    }
                }
            }

            // CHANGE 4 (post-hoc certification)
            // NOTE: this does NOT stop search; it only computes a certificate and updates stats.
            if (collect_cert) {
                (void)certify_result(
                    data_point,
                    std::min(ef, (size_t)early_.K),
                    top_candidates,
                    last_visited_,
                    last_distances_);
            }
            // CHANGE 4

            visited_list_pool_->releaseVisitedList(vl);

            // CHANGE k
            if (collect_qd) {
                std::vector<tableint> out_ids;
                std::vector<dist_t> out_ds;
                out_ids.reserve(touched.size());
                out_ds.reserve(touched.size());
                for (auto id : touched) {
                    // best[id] is finite if touched
                    out_ids.push_back(id);
                    out_ds.push_back(best[id]);
                }
                {
                    std::lock_guard<std::mutex> g(last_qd_mutex_);
                    last_qd_ids_.swap(out_ids);
                    last_qd_dists_.swap(out_ds);
                }
                {
                    std::lock_guard<std::mutex> g(last_popped_mutex_);
                    last_popped_ids_.swap(popped_ids);
                    last_popped_dists_.swap(popped_dists);
                }
            }
            if (collect_last_candidate_queue_) {
                // Distances to unchecked points still in candidate queue at termination.
                auto cand_copy = candidate_set;
                std::vector<dist_t> rem_dists;
                rem_dists.reserve(cand_copy.size());
                while (!cand_copy.empty()) {
                    rem_dists.push_back((dist_t)(-cand_copy.top().first));
                    cand_copy.pop();
                }
                std::lock_guard<std::mutex> g(last_cand_mutex_);
                last_cand_dists_.swap(rem_dists);
            }
            last_ef_explored_.store((long long)explored_count, std::memory_order_relaxed);
            last_best_found_expansion_.store((long long)best_result_expansion, std::memory_order_relaxed);
            last_turn_back_count_.store((long long)turn_back_count, std::memory_order_relaxed);
            last_turn_back_total_delta_.store((double)turn_back_total_delta, std::memory_order_relaxed);

            return top_candidates;
        }

        void getNeighborsByHeuristic2(
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
            const size_t M)
        {
            if (top_candidates.size() < M)
            {
                return;
            }

            std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
            std::vector<std::pair<dist_t, tableint>> return_list;
            while (top_candidates.size() > 0)
            {
                queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
                top_candidates.pop();
            }

            while (queue_closest.size())
            {
                if (return_list.size() >= M)
                    break;
                std::pair<dist_t, tableint> curent_pair = queue_closest.top();
                dist_t dist_to_query = -curent_pair.first;
                queue_closest.pop();
                bool good = true;

                for (std::pair<dist_t, tableint> second_pair : return_list)
                {
                    dist_t curdist =
                        fstdistfunc_(getDataByInternalId(second_pair.second),
                                     getDataByInternalId(curent_pair.second),
                                     dist_func_param_);
                    if (curdist < dist_to_query)
                    {
                        good = false;
                        break;
                    }
                }
                if (good)
                {
                    return_list.push_back(curent_pair);
                }
            }

            for (std::pair<dist_t, tableint> curent_pair : return_list)
            {
                top_candidates.emplace(-curent_pair.first, curent_pair.second);
            }
        }

        linklistsizeint *get_linklist0(tableint internal_id) const
        {
            return (linklistsizeint *)(data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
        }

        linklistsizeint *get_linklist0(tableint internal_id, char *data_level0_memory_) const
        {
            return (linklistsizeint *)(data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
        }

        linklistsizeint *get_linklist(tableint internal_id, int level) const
        {
            return (linklistsizeint *)(linkLists_[internal_id] + (level - 1) * size_links_per_element_);
        }

        linklistsizeint *get_linklist_at_level(tableint internal_id, int level) const
        {
            return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
        }

        tableint mutuallyConnectNewElement(
    const void *data_point,
    tableint cur_c,
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
    int level,
    bool isUpdate)
{
    size_t Mcurmax = level ? maxM_ : maxM0_;
    getNeighborsByHeuristic2(top_candidates, M_);
    if (top_candidates.size() > M_)
        throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

    // Keep (dist, id) so we can store edge weights without recomputing.
    std::vector<std::pair<dist_t, tableint>> selected;
    selected.reserve(M_);
    while (!top_candidates.empty())
    {
        selected.push_back(top_candidates.top()); // (dist, id)
        top_candidates.pop();
    }

    tableint next_closest_entry_point = selected.back().second;

    {
        // lock only during the update
        // because during the addition the lock for cur_c is already acquired
        std::unique_lock<std::mutex> lock(link_list_locks_[cur_c], std::defer_lock);
        if (isUpdate)
        {
            lock.lock();
        }

        linklistsizeint *ll_cur = (level == 0) ? get_linklist0(cur_c) : get_linklist(cur_c, level);

        if (*ll_cur && !isUpdate)
            throw std::runtime_error("The newly inserted element should have blank link list");

        setListCount(ll_cur, selected.size());

        tableint *ids = (tableint *)(ll_cur + 1);
        dist_t  *ws  = get_linklist_weights_ptr(ll_cur, level); // may be nullptr if disabled

        for (size_t idx = 0; idx < selected.size(); idx++)
        {
            const tableint nb = selected[idx].second;
            const dist_t  w  = selected[idx].first;

            if (ids[idx] && !isUpdate)
                throw std::runtime_error("Possible memory corruption");
            if (level > element_levels_[nb])
                throw std::runtime_error("Trying to make a link on a non-existent level");

            ids[idx] = nb;
            if (ws) ws[idx] = w;   // store weight cur_c -> nb
        }
    }

    for (size_t idx = 0; idx < selected.size(); idx++)
    {
        const tableint other = selected[idx].second;
        const dist_t w_cur_other = selected[idx].first;

        std::unique_lock<std::mutex> lock(link_list_locks_[other]);

        linklistsizeint *ll_other = (level == 0) ? get_linklist0(other) : get_linklist(other, level);
        size_t sz_link_list_other = getListCount(ll_other);

        if (sz_link_list_other > Mcurmax)
            throw std::runtime_error("Bad value of sz_link_list_other");
        if (other == cur_c)
            throw std::runtime_error("Trying to connect an element to itself");
        if (level > element_levels_[other])
            throw std::runtime_error("Trying to make a link on a non-existent level");

        tableint *ids_other = (tableint *)(ll_other + 1);
        dist_t  *ws_other  = get_linklist_weights_ptr(ll_other, level);

        bool is_cur_c_present = false;
        size_t cur_pos = 0;

        if (isUpdate)
        {
            for (size_t j = 0; j < sz_link_list_other; j++)
            {
                if (ids_other[j] == cur_c)
                {
                    is_cur_c_present = true;
                    cur_pos = j;
                    break;
                }
            }
        }

        if (is_cur_c_present)
        {
            if (ws_other) ws_other[cur_pos] = w_cur_other;
            continue;
        }

        if (sz_link_list_other < Mcurmax)
        {
            ids_other[sz_link_list_other] = cur_c;
            if (ws_other) ws_other[sz_link_list_other] = w_cur_other;
            setListCount(ll_other, sz_link_list_other + 1);
        }
        else
        {
            std::priority_queue<std::pair<dist_t, tableint>,
                                std::vector<std::pair<dist_t, tableint>>,
                                CompareByFirst> candidates;

            candidates.emplace(w_cur_other, cur_c);

            for (size_t j = 0; j < sz_link_list_other; j++)
            {
                tableint nb = ids_other[j];
                dist_t d = ws_other ? ws_other[j]
                                    : fstdistfunc_(getDataByInternalId(nb),
                                                   getDataByInternalId(other),
                                                   dist_func_param_);
                candidates.emplace(d, nb);
            }

            getNeighborsByHeuristic2(candidates, Mcurmax);

            int out = 0;
            while (!candidates.empty())
            {
                ids_other[out] = candidates.top().second;
                if (ws_other) ws_other[out] = candidates.top().first;
                candidates.pop();
                out++;
            }
            setListCount(ll_other, out);
        }
    }

    return next_closest_entry_point;
}

        void resizeIndex(size_t new_max_elements)
        {
            if (new_max_elements < cur_element_count)
                throw std::runtime_error("Cannot resize, max element is less than the current number of elements");

            visited_list_pool_.reset(new VisitedListPool(1, new_max_elements));

            element_levels_.resize(new_max_elements);

            std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);

            // Reallocate base layer
            char *data_level0_memory_new = (char *)realloc(data_level0_memory_, new_max_elements * size_data_per_element_);
            if (data_level0_memory_new == nullptr)
                throw std::runtime_error("Not enough memory: resizeIndex failed to allocate base layer");
            data_level0_memory_ = data_level0_memory_new;

            // Reallocate all other layers
            char **linkLists_new = (char **)realloc(linkLists_, sizeof(void *) * new_max_elements);
            if (linkLists_new == nullptr)
                throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
            linkLists_ = linkLists_new;

            max_elements_ = new_max_elements;
        }

        size_t indexFileSize() const
        {
            size_t size = 0;
            size += sizeof(offsetLevel0_);
            size += sizeof(max_elements_);
            size += sizeof(cur_element_count);
            size += sizeof(size_data_per_element_);
            size += sizeof(label_offset_);
            size += sizeof(offsetData_);
            size += sizeof(maxlevel_);
            size += sizeof(enterpoint_node_);
            size += sizeof(maxM_);

            size += sizeof(maxM0_);
            size += sizeof(M_);
            size += sizeof(mult_);
            size += sizeof(ef_construction_);

            size += cur_element_count * size_data_per_element_;

            for (size_t i = 0; i < cur_element_count; i++)
            {
                unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
                size += sizeof(linkListSize);
                size += linkListSize;
            }
            return size;
        }

        void saveIndex(const std::string &location)
        {
            std::ofstream output(location, std::ios::binary);
            std::streampos position;

            writeBinaryPOD(output, offsetLevel0_);
            writeBinaryPOD(output, max_elements_);
            writeBinaryPOD(output, cur_element_count);
            writeBinaryPOD(output, size_data_per_element_);
            writeBinaryPOD(output, label_offset_);
            writeBinaryPOD(output, offsetData_);
            writeBinaryPOD(output, maxlevel_);
            writeBinaryPOD(output, enterpoint_node_);
            writeBinaryPOD(output, maxM_);

            writeBinaryPOD(output, maxM0_);
            writeBinaryPOD(output, M_);
            writeBinaryPOD(output, mult_);
            writeBinaryPOD(output, ef_construction_);

            output.write(data_level0_memory_, cur_element_count * size_data_per_element_);

            for (size_t i = 0; i < cur_element_count; i++)
            {
                unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
                writeBinaryPOD(output, linkListSize);
                if (linkListSize)
                    output.write(linkLists_[i], linkListSize);
            }
            output.close();
        }

        void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i = 0)
        {
            std::ifstream input(location, std::ios::binary);

            if (!input.is_open())
                throw std::runtime_error("Cannot open file");

            clear();
            // get file size:
            input.seekg(0, input.end);
            std::streampos total_filesize = input.tellg();
            input.seekg(0, input.beg);

            readBinaryPOD(input, offsetLevel0_);
            readBinaryPOD(input, max_elements_);
            readBinaryPOD(input, cur_element_count);

            size_t max_elements = max_elements_i;
            if (max_elements < cur_element_count)
                max_elements = max_elements_;
            max_elements_ = max_elements;
            readBinaryPOD(input, size_data_per_element_);
            readBinaryPOD(input, label_offset_);
            readBinaryPOD(input, offsetData_);
            readBinaryPOD(input, maxlevel_);
            readBinaryPOD(input, enterpoint_node_);

            readBinaryPOD(input, maxM_);
            readBinaryPOD(input, maxM0_);
            readBinaryPOD(input, M_);
            readBinaryPOD(input, mult_);
            readBinaryPOD(input, ef_construction_);

            // CHANGE K:
            const size_t base0 = sizeof(linklistsizeint) + maxM0_ * sizeof(tableint);
            const size_t want0 = base0 + maxM0_ * sizeof(dist_t);

            // if offsetData_ includes the weights area, then weights exist
            store_edge_weights_ = (offsetData_ >= want0);
            //


            data_size_ = s->get_data_size();
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();

            auto pos = input.tellg();

            /// Optional - check if index is ok:
            input.seekg(cur_element_count * size_data_per_element_, input.cur);
            for (size_t i = 0; i < cur_element_count; i++)
            {
                if (input.tellg() < 0 || input.tellg() >= total_filesize)
                {
                    throw std::runtime_error("Index seems to be corrupted or unsupported");
                }

                unsigned int linkListSize;
                readBinaryPOD(input, linkListSize);
                if (linkListSize != 0)
                {
                    input.seekg(linkListSize, input.cur);
                }
            }

            // throw exception if it either corrupted or old index
            if (input.tellg() != total_filesize)
                throw std::runtime_error("Index seems to be corrupted or unsupported");

            input.clear();
            /// Optional check end

            input.seekg(pos, input.beg);

            data_level0_memory_ = (char *)malloc(max_elements * size_data_per_element_);
            if (data_level0_memory_ == nullptr)
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
            input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

            // CHANGE K: old saved indexes will load with store_edge_weights_ = false, so Dijkstra will fall back to computing
            // size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
            size_links_per_element_ = sizeof(linklistsizeint) + maxM_ * sizeof(tableint)
                        + (store_edge_weights_ ? maxM_ * sizeof(dist_t) : 0);

            // size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
            size_links_level0_ = sizeof(linklistsizeint) + maxM0_ * sizeof(tableint)
                   + (store_edge_weights_ ? maxM0_ * sizeof(dist_t) : 0);
            //

            std::vector<std::mutex>(max_elements).swap(link_list_locks_);
            std::vector<std::mutex>(MAX_LABEL_OPERATION_LOCKS).swap(label_op_locks_);

            visited_list_pool_.reset(new VisitedListPool(1, max_elements));

            linkLists_ = (char **)malloc(sizeof(void *) * max_elements);
            if (linkLists_ == nullptr)
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
            element_levels_ = std::vector<int>(max_elements);
            revSize_ = 1.0 / mult_;
            ef_ = 10;
            for (size_t i = 0; i < cur_element_count; i++)
            {
                label_lookup_[getExternalLabel(i)] = i;
                unsigned int linkListSize;
                readBinaryPOD(input, linkListSize);
                if (linkListSize == 0)
                {
                    element_levels_[i] = 0;
                    linkLists_[i] = nullptr;
                }
                else
                {
                    element_levels_[i] = linkListSize / size_links_per_element_;
                    linkLists_[i] = (char *)malloc(linkListSize);
                    if (linkLists_[i] == nullptr)
                        throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
                    input.read(linkLists_[i], linkListSize);
                }
            }

            for (size_t i = 0; i < cur_element_count; i++)
            {
                if (isMarkedDeleted(i))
                {
                    num_deleted_ += 1;
                    if (allow_replace_deleted_)
                        deleted_elements.insert(i);
                }
            }

            input.close();

            return;
        }

        template <typename data_t>
        std::vector<data_t> getDataByLabel(labeltype label) const
        {
            // lock all operations with element by label
            std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));

            std::unique_lock<std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end() || isMarkedDeleted(search->second))
            {
                throw std::runtime_error("Label not found");
            }
            tableint internalId = search->second;
            lock_table.unlock();

            char *data_ptrv = getDataByInternalId(internalId);
            size_t dim = *((size_t *)dist_func_param_);
            std::vector<data_t> data;
            data_t *data_ptr = (data_t *)data_ptrv;
            for (size_t i = 0; i < dim; i++)
            {
                data.push_back(*data_ptr);
                data_ptr += 1;
            }
            return data;
        }

        /*
         * Marks an element with the given label deleted, does NOT really change the current graph.
         */
        void markDelete(labeltype label)
        {
            // lock all operations with element by label
            std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));

            std::unique_lock<std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end())
            {
                throw std::runtime_error("Label not found");
            }
            tableint internalId = search->second;
            lock_table.unlock();

            markDeletedInternal(internalId);
        }

        /*
         * Uses the last 16 bits of the memory for the linked list size to store the mark,
         * whereas maxM0_ has to be limited to the lower 16 bits, however, still large enough in almost all cases.
         */
        void markDeletedInternal(tableint internalId)
        {
            assert(internalId < cur_element_count);
            if (!isMarkedDeleted(internalId))
            {
                unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
                *ll_cur |= DELETE_MARK;
                num_deleted_ += 1;
                if (allow_replace_deleted_)
                {
                    std::unique_lock<std::mutex> lock_deleted_elements(deleted_elements_lock);
                    deleted_elements.insert(internalId);
                }
            }
            else
            {
                throw std::runtime_error("The requested to delete element is already deleted");
            }
        }

        /*
         * Removes the deleted mark of the node, does NOT really change the current graph.
         *
         * Note: the method is not safe to use when replacement of deleted elements is enabled,
         *  because elements marked as deleted can be completely removed by addPoint
         */
        void unmarkDelete(labeltype label)
        {
            // lock all operations with element by label
            std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));

            std::unique_lock<std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end())
            {
                throw std::runtime_error("Label not found");
            }
            tableint internalId = search->second;
            lock_table.unlock();

            unmarkDeletedInternal(internalId);
        }

        /*
         * Remove the deleted mark of the node.
         */
        void unmarkDeletedInternal(tableint internalId)
        {
            assert(internalId < cur_element_count);
            if (isMarkedDeleted(internalId))
            {
                unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
                *ll_cur &= ~DELETE_MARK;
                num_deleted_ -= 1;
                if (allow_replace_deleted_)
                {
                    std::unique_lock<std::mutex> lock_deleted_elements(deleted_elements_lock);
                    deleted_elements.erase(internalId);
                }
            }
            else
            {
                throw std::runtime_error("The requested to undelete element is not deleted");
            }
        }

        /*
         * Checks the first 16 bits of the memory to see if the element is marked deleted.
         */
        bool isMarkedDeleted(tableint internalId) const
        {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
            return *ll_cur & DELETE_MARK;
        }

        unsigned short int getListCount(linklistsizeint *ptr) const
        {
            return *((unsigned short int *)ptr);
        }

        void setListCount(linklistsizeint *ptr, unsigned short int size) const
        {
            *((unsigned short int *)(ptr)) = *((unsigned short int *)&size);
        }

        /*
         * Adds point. Updates the point if it is already in the index.
         * If replacement of deleted elements is enabled: replaces previously deleted point if any, updating it with new point
         */
        void addPoint(const void *data_point, labeltype label, bool replace_deleted = false)
        {
            if ((allow_replace_deleted_ == false) && (replace_deleted == true))
            {
                throw std::runtime_error("Replacement of deleted elements is disabled in constructor");
            }

            // lock all operations with element by label
            std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));
            if (!replace_deleted)
            {
                addPoint(data_point, label, -1);
                return;
            }
            // check if there is vacant place
            tableint internal_id_replaced;
            std::unique_lock<std::mutex> lock_deleted_elements(deleted_elements_lock);
            bool is_vacant_place = !deleted_elements.empty();
            if (is_vacant_place)
            {
                internal_id_replaced = *deleted_elements.begin();
                deleted_elements.erase(internal_id_replaced);
            }
            lock_deleted_elements.unlock();

            // if there is no vacant place then add or update point
            // else add point to vacant place
            if (!is_vacant_place)
            {
                addPoint(data_point, label, -1);
            }
            else
            {
                // we assume that there are no concurrent operations on deleted element
                labeltype label_replaced = getExternalLabel(internal_id_replaced);
                setExternalLabel(internal_id_replaced, label);

                std::unique_lock<std::mutex> lock_table(label_lookup_lock);
                label_lookup_.erase(label_replaced);
                label_lookup_[label] = internal_id_replaced;
                lock_table.unlock();

                unmarkDeletedInternal(internal_id_replaced);
                updatePoint(data_point, internal_id_replaced, 1.0);
            }
        }

        void updatePoint(const void *dataPoint, tableint internalId, float updateNeighborProbability)
        {
            // update the feature vector associated with existing point with new vector
            memcpy(getDataByInternalId(internalId), dataPoint, data_size_);

            int maxLevelCopy = maxlevel_;
            tableint entryPointCopy = enterpoint_node_;
            // If point to be updated is entry point and graph just contains single element then just return.
            if (entryPointCopy == internalId && cur_element_count == 1)
                return;

            int elemLevel = element_levels_[internalId];
            std::uniform_real_distribution<float> distribution(0.0, 1.0);
            for (int layer = 0; layer <= elemLevel; layer++)
            {
                std::unordered_set<tableint> sCand;
                std::unordered_set<tableint> sNeigh;
                std::vector<tableint> listOneHop = getConnectionsWithLock(internalId, layer);
                if (listOneHop.size() == 0)
                    continue;

                sCand.insert(internalId);

                for (auto &&elOneHop : listOneHop)
                {
                    sCand.insert(elOneHop);

                    if (distribution(update_probability_generator_) > updateNeighborProbability)
                        continue;

                    sNeigh.insert(elOneHop);

                    std::vector<tableint> listTwoHop = getConnectionsWithLock(elOneHop, layer);
                    for (auto &&elTwoHop : listTwoHop)
                    {
                        sCand.insert(elTwoHop);
                    }
                }

                for (auto &&neigh : sNeigh)
                {
                    // if (neigh == internalId)
                    //     continue;

                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                    size_t size = sCand.find(neigh) == sCand.end() ? sCand.size() : sCand.size() - 1; // sCand guaranteed to have size >= 1
                    size_t elementsToKeep = std::min(ef_construction_, size);
                    for (auto &&cand : sCand)
                    {
                        if (cand == neigh)
                            continue;

                        dist_t distance = fstdistfunc_(getDataByInternalId(neigh), getDataByInternalId(cand), dist_func_param_);
                        if (candidates.size() < elementsToKeep)
                        {
                            candidates.emplace(distance, cand);
                        }
                        else
                        {
                            if (distance < candidates.top().first)
                            {
                                candidates.pop();
                                candidates.emplace(distance, cand);
                            }
                        }
                    }

                    // Retrieve neighbours using heuristic and set connections.
                    getNeighborsByHeuristic2(candidates, layer == 0 ? maxM0_ : maxM_);

                    {
                        std::unique_lock<std::mutex> lock(link_list_locks_[neigh]);
                        linklistsizeint *ll_cur;
                        ll_cur = get_linklist_at_level(neigh, layer);
                        size_t candSize = candidates.size();
                        
setListCount(ll_cur, candSize);
                        tableint *data = (tableint *)(ll_cur + 1);
                        dist_t  *ws   = get_linklist_weights_ptr(ll_cur, layer);
                        for (size_t idx = 0; idx < candSize; idx++)
                        {
                            data[idx] = candidates.top().second;
                            if (ws) ws[idx] = candidates.top().first;
                            candidates.pop();
                        }}
                }
            }

            repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId, elemLevel, maxLevelCopy);
        }

        void repairConnectionsForUpdate(
            const void *dataPoint,
            tableint entryPointInternalId,
            tableint dataPointInternalId,
            int dataPointLevel,
            int maxLevel)
        {
            tableint currObj = entryPointInternalId;
            if (dataPointLevel < maxLevel)
            {
                dist_t curdist = fstdistfunc_(dataPoint, getDataByInternalId(currObj), dist_func_param_);
                for (int level = maxLevel; level > dataPointLevel; level--)
                {
                    bool changed = true;
                    while (changed)
                    {
                        changed = false;
                        unsigned int *data;
                        std::unique_lock<std::mutex> lock(link_list_locks_[currObj]);
                        data = get_linklist_at_level(currObj, level);
                        int size = getListCount(data);
                        tableint *datal = (tableint *)(data + 1);
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
#endif
                        for (int i = 0; i < size; i++)
                        {
#ifdef USE_SSE
                            _mm_prefetch(getDataByInternalId(*(datal + i + 1)), _MM_HINT_T0);
#endif
                            tableint cand = datal[i];
                            dist_t d = fstdistfunc_(dataPoint, getDataByInternalId(cand), dist_func_param_);
                            if (d < curdist)
                            {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            if (dataPointLevel > maxLevel)
                throw std::runtime_error("Level of item to be updated cannot be bigger than max level");

            for (int level = dataPointLevel; level >= 0; level--)
            {
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> topCandidates = searchBaseLayer(
                    currObj, dataPoint, level);

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> filteredTopCandidates;
                while (topCandidates.size() > 0)
                {
                    if (topCandidates.top().second != dataPointInternalId)
                        filteredTopCandidates.push(topCandidates.top());

                    topCandidates.pop();
                }

                // Since element_levels_ is being used to get `dataPointLevel`, there could be cases where `topCandidates` could just contains entry point itself.
                // To prevent self loops, the `topCandidates` is filtered and thus can be empty.
                if (filteredTopCandidates.size() > 0)
                {
                    bool epDeleted = isMarkedDeleted(entryPointInternalId);
                    if (epDeleted)
                    {
                        filteredTopCandidates.emplace(fstdistfunc_(dataPoint, getDataByInternalId(entryPointInternalId), dist_func_param_), entryPointInternalId);
                        if (filteredTopCandidates.size() > ef_construction_)
                            filteredTopCandidates.pop();
                    }

                    currObj = mutuallyConnectNewElement(dataPoint, dataPointInternalId, filteredTopCandidates, level, true);
                }
            }
        }

        std::vector<tableint> getConnectionsWithLock(tableint internalId, int level)
        {
            std::unique_lock<std::mutex> lock(link_list_locks_[internalId]);
            unsigned int *data = get_linklist_at_level(internalId, level);
            int size = getListCount(data);
            std::vector<tableint> result(size);
            tableint *ll = (tableint *)(data + 1);
            memcpy(result.data(), ll, size * sizeof(tableint));
            return result;
        }

        tableint addPoint(const void *data_point, labeltype label, int level)
        {
            tableint cur_c = 0;
            {
                // Checking if the element with the same label already exists
                // if so, updating it *instead* of creating a new element.
                std::unique_lock<std::mutex> lock_table(label_lookup_lock);
                auto search = label_lookup_.find(label);
                if (search != label_lookup_.end())
                {
                    tableint existingInternalId = search->second;
                    if (allow_replace_deleted_)
                    {
                        if (isMarkedDeleted(existingInternalId))
                        {
                            throw std::runtime_error("Can't use addPoint to update deleted elements if replacement of deleted elements is enabled.");
                        }
                    }
                    lock_table.unlock();

                    if (isMarkedDeleted(existingInternalId))
                    {
                        unmarkDeletedInternal(existingInternalId);
                    }
                    updatePoint(data_point, existingInternalId, 1.0);

                    return existingInternalId;
                }

                if (cur_element_count >= max_elements_)
                {
                    throw std::runtime_error("The number of elements exceeds the specified limit");
                }

                cur_c = cur_element_count;
                cur_element_count++;
                label_lookup_[label] = cur_c;
            }

            std::unique_lock<std::mutex> lock_el(link_list_locks_[cur_c]);
            int curlevel = getRandomLevel(mult_);
            if (level > 0)
                curlevel = level;

            element_levels_[cur_c] = curlevel;

            std::unique_lock<std::mutex> templock(global);
            int maxlevelcopy = maxlevel_;
            if (curlevel <= maxlevelcopy)
                templock.unlock();
            tableint currObj = enterpoint_node_;
            tableint enterpoint_copy = enterpoint_node_;

            memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);

            // Initialisation of the data and label
            memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
            memcpy(getDataByInternalId(cur_c), data_point, data_size_);

            if (curlevel)
            {
                linkLists_[cur_c] = (char *)malloc(size_links_per_element_ * curlevel + 1);
                if (linkLists_[cur_c] == nullptr)
                    throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
                memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
            }

            if ((signed)currObj != -1)
            {
                if (curlevel < maxlevelcopy)
                {
                    dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                    for (int level = maxlevelcopy; level > curlevel; level--)
                    {
                        bool changed = true;
                        while (changed)
                        {
                            changed = false;
                            unsigned int *data;
                            std::unique_lock<std::mutex> lock(link_list_locks_[currObj]);
                            data = get_linklist(currObj, level);
                            int size = getListCount(data);

                            tableint *datal = (tableint *)(data + 1);
                            for (int i = 0; i < size; i++)
                            {
                                tableint cand = datal[i];
                                if (cand < 0 || cand > max_elements_)
                                    throw std::runtime_error("cand error");
                                dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                                if (d < curdist)
                                {
                                    curdist = d;
                                    currObj = cand;
                                    changed = true;
                                }
                            }
                        }
                    }
                }

                bool epDeleted = isMarkedDeleted(enterpoint_copy);
                for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--)
                {
                    if (level > maxlevelcopy || level < 0) // possible?
                        throw std::runtime_error("Level error");

                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                        currObj, data_point, level);
                    if (epDeleted)
                    {
                        top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_), enterpoint_copy);
                        if (top_candidates.size() > ef_construction_)
                            top_candidates.pop();
                    }
                    currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
                }
            }
            else
            {
                // Do nothing for the first element
                enterpoint_node_ = 0;
                maxlevel_ = curlevel;
            }

            // Releasing lock for the maximum level
            if (curlevel > maxlevelcopy)
            {
                enterpoint_node_ = cur_c;
                maxlevel_ = curlevel;
            }
            return cur_c;
        }

        std::priority_queue<std::pair<dist_t, labeltype>>
        searchKnn(const void *query_data, size_t k, BaseFilterFunctor *isIdAllowed = nullptr) const
        {
            std::priority_queue<std::pair<dist_t, labeltype>> result;
            if (cur_element_count == 0)
                return result;

            TraversalLocalCounts traversal_local;
            TraversalLocalCounts* traversal_local_ptr = collect_traversal_counts_ ? &traversal_local : nullptr;
            if (traversal_local_ptr) {
                const size_t reserve_nodes = (size_t)std::max((size_t)64, std::max(ef_, k) * (size_t)4);
                const size_t reserve_edges = (size_t)std::max((size_t)256, std::max(ef_, k) * (size_t)32);
                traversal_local.upper_node_counts.reserve(reserve_nodes);
                traversal_local.upper_edge_counts.reserve(reserve_edges);
                traversal_local.l0_node_counts.reserve(reserve_nodes);
                traversal_local.l0_edge_counts.reserve(reserve_edges);
            }

            tableint currObj = enterpoint_node_;
            dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

            for (int level = maxlevel_; level > 0; level--)
            {
                bool changed = true;
                while (changed)
                {
                    changed = false;
                    unsigned int *data;

                    data = (unsigned int *)get_linklist(currObj, level);
                    int size = getListCount(data);
                    if (traversal_local_ptr) {
                        traversal_local_ptr->upper_node_counts[currObj]++;
                    }
                    metric_hops++;
                    metric_distance_computations += size;

                    tableint *datal = (tableint *)(data + 1);
                    for (int i = 0; i < size; i++)
                    {
                        tableint cand = datal[i];
                        if (traversal_local_ptr) {
                            traversal_local_ptr->upper_edge_counts[packTraversalEdge(currObj, cand)]++;
                        }
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");
                        dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                        if (d < curdist)
                        {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            bool bare_bone_search = !num_deleted_ && !isIdAllowed;
            if (bare_bone_search)
            {
                top_candidates = searchBaseLayerST<true>(
                    currObj, query_data, std::max(ef_, k), isIdAllowed, nullptr, traversal_local_ptr);
            }
            else
            {
                top_candidates = searchBaseLayerST<false>(
                    currObj, query_data, std::max(ef_, k), isIdAllowed, nullptr, traversal_local_ptr);
            }

            if (traversal_local_ptr) {
                std::lock_guard<std::mutex> g(traversal_counts_mutex_);
                const size_t ncur = cur_element_count.load();
                if (traversal_node_counts_.size() < ncur) {
                    traversal_node_counts_.resize(ncur, (uint64_t)0);
                }
                if (traversal_upper_node_counts_.size() < ncur) {
                    traversal_upper_node_counts_.resize(ncur, (uint64_t)0);
                }
                if (traversal_l0_node_counts_.size() < ncur) {
                    traversal_l0_node_counts_.resize(ncur, (uint64_t)0);
                }
                for (const auto& kv : traversal_local.upper_node_counts) {
                    if ((size_t)kv.first < traversal_node_counts_.size()) {
                        traversal_node_counts_[kv.first] += (uint64_t)kv.second;
                    }
                    if ((size_t)kv.first < traversal_upper_node_counts_.size()) {
                        traversal_upper_node_counts_[kv.first] += (uint64_t)kv.second;
                    }
                }
                for (const auto& kv : traversal_local.l0_node_counts) {
                    if ((size_t)kv.first < traversal_node_counts_.size()) {
                        traversal_node_counts_[kv.first] += (uint64_t)kv.second;
                    }
                    if ((size_t)kv.first < traversal_l0_node_counts_.size()) {
                        traversal_l0_node_counts_[kv.first] += (uint64_t)kv.second;
                    }
                }
                for (const auto& kv : traversal_local.upper_edge_counts) {
                    traversal_edge_counts_[kv.first] += (uint64_t)kv.second;
                    traversal_upper_edge_counts_[kv.first] += (uint64_t)kv.second;
                }
                for (const auto& kv : traversal_local.l0_edge_counts) {
                    traversal_edge_counts_[kv.first] += (uint64_t)kv.second;
                    traversal_l0_edge_counts_[kv.first] += (uint64_t)kv.second;
                }
                traversal_query_count_.fetch_add((uint64_t)1, std::memory_order_relaxed);
            }

            while (top_candidates.size() > k)
            {
                top_candidates.pop();
            }
            while (top_candidates.size() > 0)
            {
                std::pair<dist_t, tableint> rez = top_candidates.top();
                result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
                top_candidates.pop();
            }
            return result;
        }

        std::vector<std::pair<dist_t, labeltype>>
        searchStopConditionClosest(
            const void *query_data,
            BaseSearchStopCondition<dist_t> &stop_condition,
            BaseFilterFunctor *isIdAllowed = nullptr) const
        {
            std::vector<std::pair<dist_t, labeltype>> result;
            if (cur_element_count == 0)
                return result;

            tableint currObj = enterpoint_node_;
            dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

            for (int level = maxlevel_; level > 0; level--)
            {
                bool changed = true;
                while (changed)
                {
                    changed = false;
                    unsigned int *data;

                    data = (unsigned int *)get_linklist(currObj, level);
                    int size = getListCount(data);
                    metric_hops++;
                    metric_distance_computations += size;

                    tableint *datal = (tableint *)(data + 1);
                    for (int i = 0; i < size; i++)
                    {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");
                        dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                        if (d < curdist)
                        {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            top_candidates = searchBaseLayerST<false>(currObj, query_data, 0, isIdAllowed, &stop_condition);

            size_t sz = top_candidates.size();
            result.resize(sz);
            while (!top_candidates.empty())
            {
                result[--sz] = top_candidates.top();
                top_candidates.pop();
            }

            stop_condition.filter_results(result);

            return result;
        }

        void checkIntegrity()
        {
            int connections_checked = 0;
            std::vector<int> inbound_connections_num(cur_element_count, 0);
            for (int i = 0; i < cur_element_count; i++)
            {
                for (int l = 0; l <= element_levels_[i]; l++)
                {
                    linklistsizeint *ll_cur = get_linklist_at_level(i, l);
                    int size = getListCount(ll_cur);
                    tableint *data = (tableint *)(ll_cur + 1);
                    std::unordered_set<tableint> s;
                    for (int j = 0; j < size; j++)
                    {
                        assert(data[j] < cur_element_count);
                        assert(data[j] != i);
                        inbound_connections_num[data[j]]++;
                        s.insert(data[j]);
                        connections_checked++;
                    }
                    assert(s.size() == size);
                }
            }
            if (cur_element_count > 1)
            {
                int min1 = inbound_connections_num[0], max1 = inbound_connections_num[0];
                for (int i = 0; i < cur_element_count; i++)
                {
                    assert(inbound_connections_num[i] > 0);
                    min1 = std::min(inbound_connections_num[i], min1);
                    max1 = std::max(inbound_connections_num[i], max1);
                }
                std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
            }
            std::cout << "integrity ok, checked " << connections_checked << " connections\n";
        }

        // Algorithm 4 (document): Tree-Modeling Random Walk with Rejection Sampling.
        // Returns sampled external labels S from the certification ball around query_data.
        std::vector<labeltype> algorithm4_tree_modeling_rw(
            const void *query_data,
            size_t sample_size,
            size_t Lmax,
            dist_t rcert_metric,
            const std::vector<labeltype> &initial_neighbors_ext,
            size_t max_degree = 0,
            uint64_t seed = 100) const
        {
            try
            {
                if (sample_size == 0)
                {
                    std::cout << "[Algorithm4] sample_size=0 -> return empty S" << std::endl;
                    return {};
                }

                const size_t ncur = (size_t)cur_element_count.load();

                // Build V_ball = {v : dist(q,v) <= r_cert}.
                std::vector<uint8_t> in_ball(ncur, 0);
                std::vector<tableint> ball_nodes;
                ball_nodes.reserve(ncur);

                for (tableint v = 0; v < (tableint)ncur; v++)
                {
                    if (isMarkedDeleted(v))
                        continue;

                    dist_t d_raw = fstdistfunc_(query_data, getDataByInternalId(v), dist_func_param_);
                    if (d_raw < (dist_t)0)
                        d_raw = (dist_t)0;
                    dist_t d_metric = (dist_t)std::sqrt((double)d_raw); // assumes L2 raw is squared L2

                    if (d_metric <= rcert_metric)
                    {
                        in_ball[(size_t)v] = 1;
                        ball_nodes.push_back(v);
                    }
                }

                if (ball_nodes.empty())
                    throw std::runtime_error("certification ball is empty");
                if (sample_size > ball_nodes.size())
                    throw std::runtime_error("sample_size is larger than number of nodes in certification ball");

                // Build virtual query neighbors from Python-provided initial result nodes (external labels).
                std::vector<tableint> initial_neighbors_internal;
                initial_neighbors_internal.reserve(initial_neighbors_ext.size());
                {
                    std::unordered_set<tableint> seen;
                    seen.reserve(initial_neighbors_ext.size() * 2 + 1);

                    std::unique_lock<std::mutex> lock_table(label_lookup_lock);
                    for (auto lab : initial_neighbors_ext)
                    {
                        auto it = label_lookup_.find(lab);
                        if (it == label_lookup_.end())
                            continue;

                        tableint id = it->second;
                        if ((size_t)id >= ncur)
                            continue;
                        if (!in_ball[(size_t)id])
                            continue;
                        if (isMarkedDeleted(id))
                            continue;
                        if (seen.insert(id).second)
                            initial_neighbors_internal.push_back(id);
                    }
                }

                if (initial_neighbors_internal.empty())
                    throw std::runtime_error("initial_neighbors has no valid nodes inside certification ball");

                const size_t maxdeg = (max_degree == 0 ? M_ : max_degree);
                const long double C = std::pow((long double)(maxdeg + 1), -(long double)(Lmax + 1));

                std::mt19937_64 rng(seed);
                std::uniform_real_distribution<long double> uni01(0.0L, 1.0L);

                auto get_edge_weight_metric = [&](tableint src, tableint dst) -> dist_t
                {
                    linklistsizeint *ll = get_linklist0(src);
                    const size_t sz = getListCount(ll);
                    tableint *nbrs = (tableint *)(ll + 1);
                    const dist_t *ws = get_linklist_weights_ptr(ll, 0);

                    for (size_t j = 0; j < sz; j++)
                    {
                        if (nbrs[j] != dst)
                            continue;

                        if (ws)
                        {
                            dist_t w_raw = ws[j];
                            if (w_raw < (dist_t)0)
                                w_raw = (dist_t)0;
                            return to_metric(w_raw);
                        }

                        dist_t d_raw = fstdistfunc_(getDataByInternalId(src), getDataByInternalId(dst), dist_func_param_);
                        if (d_raw < (dist_t)0)
                            d_raw = (dist_t)0;
                        return to_metric(d_raw);
                    }

                    throw std::runtime_error("selected child edge not found in adjacency list");
                };

                std::unordered_set<tableint> S_internal;
                S_internal.reserve(sample_size * 2 + 1);

                size_t attempts = 0;
                size_t no_progress_trials = 0;
                size_t total_steps = 0;

                long double ps_sum = 0.0L;
                long double alpha_sum = 0.0L;
                long double ps_min = std::numeric_limits<long double>::infinity();
                long double ps_max = 0.0L;

                const size_t max_trials = std::max((size_t)10000, sample_size * (size_t)20000);
                const size_t max_no_progress_trials = std::max((size_t)20000, sample_size * (size_t)200);

                while (S_internal.size() < sample_size)
                {
                    const size_t s_before = S_internal.size();
                    attempts++;
                    // if (attempts > max_trials)
                    //     break;

                    bool at_root = true; // virtual root is query q
                    tableint u = (tableint)-1;
                    std::unordered_set<tableint> T; // tree vertices visited in this walk
                    T.reserve(Lmax + 2);

                    long double ps = 1.0L;
                    long double walk_dist_metric = 0.0L;

                    for (size_t step = 0; step < Lmax; step++)
                    {
                        std::vector<tableint> children;

                        if (at_root)
                        {
                            // Virtual query root connects to Python-provided initial neighbors.
                            children = initial_neighbors_internal;
                        }
                        else
                        {
                            linklistsizeint *ll = get_linklist0(u);
                            const size_t sz = getListCount(ll);
                            tableint *nbrs = (tableint *)(ll + 1);
                            children.reserve(sz);

                            for (size_t i = 0; i < sz; i++)
                            {
                                tableint v = nbrs[i];
                                if ((size_t)v >= ncur)
                                    continue;
                                if (!in_ball[(size_t)v])
                                    continue;
                                if (isMarkedDeleted(v))
                                    continue;
                                if (T.find(v) != T.end()){
                                    no_progress_trials++;
                                    continue; // keep acyclic
                                } else {
                                    no_progress_trials = 0;
                                }
                                children.push_back(v);
                            }
                        }

                        const size_t deg = children.size();
                        std::uniform_int_distribution<size_t> pick(0, deg); // deg index means self-loop
                        const size_t idx = pick(rng);

                        ps *= (1.0L / (long double)(deg + 1));

                        if (idx == deg)
                        {
                            break; // self-loop terminates walk
                        }

                        tableint v = children[idx];

                        dist_t edge_metric = (dist_t)0;
                        if (at_root)
                        {
                            // Root is virtual query node, so use query->v metric distance.
                            dist_t qv_raw = fstdistfunc_(query_data, getDataByInternalId(v), dist_func_param_);
                            if (qv_raw < (dist_t)0)
                                qv_raw = (dist_t)0;
                            edge_metric = to_metric(qv_raw);
                        }
                        else
                        {
                            edge_metric = get_edge_weight_metric(u, v);
                        }

                        const long double next_walk_dist_metric = walk_dist_metric + (long double)edge_metric;
                        if (next_walk_dist_metric > (long double)rcert_metric)
                        {
                            break;
                        }
                        walk_dist_metric = next_walk_dist_metric;

                        T.insert(v);
                        u = v;
                        at_root = false;
                        total_steps++;
                    }

                    // ** stats
                    if (ps < ps_min)
                        ps_min = ps;
                    if (ps > ps_max)
                        ps_max = ps;
                    ps_sum += ps;
                    //  

                    long double alpha = (ps > 0.0L ? C / ps : 1.0L);
                    if (alpha > 1.0L)
                        alpha = 1.0L;
                    if (alpha < 0.0L)
                        alpha = 0.0L;
                    alpha_sum += alpha; //stats

                    const long double x = uni01(rng);
                    if (x <= alpha)
                    {
                        if (!at_root)
                        {
                            S_internal.insert(u);
                        }
                    }

                    // if (S_internal.size() == s_before)
                    // {
                    //     no_progress_trials++;
                    //     if (no_progress_trials > max_no_progress_trials)
                    //         break;
                    // }
                    // else
                    // {
                    //     no_progress_trials = 0;
                    // }
                    if (no_progress_trials > max_no_progress_trials)
                        break;
                }

                std::vector<labeltype> S;
                S.reserve(S_internal.size());
                for (auto id : S_internal)
                    S.push_back(getExternalLabel(id));

                const long double avg_path_probability = (ps_sum / (long double)attempts);

                std::cout
                    << "[Algorithm4]"
                    << " sample_size=" << sample_size
                    << " returned=" << S.size()
                    << " Lmax=" << Lmax
                    << " MAXDegree=" << maxdeg
                    << " C=" << (double)C
                    << " rcert=" << (double)rcert_metric
                    // << " ball_nodes=" << ball_nodes.size()
                    << " attempts=" << attempts
                    << " total_steps=" << total_steps
                    << " avg_path_probability=" << (double)avg_path_probability
                    << " avg_ps=" << (double)avg_path_probability
                    << " min_ps=" << (double)ps_min
                    << " max_ps=" << (double)ps_max
                    << " avg_alpha=" << (double)(alpha_sum / (long double)attempts)
                    << std::endl;

                return S;
            }
            catch (const std::exception &e)
            {
                throw std::runtime_error(std::string("Algorithm4 ERROR: ") + e.what());
            }
            catch (...)
            {
                throw std::runtime_error("Algorithm4 ERROR: unknown exception");
            }
        }

// CHANGE 3
#ifndef EARLYSTOP_ASSUME_SQUARED_L2
#define EARLYSTOP_ASSUME_SQUARED_L2 1
#endif

    private:
        // Convert hnswlib raw distance to a metric distance (needed by triangle inequality proofs).
        inline dist_t to_metric(dist_t raw) const
        {
#if EARLYSTOP_ASSUME_SQUARED_L2
            if (raw <= (dist_t)0)
                return (dist_t)0;
            return (dist_t)std::sqrt((double)raw);
#else
            return raw;
#endif
        }

        // --------------------------
        // CHECK 1: Neighbor-of-result
        // --------------------------
        bool check1_result_neighbors(
            const void *data_point,
            const std::vector<std::pair<dist_t, tableint>> &RvecK, // size K, raw distance units
            const std::unordered_set<tableint> &Rset,
            dist_t threshold_raw // raw threshold: dK_raw - epsilon
        ) const
        {
            uint64_t t0 = now_ns();
            bool pass = true;

            const size_t ncur = (size_t)cur_element_count.load();

            for (auto &pr : RvecK)
            {
                tableint r = pr.second;
                if ((size_t)r >= ncur)
                    continue;

                linklistsizeint *llr = get_linklist0(r);
                size_t sz = getListCount(llr);
                tableint *nbrs = (tableint *)(llr + 1);

                for (size_t u = 0; u < sz; u++)
                {
                    tableint v = nbrs[u];
                    if ((size_t)v >= ncur)
                        continue;
                    if (Rset.find(v) != Rset.end())
                        continue;

                    dist_t dv = fstdistfunc_(data_point, getDataByInternalId(v), dist_func_param_);
                    early_stats_.dist_comps_check1++;
                    early_stats_.comparisons_check1++;

                    if (dv < threshold_raw)
                    {
                        pass = false;
                        early_stats_.check1_fail++;
                        break;
                    }
                }
                if (!pass)
                    break;
            }

            early_stats_.time_check1_ns += (now_ns() - t0);
            return pass;
        }

        // --------------------------
        // CHECK 2: Cluster bound
        // --------------------------
        bool check2_cluster_bound(
            const void *data_point,
            const std::vector<tableint> &last_visited,
            dist_t threshold_raw) const
        {
            const bool have_clusters =
                (early_.n_clusters > 0 &&
                 !cluster_labels_.empty() &&
                 !cluster_centers_.empty() &&
                 !cluster_radii_.empty());

            if (!have_clusters)
                return true;

            const size_t ncur = (size_t)cur_element_count.load();

            // mark which clusters were touched
            std::vector<char> touched((size_t)early_.n_clusters, 0);
            for (auto v : last_visited)
            {
                if ((size_t)v >= cluster_labels_.size())
                    continue;
                int c = cluster_labels_[v];
                if (0 <= c && c < early_.n_clusters)
                    touched[(size_t)c] = 1;
            }

            const size_t dim = get_dim();
            uint64_t t0 = now_ns();
            bool pass = true;

            for (int c = 0; c < early_.n_clusters; c++)
            {
                if (!touched[(size_t)c])
                    continue;
                const float *center = &cluster_centers_[(size_t)c * dim];

                // fstdistfunc_ expects "void* to point data". For float vectors, center is ok.
                dist_t dc = fstdistfunc_(data_point, (const void *)center, dist_func_param_);
                early_stats_.dist_comps_check2++;
                early_stats_.comparisons_check2++;

                dist_t dc_m = to_metric(dc);                     // sqrt if squared L2
                // dist_t threshold_m = to_metric(threshold_raw);   // note: threshold_raw can go negative
                dist_t threshold_m = to_metric(threshold_raw < (dist_t)0 ? (dist_t)0 : threshold_raw);

                // same raw units as threshold_raw
                if (dc_m - (dist_t)cluster_radii_[(size_t)c] < threshold_m)
                {
                    pass = false;
                    early_stats_.check2_fail++;
                    break;
                }
            }

            early_stats_.time_check2_ns += (now_ns() - t0);
            return pass;
        }

        // --------------------------
        // CHECK 3: Graph bound (delta_graph hops), triangle inequality in METRIC units
        // --------------------------
        bool check3_graph_bound(
            const void *data_point,
            const std::unordered_set<tableint> &Vset,
            std::unordered_map<tableint, dist_t> &dist_to_q_raw, // cache raw distances for visited nodes
            dist_t threshold_raw) const
        {
            const bool have_graph_bounds = !d_max_.empty();
            if (!have_graph_bounds || early_.delta_graph == 0)
                return true;

            const size_t ncur = (size_t)cur_element_count.load();
            const size_t N = Vset.size();
            if (N == 0)
                return true;

            uint64_t t_check3_start = now_ns();
            bool pass = true;

            struct NodeInfo
            {
                tableint src;
                uint16_t hop;
            };
            std::unordered_map<tableint, NodeInfo> info;
            info.reserve(N * 8);

            std::queue<tableint> q;
            std::queue<tableint> qsrc;
            std::queue<uint16_t> qhop;

            for (auto v : Vset)
            {
                info.emplace(v, NodeInfo{v, 0});
                q.push(v);
                qsrc.push(v);
                qhop.push(0);
            }

            while (!q.empty() && pass)
            {
                uint64_t t_iter_start = now_ns();

                tableint cur = q.front();
                q.pop();
                tableint src = qsrc.front();
                qsrc.pop();
                uint16_t hop = qhop.front();
                qhop.pop();

                early_stats_.bfs_nodes_check3++;

                if (hop < (uint16_t)early_.delta_graph)
                {
                    linklistsizeint *llc = get_linklist0(cur);
                    size_t sz = getListCount(llc);
                    tableint *nbrs = (tableint *)(llc + 1);

                    for (size_t i = 0; i < sz; i++)
                    {
                        tableint nxt = nbrs[i];
                        if ((size_t)nxt >= ncur)
                            continue;
                        if (info.find(nxt) != info.end())
                            continue;

                        info.emplace(nxt, NodeInfo{src, (uint16_t)(hop + 1)});

                        // Only certify nodes not already in V
                        if (Vset.find(nxt) == Vset.end())
                        {

                            if ((size_t)src >= d_max_.size())
                            {
                                pass = false;
                                early_stats_.check3_fail++;
                                break;
                            }

                            // raw dist(xq,src)
                            dist_t dsrc_raw;
                            auto it = dist_to_q_raw.find(src);
                            if (it != dist_to_q_raw.end())
                            {
                                dsrc_raw = it->second;
                            }
                            else
                            {
                                dsrc_raw = fstdistfunc_(data_point, getDataByInternalId(src), dist_func_param_);
                                early_stats_.dist_comps_check3++;
                                dist_to_q_raw[src] = dsrc_raw;
                            }

                            // Convert to metric for triangle inequality
                            const dist_t dsrc_m = to_metric(dsrc_raw);
                            const dist_t dmax_m = to_metric(d_max_[src]);

                            dist_t lb_m = dsrc_m - (dist_t)(hop + 1) * dmax_m;
                            if (lb_m < (dist_t)0)
                                lb_m = (dist_t)0;

                            // Compare in raw units: threshold_raw is raw.
                            // If raw is squared L2, then lb_m is L2; square it back.
                            dist_t lb_raw;

#if EARLYSTOP_ASSUME_SQUARED_L2
                            lb_raw = lb_m * lb_m;
#else
                            lb_raw = lb_m;
#endif

                            early_stats_.comparisons_check3++;
                            if (lb_raw < threshold_raw)
                            {
                                pass = false;
                                early_stats_.check3_fail++;
                                break;
                            }
                        }

                        if (!pass)
                            break;

                        q.push(nxt);
                        qsrc.push(src);
                        qhop.push((uint16_t)(hop + 1));
                    }
                }

                early_stats_.bfs_check3_ns += (now_ns() - t_iter_start);
            }

            early_stats_.time_check3_ns += (now_ns() - t_check3_start);
            return pass;
        }

        // --------------------------
        // CHECK 4: Landmark bound (delta_landmark), spec-faithful in METRIC units
        //   Z = nodes within deltaL hops from V but not in V
        //   For each z in Z, need max_{l in L} [ dist(xq,l) - hops(l,z)*dmax(l) ] >= dK - epsilon   (metric units)
        // --------------------------
        bool check4_landmark_bound(
            const void *data_point,
            const std::unordered_set<tableint> &Vset,
            const std::vector<tableint> &last_visited, // for selecting landmarks from visited order
            std::unordered_map<tableint, dist_t> &dist_to_q_raw,
            dist_t dK_raw // raw dK (Kth radius) to convert to metric threshold
        ) const
        {
            const bool have_graph_bounds = !d_max_.empty();
            if (!have_graph_bounds)
                return true;
            if (early_.delta_landmark == 0 || early_.landmark_k == 0)
                return true;

            uint64_t t_check4_start = now_ns();
            bool pass = true;

            const size_t ncur = (size_t)cur_element_count.load();
            const uint16_t deltaL = (uint16_t)early_.delta_landmark;

            // threshold in METRIC units
            const dist_t dK_m = to_metric(dK_raw);
            const dist_t threshold_m = dK_m - (dist_t)early_.epsilon;

            // (A) Build Z via multi-source BFS from all V nodes up to deltaL
            std::vector<int16_t> hop_from_V(ncur, (int16_t)-1);
            std::queue<tableint> qV;

            for (auto v : Vset)
            {
                if ((size_t)v >= ncur)
                    continue;
                if (hop_from_V[(size_t)v] != -1)
                    continue;
                hop_from_V[(size_t)v] = 0;
                qV.push(v);
            }

            std::vector<char> in_Z(ncur, 0);
            std::vector<tableint> Z;
            Z.reserve(1024);

            while (!qV.empty())
            {
                tableint cur = qV.front();
                qV.pop();
                const int16_t hop = hop_from_V[(size_t)cur];
                if (hop >= (int16_t)deltaL)
                    continue;

                linklistsizeint *llc = get_linklist0(cur);
                size_t sz = getListCount(llc);
                tableint *nbrs = (tableint *)(llc + 1);

                for (size_t i = 0; i < sz; i++)
                {
                    tableint nxt = nbrs[i];
                    if ((size_t)nxt >= ncur)
                        continue;
                    if (hop_from_V[(size_t)nxt] != -1)
                        continue;

                    hop_from_V[(size_t)nxt] = (int16_t)(hop + 1);
                    qV.push(nxt);

                    if (Vset.find(nxt) == Vset.end())
                    {
                        if (!in_Z[(size_t)nxt])
                        {
                            in_Z[(size_t)nxt] = 1;
                            Z.push_back(nxt);
                        }
                    }
                }
            }

            if (Z.empty())
            {
                early_stats_.time_check4_ns += (now_ns() - t_check4_start);
                return true; // vacuous
            }

            // (B) Select landmarks L subset of V, size = landmark_k, using visited order
            const size_t kL_target = std::min((size_t)early_.landmark_k, (size_t)Vset.size());
            std::vector<tableint> L;
            L.reserve(kL_target);

            {
                std::unordered_set<tableint> used;
                used.reserve(kL_target * 2);
                for (auto v : last_visited)
                {
                    if (L.size() >= kL_target)
                        break;
                    if ((size_t)v >= ncur)
                        continue;
                    if (used.find(v) != used.end())
                        continue;
                    if (Vset.find(v) == Vset.end())
                        continue;
                    used.insert(v);
                    L.push_back(v);
                }
            }
            if (L.size() < kL_target)
            {
                for (auto v : Vset)
                {
                    if (L.size() >= kL_target)
                        break;
                    if ((size_t)v >= ncur)
                        continue;
                    L.push_back(v);
                }
            }

            // (C) maxBound[z] init
            std::unordered_map<tableint, dist_t> maxBound;
            maxBound.reserve(Z.size() * 2);
            const dist_t NEG_INF = (dist_t)-1e30;
            for (auto z : Z)
                maxBound.emplace(z, NEG_INF);

            std::vector<int16_t> hop_from_l(ncur, (int16_t)-1);
            std::vector<tableint> touched;
            touched.reserve(4096);
            std::queue<tableint> qL;

            uint64_t t_bfs4_start = now_ns();

            for (size_t li = 0; li < L.size() && pass; li++)
            {
                const tableint l = L[li];

                // dist(xq,l) in METRIC units
                dist_t dl_raw;
                auto it = dist_to_q_raw.find(l);
                if (it != dist_to_q_raw.end())
                {
                    dl_raw = it->second;
                }
                else
                {
                    dl_raw = fstdistfunc_(data_point, getDataByInternalId(l), dist_func_param_);
                    early_stats_.dist_comps_check4++;
                    dist_to_q_raw[l] = dl_raw;
                }
                const dist_t dl_m = to_metric(dl_raw);

                // dmax(l) in METRIC units
                if ((size_t)l >= d_max_.size())
                {
                    pass = false;
                    early_stats_.check4_fail++;
                    break;
                }
                const dist_t dmax_m = to_metric(d_max_[l]);

                // reset BFS arrays
                while (!qL.empty())
                    qL.pop();
                for (auto node : touched)
                    hop_from_l[(size_t)node] = (int16_t)-1;
                touched.clear();

                hop_from_l[(size_t)l] = 0;
                touched.push_back(l);
                qL.push(l);

                while (!qL.empty() && pass)
                {
                    tableint cur = qL.front();
                    qL.pop();
                    const int16_t hop = hop_from_l[(size_t)cur];
                    early_stats_.bfs_nodes_check4++;

                    if (hop >= (int16_t)deltaL)
                        continue;

                    linklistsizeint *llc = get_linklist0(cur);
                    size_t sz = getListCount(llc);
                    tableint *nbrs = (tableint *)(llc + 1);

                    for (size_t i = 0; i < sz; i++)
                    {
                        tableint nxt = nbrs[i];
                        if ((size_t)nxt >= ncur)
                            continue;
                        if (hop_from_l[(size_t)nxt] != (int16_t)-1)
                            continue;

                        hop_from_l[(size_t)nxt] = (int16_t)(hop + 1);
                        touched.push_back(nxt);
                        qL.push(nxt);

                        if (in_Z[(size_t)nxt])
                        {
                            dist_t bound = dl_m - (dist_t)(hop + 1) * dmax_m;
                            auto itB = maxBound.find(nxt);
                            if (itB != maxBound.end())
                            {
                                if (bound > itB->second)
                                    itB->second = bound;
                            }
                        }
                    }
                }
            }

            early_stats_.bfs_check4_ns += (now_ns() - t_bfs4_start);

            // (D) certify
            if (pass)
            {
                for (auto z : Z)
                {
                    auto itB = maxBound.find(z);
                    early_stats_.comparisons_check4++;
                    if (itB == maxBound.end() || itB->second < threshold_m)
                    {
                        pass = false;
                        early_stats_.check4_fail++;
                        break;
                    }
                }
            }

            early_stats_.time_check4_ns += (now_ns() - t_check4_start);
            return pass;
        }

    public:
        // --------------------------
        // POST-HOC certification entrypoint
        // --------------------------
        bool certify_result(
            const void *data_point,
            size_t K,
            const std::priority_queue<std::pair<dist_t, tableint>,
                                      std::vector<std::pair<dist_t, tableint>>,
                                      CompareByFirst> &top_candidates,
            const std::vector<tableint> &last_visited,
            const std::vector<dist_t> &last_distances) const
        {
            // If certification disabled or not enough info, return false (or true if you prefer "no-cert == pass").
            if (!early_.enabled)
                return false;
            if (K == 0)
                return false;
            if (top_candidates.empty())
                return false;
            if (last_visited.empty() || last_distances.size() != last_visited.size())
                return false;

            const bool have_graph_bounds = !d_max_.empty();
            const bool have_clusters =
                (early_.n_clusters > 0 &&
                 !cluster_labels_.empty() &&
                 !cluster_centers_.empty() &&
                 !cluster_radii_.empty());

            // If we have neither clusters nor graph bounds, checks 2/3/4 can’t run; check1 still can.
            // Keep it consistent with your earlier gating: require (graph OR clusters) for meaningful certification.
            if (!have_graph_bounds && !have_clusters)
                return false;

            early_stats_.checks_executed++;

            // (1) Build V and dist(xq,·) cache from last_visited/distances
            std::unordered_set<tableint> Vset;
            Vset.reserve(last_visited.size() * 2);

            std::unordered_map<tableint, dist_t> dist_to_q_raw;
            dist_to_q_raw.reserve(last_visited.size() * 2);

            const size_t ncur = (size_t)cur_element_count.load();
            for (size_t i = 0; i < last_visited.size(); i++) {
                tableint v = last_visited[i];
                if ((size_t)v >= ncur) continue;
                Vset.insert(v);
                dist_to_q_raw[v] = last_distances[i];
            }

            // (2) Build R = best K items from top_candidates
            std::vector<std::pair<dist_t, tableint>> Rvec;
            Rvec.reserve(top_candidates.size());
            {
                auto tc_copy = top_candidates;
                while (!tc_copy.empty())
                {
                    Rvec.push_back(tc_copy.top());
                    tc_copy.pop();
                }
            }

            if (Rvec.size() < K)
                return false;

            std::nth_element(
                Rvec.begin(),
                Rvec.begin() + (K - 1),
                Rvec.end(),
                [](const auto &a, const auto &b)
                { return a.first < b.first; });
            Rvec.resize(K);

            // dK_raw = max distance among the K returned elements
            dist_t dK_raw = Rvec[0].first;
            for (size_t i = 1; i < Rvec.size(); i++)
                if (Rvec[i].first > dK_raw)
                    dK_raw = Rvec[i].first;

            const dist_t threshold_raw = dK_raw - (dist_t)early_.epsilon;

            std::unordered_set<tableint> Rset;
            Rset.reserve(K * 2);
            for (auto &p : Rvec)
                Rset.insert(p.second);

            // (3) Run checks (ALL post-hoc)
            bool pass1 = check1_result_neighbors(data_point, Rvec, Rset, threshold_raw);
            bool pass2 = true;
            bool pass3 = true;
            bool pass4 = true;

            if (pass1)
            {
                pass2 = check2_cluster_bound(data_point, last_visited, threshold_raw);
            }
            if (pass1 && pass2)
            {
                pass3 = check3_graph_bound(data_point, Vset, dist_to_q_raw, threshold_raw);
            }
            if (pass1 && pass2 && pass3)
            {
                pass4 = check4_landmark_bound(data_point, Vset, last_visited, dist_to_q_raw, dK_raw);
            }

            const bool certified = pass1 && pass2 && pass3 && pass4;

            // This counter is now “certified count”, not “early stop triggered”.
            if (certified)
                early_stats_.certified_count++;

            return certified;
        }
        // CHANGE 3

    };
} // namespace hnswlib
