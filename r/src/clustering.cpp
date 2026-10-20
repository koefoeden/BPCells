// Copyright 2023 BPCells contributors
// 
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <future>
#include <limits>
#include <mutex>
#include <utility>

#define RCPP_NO_RTTI
#define RCPP_NO_SUGAR
#include <Rcpp.h>
#include <RcppEigen.h>

#include "R_interrupts.h"
using namespace Rcpp;

// Given a list of k nearest neighbors for each cell,
// return a sparse matrix of the number of shared nearest neighbors.
// param neghbor_indices: cells x k matrix, where entry (i,j) lists the
//    1-based index of the jth nearest neighbor of cell i
// param min_neighbors: if a pair (i, j) has less than min_neighbors shared, don't output it.
// Return: List of 3 integer vectors: i, j, and snn_count, marking triplets of
//    (cell 1, cell 2, SNN count). The values in i are guaranteed to be less or equal to the
//    corresponding values in j. Given in order sorted by (i, j)
// [[Rcpp::export]]
SEXP build_snn_graph_cpp(IntegerMatrix neighbor_indices, int min_neighbors) {
    int cells = neighbor_indices.rows();
    int k = neighbor_indices.cols();

    // For each cell, list the cells that mark it as a direct neighbor, in ascending order.
    // The list for cell c is reverse_neighbors[reverse_start[c]:reverse_start[c+1]]
    std::vector<size_t> reverse_start(cells + 1, 0);
    for (int i = 0; i < cells; i++) {
        for (int j = 0; j < k; j++) {
            reverse_start[neighbor_indices(i, j)]++;
        }
    }
    for (int c = 0; c < cells; c++) {
        reverse_start[c + 1] += reverse_start[c];
    }
    std::vector<int> reverse_neighbors(reverse_start[cells]);
    std::vector<size_t> next_slot(reverse_start.begin(), reverse_start.end() - 1);
    for (int i = 0; i < cells; i++) {
        if (i % 1024 == 0) Rcpp::checkUserInterrupt();
        for (int j = 0; j < k; j++) {
            reverse_neighbors[next_slot[neighbor_indices(i, j) - 1]++] = i;
        }
    }

    // For each cell, count up the shared neighbors with lower-index cells (so we get lower-triangle
    // matrix out). shared[c] counts the neighbors of cell i that cell c also lists, and
    // touched holds the cells with a non-zero count.
    std::vector<int> shared(cells, 0);
    std::vector<int> touched;
    std::vector<int> c1, c2, snn_count;

    for (int i = 0; i < cells; i++) {
        if (i % 1024 == 0) Rcpp::checkUserInterrupt();
        for (int j = 0; j < k; j++) {
            int neighbor = neighbor_indices(i, j) - 1;
            for (size_t r = reverse_start[neighbor]; r < reverse_start[neighbor + 1]; r++) {
                int c = reverse_neighbors[r];
                if (c > i) break;
                if (shared[c]++ == 0) touched.push_back(c);
            }
        }
        // For the cell itself, shared[i] sums the squared multiplicities of its neighbors.
        // Count k plus one per pair of repeated neighbor entries, which is k when no neighbor repeats.
        shared[i] = (shared[i] + k) / 2;

        // Keep the cells with enough shared neighbors, then sort just those
        size_t kept = 0;
        for (int c : touched) {
            if (shared[c] >= min_neighbors) touched[kept++] = c;
            else shared[c] = 0;
        }
        std::sort(touched.begin(), touched.begin() + kept);
        for (size_t t = 0; t < kept; t++) {
            c1.push_back(i);
            c2.push_back(touched[t]);
            snn_count.push_back(shared[touched[t]]);
            shared[touched[t]] = 0;
        }
        touched.clear();
    }

    return List::create(Named("i") = c1, Named("j") = c2, Named("snn_count") = snn_count);
}

// Given the distances to k nearest neighbors for each cell, return
// an undirected similarity graph using the method of UMAP's `fuzzy_simplical_set`.
// See Algorithm 2 LocalFuzzySimplicialSet in section 4.1 of McInnes et al
// https://arxiv.org/pdf/1802.03426.pdf This is the approach used by Scanpy by default, and should
// result in much less memory usage than constructing an SNN graph. This bakes in the default
// parameters of local_connectivity=1 and set_op_mix_ratio=1 used in:
// https://umap-learn.readthedocs.io/en/latest/_modules/umap/umap_.html#fuzzy_simplicial_set
// Input:
//  - dists: matrix (n_cells x n_neighbors). Sorted distance to each neighbor, where dists[i,j] is
//    the distance from cell i to its jth-nearest neighbor
//  - idx: matrix (n_cells x n_neighbors). Index of each neighbor. idx[i,j] is the 1-based cell index of
//    the jth-nearest neighbor of cell i
//  - threads: number of threads to use during parallel execution segments
//  - umap_learn_sigma_sum: if TRUE, calculate the sum in the sigma search over neighbors index 1-N unconditionally like
//    the umap-learn python package. Otherwise calculate the sum over all non-self neighbors (either 0-N or 1-N depending on if
//    the first neighbor is actually self)
// Output: Undirected, weighted graph edge list after the UMAP normalization
//    List of 3 integer vectors: i, j, and weight, marking triplets of
//    (cell 1, cell 2, weight). The values in i are guaranteed to be less or equal to the
//    corresponding values in j. Given in order sorted by (i, j)
// [[Rcpp::export]]
SEXP build_umap_graph_cpp(const NumericMatrix dists, const IntegerMatrix idx, size_t threads = 0, bool umap_learn_sigma_sum = false) {
    size_t n_neighbors = dists.cols();
    size_t n_cells = dists.rows();

    if (dists.rows() != idx.rows() || dists.cols() != idx.cols()) {
        throw std::invalid_argument("build_umap_graph_cpp: dists and idx must have same dimensions.");
    }

    const double target_sum = std::log2(n_neighbors);
    const int sigma_iter = 64; // Iterations to for binary search on sigma

    // Keep lock-protected adjacency lists where we maintain a guarantee that
    // all elements in neighbor_idx[i] are >= i
    std::vector<std::vector<std::pair<int, double>>> neighbors(n_cells);
    std::vector<std::mutex> neighbor_locks(n_cells);

    // Calculate normalized distances and add to the adjacency list in neighbors
    auto calc_distance_transform = [&](size_t i_start, size_t i_end, std::atomic<bool> *interrupt) {
        std::vector<double> local_dists(n_neighbors);
        for (size_t i = i_start; i < i_end; i++) {
            if ((i - i_start) % 128 == 0 && *interrupt) break;
            double rho = std::numeric_limits<double>::infinity();

            for (size_t j = 0; j < n_neighbors; j++) {
                local_dists[j] = dists(i, j);
                if (idx(i, j)-1 != static_cast<int64_t>(i) && local_dists[j] > 0) rho = std::min(rho, local_dists[j]);
            }

            double sigma_lo = 0.0;
            double sigma_hi = std::numeric_limits<double>::infinity();
            double sigma = 1.0;

            // Binary search to find a sigma value that will cause the transformed differences to
            // add up to 1
            size_t j_start = 0;
            if (umap_learn_sigma_sum || idx(i,0)-1 == static_cast<int64_t>(i)) j_start += 1;
            for (size_t k = 0; k < sigma_iter; k++) {
                double sum = 0;
                // Putting in a 1 here to match umap-learn implementation...
                for (size_t j = j_start; j < n_neighbors; j++) {
                    double d = std::max(0.0, local_dists[j] - rho);
                    sum += std::exp(-d / sigma);
                }
                // Break for early convergence
                if (std::abs(sum - target_sum) < 1e-5) break;

                if (sum > target_sum) sigma_hi = sigma;
                else sigma_lo = sigma;

                if (std::isinf(sigma_hi)) sigma = sigma * 2;
                else sigma = (sigma_lo + sigma_hi) / 2.0;
            }
            // Limit how small sigma can get
            double mean_dist = 0.0;
            for (const auto &d: local_dists) {
                mean_dist += d / local_dists.size();
            }
            sigma = std::max(sigma, 1e-3 * mean_dist);

            // Add transformed distances into our global lists
            for (size_t j = 0; j < n_neighbors; j++) {
                size_t j_idx = idx(i, j) - 1;
                size_t min_idx = std::min(i, j_idx);
                size_t max_idx = std::max(i, j_idx);
                std::lock_guard<std::mutex> guard(neighbor_locks[min_idx]);
                double d = std::exp(-std::max(0.0, local_dists[j] - rho) / sigma);
                if (d > 0 && min_idx != max_idx) neighbors[min_idx].push_back({max_idx, d});
            }
        }
    };

    // Calculate adjacency lists in parallel
    run_parallel_with_R_interrupt_check(calc_distance_transform, n_cells, threads);
    neighbor_locks.clear();

    // Combine any bi-directional distances as a + b - a*b
    auto consolidate_neighbors = [&](size_t i_start, size_t i_end, std::atomic<bool> *interrupt) {
        for (size_t i = i_start; i < i_end; i++) {
            if ((i - i_start) % 128 == 0 && *interrupt) break;
            std::sort(neighbors[i].begin(), neighbors[i].end());
            size_t out = 0;
            
            for (size_t j = 0; j < neighbors[i].size(); j++) {
                if (j + 1 < neighbors[i].size() &&
                    neighbors[i][j].first == neighbors[i][j + 1].first) {
                    // If we have two edges, combine and skip outputting the second
                    double a = neighbors[i][j].second;
                    double b = neighbors[i][j + 1].second;
                    neighbors[i][j+1].second = a + b - a * b;
                    j += 1;
                }
                neighbors[i][out++] = neighbors[i][j];
            }
            neighbors[i].resize(out);
        }
    };

    run_parallel_with_R_interrupt_check(consolidate_neighbors, n_cells, threads);

    size_t total_neighbors = 0;
    for (const auto &x : neighbors) {
        total_neighbors += x.size();
    }
    // Construct output adjacency lists
    Rcpp::IntegerVector c1(total_neighbors), c2(total_neighbors);
    Rcpp::NumericVector similarity(total_neighbors);
    size_t out = 0;
    for (size_t i = 0; i < n_cells; i++) {
        for (const auto &n : neighbors[i]) {
            c1[out] = n.first;
            c2[out] = i;
            similarity[out] = n.second;
            out += 1;
        }
    }
    return List::create(Named("i") = c1, Named("j") = c2, Named("weight") = similarity);
}