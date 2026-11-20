/**
 * Driver for the benchmark comparison. Generates random data,
 * runs the CPU baseline, and then runs your code.
 */

#include <algorithm>
#include <chrono>   // for timing
#include <iostream> // std::cout, std::endl
#include <vector>

#include "dense_graph.h"
#include "sparse_graph.h"

#include "data_generator.h"
#include "data_types.h"

/**
 * Runs timing tests on a CUDA graph implementation.
 * Consists of independently constructing the graph and then
 * modifying it to its two-hop neighbourhood.
 */
template <typename DeviceGraph>
void run(DeviceGraph input,
    DeviceGraph output,
    csc485b::a2::edge_t const *d_edges,
    std::size_t m) {

    cudaDeviceSynchronize();
    auto const build_start = std::chrono::high_resolution_clock::now();

    int threads_per_block = 256;

    std::size_t launches = std::max(m, input.n);

    int num_blocks = (launches + threads_per_block - 1) / threads_per_block;

    if (num_blocks > 0) {
        csc485b::a2::gpu::build_graph<<<num_blocks, threads_per_block>>>(
            input, d_edges, m);
    }

    cudaDeviceSynchronize();
    auto const reachability_start = std::chrono::high_resolution_clock::now();

    std::size_t cells = input.n * input.n;
    int reach_blocks = (cells + threads_per_block - 1) / threads_per_block;

    // neither does this!
    csc485b::a2::gpu::two_hop_reachability<<<reach_blocks, threads_per_block>>>(
        input, output);

    cudaDeviceSynchronize();
    auto const end = std::chrono::high_resolution_clock::now();

    std::cout << "Build time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(
                     reachability_start - build_start)
                     .count()
              << " us" << std::endl;

    std::cout << "Reachability time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(
                     end - reachability_start)
                     .count()
              << " us" << std::endl;
}

/**
 * Allocates space for a dense graph and then runs the test code on it.
 */
void run_dense(csc485b::a2::edge_t const *d_edges,
    std::size_t n,
    std::size_t m) {
    using namespace csc485b;

    // allocate device DenseGraph input and output
    a2::node_t *d_input;
    a2::node_t *d_output;

    std::size_t bytes = sizeof(a2::node_t) * n * n;

    cudaMalloc((void **)&d_input, bytes);
    cudaMalloc((void **)&d_output, bytes);

    cudaMemset(d_input, 0, bytes);
    cudaMemset(d_output, 0, bytes);

    a2::DenseGraph d_initial{n, d_input};
    a2::DenseGraph d_result{n, d_output};

    run(d_initial, d_result, d_edges, m);

    // check output?
    std::vector<a2::node_t> host_initial(d_initial.matrix_size());
    std::vector<a2::node_t> host_result(d_result.matrix_size());

    a2::DenseGraph initial{n, host_initial.data()};
    a2::DenseGraph result{n, host_result.data()};

    cudaMemcpy(initial.adjacencyMatrix,
        d_initial.adjacencyMatrix,
        sizeof(a2::node_t) * d_initial.matrix_size(),
        cudaMemcpyDeviceToHost);

    cudaMemcpy(result.adjacencyMatrix,
        d_result.adjacencyMatrix,
        sizeof(a2::node_t) * d_result.matrix_size(),
        cudaMemcpyDeviceToHost);

    std::cout << "Adjacency Matrix with m = " << m << " edges:\n";

    for (std::size_t i = 0; i < n; i++) {
        for (std::size_t j = 0; j < n; j++) {
            std::cout << host_initial[i * n + j] << "  ";
        }
        std::cout << "\n";
    }

    std::cout << "Two Hop Adjacency Matrix with " << n * n << " total cells:\n";

    for (std::size_t i = 0; i < n; i++) {
        for (std::size_t j = 0; j < n; j++) {
            std::cout << host_result[i * n + j] << "  ";
        }
        std::cout << "\n";
    }

    // clean up
    cudaFree(d_input);
    cudaFree(d_output);
}

/**
 * Allocates space for a sparse graph and then runs the test code on it.
 */
void run_sparse(csc485b::a2::edge_t const *d_edges,
    std::size_t n,
    std::size_t m) {
    using namespace csc485b;

    // allocate device SparseGraph
    a2::node_t *d_offsets, *d_neighbours;
    cudaMalloc((void **)&d_offsets, sizeof(a2::node_t) * n);
    cudaMalloc((void **)&d_neighbours, sizeof(a2::node_t) * m);
    a2::SparseGraph d_sg{n, m, d_offsets, d_neighbours};

    a2::SparseGraph d_result{n, 0, nullptr, nullptr};

    run(d_sg, d_result, d_edges, m);

    std::vector<a2::node_t> rowptr(n), index(m);

    cudaMemcpy(rowptr.data(),
        d_offsets,
        n * sizeof(a2::node_t),
        cudaMemcpyDeviceToHost);

    cudaMemcpy(index.data(),
        d_neighbours,
        m * sizeof(a2::node_t),
        cudaMemcpyDeviceToHost);

    std::cout << "\nCSR rowptr: ";
    for (auto x : rowptr) {
        std::cout << x << " ";
    }

    std::cout << "\nCSR index: ";
    for (auto x : index) {
        std::cout << x << " ";
    }

    std::cout << "\n";

    // clean up
    cudaFree(d_neighbours);
    cudaFree(d_offsets);
}

int main() {
    using namespace csc485b;

    // Create input
    std::size_t constexpr n = 32;
    std::size_t constexpr expected_degree = n >> 1;

    a2::edge_list_t const graph = a2::generate_graph(n, n * expected_degree);
    std::size_t const m = graph.size();

    // lazily echo out input graph
    for (auto const &e : graph) {
        std::cout << "(" << e.x << "," << e.y << ") ";
    }

    // allocate and memcpy input to device
    a2::edge_t *d_edges;
    cudaMalloc((void **)&d_edges, sizeof(a2::edge_t) * m);
    cudaMemcpyAsync(
        d_edges, graph.data(), sizeof(a2::edge_t) * m, cudaMemcpyHostToDevice);

    // run your code!
    run_dense(d_edges, n, m);
    run_sparse(d_edges, n, m);

    cudaFree(d_edges);
    return EXIT_SUCCESS;
}
