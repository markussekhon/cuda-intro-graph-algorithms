/**
 * The file in which you will implement your DenseGraph GPU solutions!
 */

#include <cstddef> // std::size_t type

#include "cuda_common.h"
#include "data_types.h"

namespace csc485b {
namespace a2 {

/**
 * A DenseGraph is optimised for a graph in which the number of edges
 * is close to n(n-1). It is represented using an adjacency matrix.
 */
struct DenseGraph {
    std::size_t n;           /**< Number of nodes in the graph. */
    node_t *adjacencyMatrix; /** Pointer to an n x n adj. matrix */

    /** Returns number of cells in the adjacency matrix. */
    __device__ __host__ __forceinline__ std::size_t matrix_size() const {
        return n * n;
    }
};

namespace gpu {

/**
 * Constructs a DenseGraph from an input edge list of m edges.
 *
 * @pre The pointers in DenseGraph g have already been allocated.
 */
__global__ void
build_graph(DenseGraph g, edge_t const *edge_list, std::size_t m) {
    std::size_t th_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (th_id >= m) {
        return;
    }

    node_t u = edge_list[th_id].x;
    node_t v = edge_list[th_id].y;

    atomicExch(&g.adjacencyMatrix[u * g.n + v], 1);
    atomicExch(&g.adjacencyMatrix[v * g.n + u], 1);
}

/**
 * Repopulates the adjacency matrix as a new graph that represents
 * the two-hop neighbourhood of input graph input
 */
__global__ void two_hop_reachability(DenseGraph input, DenseGraph output) {
    std::size_t th_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (th_id >= input.n * input.n) {
        return;
    }

    std::size_t row = th_id / input.n;
    std::size_t col = th_id % input.n;

    node_t reachable = 0;

    // square the matrix
    for (std::size_t k = 0; k < input.n; k++) {

        bool rowCheck = input.adjacencyMatrix[row * input.n + k];
        bool colCheck = input.adjacencyMatrix[k * input.n + col];

        // compression avoided by exiting out if at least 1 path exists
        if (rowCheck && colCheck) {
            reachable = 1;
            break;
        }
    }

    // remove the diagonal to prevent loops
    output.adjacencyMatrix[th_id] = row == col ? 0 : reachable;
}

} // namespace gpu
} // namespace a2
} // namespace csc485b
