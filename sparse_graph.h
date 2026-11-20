/**
 * The file in which you will implement your SparseGraph GPU solutions!
 */

#include <cstddef> // std::size_t type

#include "cuda_common.h"
#include "data_types.h"

namespace csc485b {
namespace a2 {

/**
 * A SparseGraph is optimised for a graph in which the number of edges
 * is close to cn, for a small constanct c. It is represented in CSR format.
 */
struct SparseGraph {
    std::size_t n;  /**< Number of nodes in the graph. */
    std::size_t m;  /**< Number of edges in the graph. */
    node_t *rowptr; /** Pointer to an n=|V| offset array */
    node_t *index;  /** Pointer to an m=|E| array of edge destinations */
};

namespace gpu {

/**
 * Constructs a SparseGraph from an input edge list of m edges.
 *
 * @pre The pointers in SparseGraph g have already been allocated.
 */
__global__ void
build_graph(SparseGraph g, edge_t const *edge_list, std::size_t m) {
    std::size_t th_id = blockIdx.x * blockDim.x + threadIdx.x;

    // one thread per node
    if (th_id >= g.n) {
        return;
    }

    // Count how many edges belond to previous nodes
    std::size_t offset = 0;

    for (std::size_t i = 0; i < m; i++) {
        offset += edge_list[i].x < th_id ? 1 : 0;
    }

    g.rowptr[th_id] = offset;

    // Store this node's neighbours in the index array
    std::size_t position = offset;

    for (std::size_t i = 0; i < m; i++) {
        if (edge_list[i].x == th_id) {
            g.index[position] = edge_list[i].y;
            position++;
        }
    }
}

/**
 * Repopulates the adjacency lists as a new graph that represents
 * the two-hop neighbourhood of input graph input
 */
__global__ void two_hop_reachability(SparseGraph input, SparseGraph output) {
    // IMPLEMENT ME!
    // algorithm unknown
    return;
}

} // namespace gpu
} // namespace a2
} // namespace csc485b
