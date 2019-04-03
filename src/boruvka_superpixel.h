// Boruvka Superpixel: split image into superpixels
// full rewrite after:  Wei, Yang, Gong, Ahuja, Yang: Superpixel Hierarchy,
//         IEEE Trans Image Proc, 2018
// based on Boruvka's minimum spanning tree algorithm

#ifndef BORUVKA_SUPERPIXEL_H_INCLUDED
#define BORUVKA_SUPERPIXEL_H_INCLUDED

#include <cstdint>
#include <algorithm>
#include <cassert>
#include <cmath>

#define CHECK_VERT(v) {                 \
    assert((v) >= 0);                   \
    assert((v) < n_vert_);              \
}

#define CHECK_EDGE(edge) {              \
    CHECK_VERT(edge->v0);               \
    CHECK_VERT(edge->v1);               \
}

#define PENALTY     1e8

struct Edge {
    int v0;                             // vertex 0
    int v1;                             // vertex 1
    struct Edge* v0_next;               // next edge on v0
    struct Edge* v1_next;               // next edge on v1
    float border_strength;              // at least 1
};


class BoruvkaSuperpixel {
public:
    BoruvkaSuperpixel()
        : dim_(0), avg_data_(nullptr), edgeless_(nullptr)
        {}
    ~BoruvkaSuperpixel()
        { clean(); }

    // data access
    int dim() const
        { return dim_; }
    int * shape()   // treat it as const int *; written non-const for cython
        { return shape_; }
    int n_vert() const      // number of vertices
        { return n_vert_; }
    int min_n_supix() const // minimum number of superpixels
        { return n_vert_ - n_mst_; }
    int n_edgeless() const  // number of edgeless vertices
        { return n_edgeless_; }

    // TODO parameter: connectivity of pixel graph (now: only nearest neighb)
    template <typename T>
        void build_2d(int shape_x, int shape_y,
            int n_feature_channels, T const *feature, T const *border_strength,
            int8_t const *edgeless=nullptr, int32_t const *seed=nullptr);
    template <typename T>
        void build_3d(int shape_x, int shape_y, int shape_z,
            int n_feature_channels, T const *feature, T const *border_strength);
    template <typename T>
        void build_3d_of(int shape_x, int shape_y, int shape_z,
            int n_feature_channels, T const *feature, T const *border_strength,
            int16_t const *optical_flow, double ofedge_prefactor);
    template <typename T>
        void build_3d_of2(int shape_x, int shape_y, int shape_z,
            int n_feature_channels, T const *feature, T const *border_strength,
            int16_t const *optical_flow, int16_t const *optical_flow_reverse,
            double ofedge_prefactor, int of_tolerance_sq,
            double of_rel_tolerance=0.);

    // calculate labels or average
    // return array owned by class, contents changes on subsequent call
    int32_t * label(int n_supix);
    template <typename T>
        void label_o(int n_supix, T * label);
    void seed_id(int32_t *result);
    template <typename T>
        T * average(int n_supix, int n_channels, T const *data);

private:
    // global data structures
    int dim_;               // dim of pixel array; features are extra
    int shape_[4];          // shape of pixel array
    int stride_[4];         // stride of pixel array
    int n_vert_;            // number of graph vertices (ie. pixels)
    int n_tree_;            // number of trees
    int n_mst_;             // number of mst edges
    int n_edgeless_;        // number of edgeless vertices
    int32_t *label_;        // per vertex: tree label [0 .. n_tree_-1]
    int32_t *seed_;         // per vertex: seed cluster index or negative
    int *parent_;           // per vertex: parent or root vertex within tree
    int *mst_v0_;           // v0 vertex of MST edge, final size=n_vert_-1
    int *mst_v1_;           // v1 vertex of MST edge, final size=n_vert_-1
    void *avg_data_;        // output of average()
    // build-only data structures
    int n_feat_ch_;         // number of feature channels
    int n_edge_;            // number of inter-tree edges in edge_
    int8_t const *edgeless_;// nullptr or per vertex: bool no edges on vertex
    float *feature_;        // n_feat_ch_ channels per vertex: features
    float *border_strength_;// per vertex: border strength
    Edge *edge_store_;      // array of all edges
    Edge **edge_;           // inter-tree edges
    Edge **edges_head_;     // per vertex: head of Edge linked list

    // inline
    void add_graph_edge(int v0, int stride, double border_prefactor=1.);
    void add_edges_2d(int dx0, int dx1, int dy0, int dy1, int stride);
    void add_edges_3d(int dx0, int dx1, int dy0, int dy1, int dz0, int dz1,
            int stride);
    double calc_dist(Edge *edge, int boruvka_iter);
    int find_root(int v);
    Edge *find_edge(Edge *edge, int v0, int v1);
    // in cpp
    template <typename T>
        void build_common(int n_edge_store, T const *feature,
                T const *border_strength, int32_t const *seed=nullptr);
    void build_hierarchy(); // final part of build
    void clean();
};

// ****************************************************************************

inline void
BoruvkaSuperpixel::add_graph_edge(int v0, int stride, double border_prefactor)
{
    int v1 = v0 + stride;
    if (edgeless_ and (edgeless_[v0] or edgeless_[v1])) {
        return;
    }
    Edge *edge = edge_store_ + n_edge_;    // pointer to next available edge
    edge_[n_edge_] = edge;
    n_edge_++;

    edge->v0 = v0;
    edge->v1 = v1;
    // insert at head of edge-list for both trees
    edge->v0_next = edges_head_[v0];
    edge->v1_next = edges_head_[v1];
    edges_head_[v0] = edge;
    edges_head_[v1] = edge;
    // calc border_strength:
    // min of border_strength at the two vertices, but at least 1
    edge->border_strength = border_prefactor * std::max(
            std::min(border_strength_[v0], border_strength_[v1]), (float)1.);
    CHECK_EDGE(edge)
}

#define IDX2(x, y)      ((x) * stride_[0] + (y) * stride_[1])
#define IDX3(x, y, z)   ((x) * stride_[0] + (y) * stride_[1] + (z) * stride_[2])

inline void
BoruvkaSuperpixel::add_edges_2d(int dx0, int dx1, int dy0, int dy1, int stride)
{
    // nonstandard indent for cleaner look
    for (int x = dx0; x < shape_[0] + dx1; x++) {
    for (int y = dy0; y < shape_[1] + dy1; y++) {
        add_graph_edge(IDX2(x, y), stride);
    } }
}

inline void
BoruvkaSuperpixel::add_edges_3d(int dx0, int dx1, int dy0, int dy1,
        int dz0, int dz1, int stride)
{
    // nonstandard indent for cleaner look
    for (int x = dx0; x < shape_[0] + dx1; x++) {
    for (int y = dy0; y < shape_[1] + dy1; y++) {
    for (int z = dz0; z < shape_[2] + dz1; z++) {
        add_graph_edge(IDX3(x, y, z), stride);
    } } }
}


inline double
BoruvkaSuperpixel::calc_dist(Edge *edge, int boruvka_iter)
{
    double dist = 0;
    if (seed_) {
        int seed0 = seed_[edge->v0];
        int seed1 = seed_[edge->v1];
        if (seed0 >= 0 and seed0 == seed1) {
            // vertices belong to same seed
            // edges with negative weight connect early
            return -(edge->v0 + abs(edge->v1 - edge->v0));
        }
        if (seed0 >= 0 and seed1 >= 0 and seed0 != seed1) {
            // vertices/trees belong to different seed
            // increase weight to postpone connection as much as possible
            dist = PENALTY;
        }
    }
    for (int c = 0; c < n_feat_ch_; c++) {     // all feature channels
        dist += fabsf(feature_[n_feat_ch_ * edge->v0 + c]
                - feature_[n_feat_ch_ * edge->v1+ c]);
    }
    // unconditionally multiply with border_strength
    return dist * edge->border_strength;
}


inline int
BoruvkaSuperpixel::find_root(int v)
{
    // find root by parent_[]
    // use path compression: intermediate nodes' parent are set to ultimate root
    int par = parent_[v];
    if (par != v) {
        parent_[v] = find_root(par);
    }
    return parent_[v];
}


inline Edge *
BoruvkaSuperpixel::find_edge(Edge *edge, int v0, int v1)
{
    // find an edge connecting tree-roots v0 and v1, following v0's linked list
    while (edge) {
        // edge->v0 == v0 or edge->v1 == v0
        if (edge->v0 == v0) {
            if (edge->v1 == v1) {
                return edge;
            }
            edge = edge->v0_next;
        } else {
            assert(edge->v1 == v0);
            if (edge->v0 == v1) {
                return edge;
            }
            edge = edge->v1_next;
        }
    }
    return nullptr;
}

#endif // BORUVKA_SUPERPIXEL_H_INCLUDED
// vim: set sw=4 cin cino=\:0 sts=4 expandtab :
