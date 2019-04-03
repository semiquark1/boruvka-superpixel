// Boruvka Superpixel: split image into superpixels
// full rewrite after:  Wei, Yang, Gong, Ahuja, Yang: Superpixel Hierarchy,
//         IEEE Trans Image Proc, 2018
// based on Boruvka's minimum spanning tree algorithm

#include <cstdlib>
#include <cstring>
#include <climits>
#include <cassert>
#include <float.h>

#include "boruvka_superpixel.h"


template <typename T> void
BoruvkaSuperpixel::build_2d(int shape_x, int shape_y,
        int n_feature_channels, T const *feature, T const *border_strength,
        int8_t const *edgeless, int32_t const *seed)
{
    // TODO argument error check
    clean();

    // init global data structures
    dim_ = 2;
    shape_[0] = shape_x;
    shape_[1] = shape_y;
    stride_[0] = shape_[1];
    stride_[1] = 1;
    n_vert_ = shape_x * shape_y;

    // init build-only data structures
    n_feat_ch_ = n_feature_channels;
    edgeless_ = edgeless;
    int n_edge_store =  // number of graph edges
        (shape_[0] - 1) * shape_[1] + (shape_[1] - 1) * shape_[0];
    build_common(n_edge_store, feature, border_strength, seed);

    // build image graph: add edges
    add_edges_2d( 0, -1,  0,  0, stride_[0]);   // x+
    add_edges_2d( 0,  0,  0, -1, stride_[1]);   // y+
    assert(n_edge_ == n_edge_store);

    build_hierarchy();
}


template <typename T> void
BoruvkaSuperpixel::build_3d(int shape_x, int shape_y, int shape_z,
        int n_feature_channels, T const *feature, T const *border_strength)
{
    // TODO argument error check
    clean();

    // init global data structures
    dim_ = 3;
    shape_[0] = shape_x;
    shape_[1] = shape_y;
    shape_[2] = shape_z;
    stride_[0] = shape_[1] * shape_[2];
    stride_[1] = shape_[2];
    stride_[2] = 1;
    n_vert_ = shape_x * shape_y * shape_z;

    // init build-only data structures
    n_feat_ch_ = n_feature_channels;
    int n_edge_store =  // number of graph edges
        (shape_[0] - 1) * shape_[1] * shape_[2] +
        (shape_[1] - 1) * shape_[2] * shape_[0] +
        (shape_[2] - 1) * shape_[0] * shape_[1];
    build_common(n_edge_store, feature, border_strength);

    // build image graph: add edges
    add_edges_3d( 0, -1,  0,  0,  0,  0, stride_[0]);   // x+
    add_edges_3d( 0,  0,  0, -1,  0,  0, stride_[1]);   // y+
    add_edges_3d( 0,  0,  0,  0,  0, -1, stride_[2]);   // z+
    assert(n_edge_ == n_edge_store);

    build_hierarchy();
}

template <typename T> void
BoruvkaSuperpixel::build_3d_of(int shape_x, int shape_y, int shape_z,
        int n_feature_channels, T const *feature, T const *border_strength,
        int16_t const *optical_flow, double ofedge_prefactor)
{
    // TODO argument error check
    clean();

    // init global data structures
    dim_ = 3;
    shape_[0] = shape_x;
    shape_[1] = shape_y;
    shape_[2] = shape_z;
    stride_[0] = shape_[1] * shape_[2];
    stride_[1] = shape_[2];
    stride_[2] = 1;
    n_vert_ = shape_x * shape_y * shape_z;

    // init build-only data structures
    n_feat_ch_ = n_feature_channels;
    int n_edge_store =  // number of graph edges
        (shape_[0] - 1) * shape_[1] * shape_[2] +
        (shape_[1] - 1) * shape_[2] * shape_[0] +
        // and at most 1 edge in (z+ + o.f.) direction
        (shape_[2] - 1) * shape_[0] * shape_[1];
    build_common(n_edge_store, feature, border_strength);

    // build image graph: add edges
    add_edges_3d( 0, -1,  0,  0,  0,  0, stride_[0]);   // x+
    add_edges_3d( 0,  0,  0, -1,  0,  0, stride_[1]);   // y+
    //add_edges_3d( 0,  0,  0,  0,  0, -1, stride_[2]);   // z+
    // add optical flow edges
    for (int x = 0; x < shape_[0]; x++) {
    for (int y = 0; y < shape_[1]; y++) {
    for (int z = 0; z < shape_[2] - 1; z++) {
        int dx = optical_flow[2 * IDX3(x, y, z)    ];
        int dy = optical_flow[2 * IDX3(x, y, z) + 1];
        int x1 = x + dx;
        int y1 = y + dy;
        if (0 <= x1 and x1 < shape_[0] and 0 <= y1 and y1 < shape_[1]) {
            int stride = IDX3(dx, dy, 1);
            add_graph_edge(IDX3(x, y, z), stride, ofedge_prefactor);
        }
    } } }
    assert(n_edge_ <= n_edge_store);    // less if o.f. points outside

    build_hierarchy();
}

template <typename T> void
BoruvkaSuperpixel::build_3d_of2(int shape_x, int shape_y, int shape_z,
        int n_feature_channels, T const *feature, T const *border_strength,
        int16_t const *optical_flow, int16_t const *optical_flow_reverse,
        double ofedge_prefactor, int of_tolerance_sq, double of_rel_tolerance)
{
    // TODO argument error check
    clean();

    // init global data structures
    dim_ = 3;
    shape_[0] = shape_x;
    shape_[1] = shape_y;
    shape_[2] = shape_z;
    stride_[0] = shape_[1] * shape_[2];
    stride_[1] = shape_[2];
    stride_[2] = 1;
    n_vert_ = shape_x * shape_y * shape_z;

    // init build-only data structures
    n_feat_ch_ = n_feature_channels;
    int n_edge_store =  // number of graph edges
        (shape_[0] - 1) * shape_[1] * shape_[2] +
        (shape_[1] - 1) * shape_[2] * shape_[0] +
        // and at most 1 edge in (z+ + o.f.) direction
        (shape_[2] - 1) * shape_[0] * shape_[1];
    build_common(n_edge_store, feature, border_strength);

    // build image graph: add edges
    add_edges_3d( 0, -1,  0,  0,  0,  0, stride_[0]);   // x+
    add_edges_3d( 0,  0,  0, -1,  0,  0, stride_[1]);   // y+
    //add_edges_3d( 0,  0,  0,  0,  0, -1, stride_[2]);   // z+
    // add optical flow edges
    for (int x = 0; x < shape_[0]; x++) {
    for (int y = 0; y < shape_[1]; y++) {
    for (int z = 0; z < shape_[2] - 1; z++) {
        int dx = optical_flow[2 * IDX3(x, y, z)    ];
        int dy = optical_flow[2 * IDX3(x, y, z) + 1];
        int x1 = x + dx;
        int y1 = y + dy;
        if (0 <= x1 and x1 < shape_[0] and 0 <= y1 and y1 < shape_[1]) {
            int dx_r = optical_flow_reverse[2 * IDX3(x1, y1, z)    ];
            int dy_r = optical_flow_reverse[2 * IDX3(x1, y1, z) + 1];
            int d_sq = dx * dx + dy * dy;
            int d_r_sq = dx_r * dx_r + dy_r * dy_r;
            int tolerance_sq = std::max({
                    of_tolerance_sq,
                    (int)(of_rel_tolerance * of_rel_tolerance * d_sq),
                    (int)(of_rel_tolerance * of_rel_tolerance * d_r_sq) });
            int errx = dx + dx_r;
            int erry = dy + dy_r;
            if (errx * errx + erry * erry <= tolerance_sq) {                
                int stride = IDX3(dx, dy, 1);
                add_graph_edge(IDX3(x, y, z), stride, ofedge_prefactor);
            }
        }
    } } }
    assert(n_edge_ <= n_edge_store);    // less if o.f. points outside

    build_hierarchy();
}

template <typename T> void
BoruvkaSuperpixel::label_o(int n_supix, T * label)
{
    // TODO error check on n_supix

    int beg = n_vert_ - n_tree_;
    int end = n_vert_ - n_supix;
    if (end < beg) {
        // more superpixels: start fresh, every pixel is superpixel
        for (int v = 0; v < n_vert_; v++) {  // for all vertices
            parent_[v] = v;
        }
        beg = 0;
    }

    // decrease number of superpixels:
    // further connect trees via parent_ according to subsequent edges
    for (int e = beg; e < end; e++) {   // loop on mst edges
        int root0 = find_root(mst_v0_[e]);
        int root1 = find_root(mst_v1_[e]);
        // ensure root_of_vertex <= vertex
        if (root0 < root1) {
            parent_[root1] = root0;
        } else {
            parent_[root0] = root1;
        }
    }

    // supix label: number sequentially the trees
    n_tree_ = 0;
    for (int v = 0; v < n_vert_; v++) { // for all vertices
        int root = find_root(v);
        if (v == root) {                // first encounter with this tree
            label[v] = n_tree_++;      // take next label
        } else {
            label[v] = label[root];   // label was taken at root vertex
        }
    }
    assert(n_tree_ == n_supix);
}

int32_t *
BoruvkaSuperpixel::label(int n_supix)
{
    if (n_supix == 0) {
        // only makes sense when n_mst_ < n_vert_-1,
        // eg. when seeded, and penalized merges are not executed
        n_supix = min_n_supix();
    }
    if (n_supix < min_n_supix()) {
        return nullptr;
    }
    // too large n_supix silently returns the original image

    int beg = n_vert_ - n_tree_;
    int end = n_vert_ - n_supix;
    if (end < beg) {
        // more superpixels: start fresh, every pixel is superpixel
        for (int v = 0; v < n_vert_; v++) {  // for all vertices
            parent_[v] = v;
        }
        beg = 0;
    }

    // decrease number of superpixels:
    // further connect trees via parent_ according to subsequent edges
    for (int e = beg; e < end; e++) {   // loop on mst edges
        int root0 = find_root(mst_v0_[e]);
        int root1 = find_root(mst_v1_[e]);
        // ensure root_of_vertex <= vertex
        if (root0 < root1) {
            parent_[root1] = root0;
        } else {
            parent_[root0] = root1;
        }
    }

    // supix label: number sequentially the trees
    n_tree_ = 0;
    for (int v = 0; v < n_vert_; v++) { // for all vertices
        int root = find_root(v);
        if (v == root) {                // first encounter with this tree
            label_[v] = n_tree_++;      // take next label
        } else {
            label_[v] = label_[root];   // label was taken at root vertex
        }
    }
    assert(n_tree_ == n_supix);
    return label_;  // array owned by *this
}

void
BoruvkaSuperpixel::seed_id(int32_t *result)
{
    if (seed_) {
        for (int v = 0; v < n_vert_; v++) { // for all vertices
            result[v] = seed_[find_root(v)];
        }
    } else {
        // invalidate seed_id for all pixels
        for (int v = 0; v < n_vert_; v++) { // for all vertices
            result[v] = -1;
        }
    }
}


template <typename T> T *
BoruvkaSuperpixel::average(int n_supix, int n_channels, T const *data)
{
    if (n_supix == 0) {
        // only makes sense when n_mst_ < n_vert_-1,
        // eg. when seeded, and penalized merges are not executed
        n_supix = min_n_supix();
    }

    // prepare output array
    if (avg_data_) {
        free(avg_data_);
    }
    avg_data_ = malloc(n_channels * n_vert_ * sizeof(T));

    int32_t *ret = label(n_supix);
    if (not ret) {
        return nullptr;
    }

    // perform average
    int *count = new int[n_supix];
    float *sum = new float[n_supix * n_channels];
    memset(count, 0, sizeof(int) * n_supix);
    memset(sum, 0, sizeof(float) * n_supix * n_channels);
    for (int v = 0; v < n_vert_; v++) {
        count[label_[v]]++;
        for (int c = 0; c < n_channels; c++) {
            sum[n_channels * label_[v] + c] += data[n_channels * v + c];
        }
    }
    // write to output
    for (int v = 0; v < n_vert_; v++) {
        for (int c = 0; c < n_channels; c++) {
            ((T *)avg_data_)[n_channels * v + c] = 
                sum[n_channels * label_[v] + c] / count[label_[v]];
        }
    }

    delete[] sum;
    delete[] count;
    return (T *)avg_data_;
}


#define INSTANTIATE(T)                                          \
    template void BoruvkaSuperpixel::build_2d<T>                \
        (int, int, int, T const *, T const *,                   \
         int8_t const *, int32_t const *);                      \
    template void BoruvkaSuperpixel::build_3d<T>                \
        (int, int, int, int, T const *, T const *);             \
    template void BoruvkaSuperpixel::build_3d_of<T>             \
        (int, int, int, int, T const *, T const *,              \
         int16_t const *, double);                              \
    template void BoruvkaSuperpixel::build_3d_of2<T>            \
        (int, int, int, int, T const *, T const *,              \
         int16_t const *, int16_t const *, double, int, double);\
    template T *BoruvkaSuperpixel::average<T>                   \
        (int, int, T const *);                                  \
    template void BoruvkaSuperpixel::label_o<T>                 \
        (int, T *);                                 
 
INSTANTIATE(uint8_t)
INSTANTIATE(uint16_t)
INSTANTIATE(int8_t)
INSTANTIATE(int16_t)
INSTANTIATE(int32_t)
INSTANTIATE(float)
INSTANTIATE(double)

// ******************************* internals **********************************

void
BoruvkaSuperpixel::clean()
{
    if (dim_) {
        // allocated by average
        if (avg_data_) {
            free(avg_data_);
            avg_data_ = nullptr;
        }
        // allocated by build_hierarchy
        delete[] mst_v1_;
        delete[] mst_v0_;
        delete[] parent_;
        delete[] label_;
        // allocated by build_common
        if (seed_) {
            delete[] seed_;
            seed_ = nullptr;
        }

        dim_ = 0;
    }
}


template <typename T> void
BoruvkaSuperpixel::build_common(int n_edge_store, T const *feature,
        T const *border_strength, int32_t const *seed)
{
    // init build-only data structures
    n_edge_ = 0;
    edge_store_ = new Edge[n_edge_store];       // will be filled soon
    edge_ = new Edge*[n_edge_store];            // likewise
    edges_head_ = new Edge*[n_vert_];
    for (int v = 0; v < n_vert_; v++) {         // all vertices
        edges_head_[v] = nullptr;               // empty linked list
    }

    // convert input arrays to float
    feature_ = new float[n_vert_ * n_feat_ch_]; // float array
    border_strength_ = new float[n_vert_];      // also float array
    for (int vc = 0; vc < n_vert_ * n_feat_ch_; vc++) { // all vertices&features
        feature_[vc] = feature[vc];                     // conversion
    }
    for (int v = 0; v < n_vert_; v++) {                 // all vertices
        border_strength_[v] = border_strength[v];       // conversion
    }

    // calc n_edgeless
    n_edgeless_ = 0;
    if (edgeless_) {
        for (int v = 0; v < n_vert_; v++) {
            if (edgeless_[v]) {
                n_edgeless_++;
            }
        }
    }

    // seed: alloc only if supplied
    if (seed) {
        seed_ = new int32_t[n_vert_];
        for (int v = 0; v < n_vert_; v++) {         // all vertices
            seed_[v] = seed[v];
        }
    } else {
        seed_ = nullptr;
    }
}


void BoruvkaSuperpixel::build_hierarchy()
{
    // init global data structures
    n_tree_ = n_vert_;                  // initially each vertex is a tree
    label_ = new int[n_vert_];
    parent_ = new int[n_vert_];
    mst_v0_ = new int[n_vert_];         // no need to initialize
    mst_v1_ = new int[n_vert_];
    for (int v = 0; v < n_vert_; v++) { // all vertices
        label_[v] = v;
        parent_[v] = v;
    }

    // init build-only data structures
    n_mst_ = 0;
    int *tree_root = new int[n_vert_];          // per tree label: root vertex
    int *tree_size = new int[n_vert_];          // per root vertex: tree size
    typedef struct {
        double value;    // value of min outgoing edge
        Edge *edge;     // min outgoing edge
    } MinPair_t;
    MinPair_t *min_pair = new MinPair_t[n_vert_];// will be used for sorting
    //
    for (int v = 0; v < n_vert_; v++) {         // all vertices
        edges_head_[v] = nullptr;               // empty linked list
        tree_root[v] = v;
        tree_size[v] = 1;
    }

    // build Boruvka minimum spanning tree hierarchy
    for (int boruvka_iter = 0; n_tree_ > 1; boruvka_iter++) {

        // STEP 1
        // find minimal outgoing edge for each tree
        for (int t = 0; t < n_tree_; t++) {     // all trees
            min_pair[t].value = FLT_MAX;        // invalidate distance
        }
        for (int e = 0; e < n_edge_; e++) {     // all inter-tree edges
            Edge *edge = edge_[e];
            int label0 = label_[edge->v0];      // label: 0 .. n_tree_-1
            int label1 = label_[edge->v1];
            double dist = calc_dist(edge, boruvka_iter);
            if (dist < min_pair[label0].value) {
                min_pair[label0].value = dist;
                min_pair[label0].edge = edge;
            }
            if (dist < min_pair[label1].value) {
                min_pair[label1].value = dist;
                min_pair[label1].edge = edge;
            }
        }

        // STEP 2
        // connect trees with min outgoing edges
        // update: mst_v0_, mst_v1_, n_mst_, parent_, seed_
        std::sort(min_pair, min_pair + n_tree_,
                [](const MinPair_t & a, const MinPair_t & b)
                -> bool { return a.value < b.value; });
            // within a Boruvka iteration: connect weak edges earlier
        bool all_penalized = (min_pair[0].value >= PENALTY);
        if (all_penalized) {
            // break out of boruvka_iter loop
            break;
        }
        for (int t = 0; t < n_tree_; t++) {     // all min outgoing edges
            if (min_pair[t].value >= FLT_MAX) {
                break;
            }
            Edge *edge = min_pair[t].edge;
            int root0 = find_root(edge->v0);
            int root1 = find_root(edge->v1);
            if (root0 == root1) {       // reverse link was already added
                continue;
            }
            if ((not all_penalized) and min_pair[t].value >= PENALTY) {
                // leave condition in case later wanted to change code 
                //   to do penalized merges
                // process penalized edges only when no other left
                continue;
            }
            if (seed_) {
                int seed0 = seed_[root0];
                int seed1 = seed_[root1];
                if (seed0 >= 0 and seed1 >= 0 and seed0 != seed1) {
                    continue;
                }
                if (seed0 >= 0 and seed1 < 0) {
                    seed_[root1] = seed0;
                }
                if (seed1 >= 0 and seed0 < 0) {
                    seed_[root0] = seed1;
                }
            }
            // add edge to MST, connecting the root vertices of the trees
            mst_v0_[n_mst_] = root0;
            mst_v1_[n_mst_] = root1;
            n_mst_++;
            if (root0 < root1) {        // ensure root(v) <= v
                parent_[root1] = root0;
            } else {
                parent_[root0] = root1;
            }
        }

        // STEP 3
        // update: n_tree_, label_, tree_root, edges_head_, feature_
        // (and tree_size, only used here)
        int n_tree_old = n_tree_;   // number of trees in previous iteration
        n_tree_ = 0;                // counting new trees
        for (int t = 0; t < n_tree_old; t++) {  // all old trees
            int v = tree_root[t];       // root vertex of old tree
            int root = find_root(v);    // new root

            if (v == root) {
                // first encounter with this new tree
                label_[v] = n_tree_;
                tree_root[n_tree_] = v;
                edges_head_[n_tree_] = nullptr;
                n_tree_++;
            } else {
                // this old tree is now appended to another tree
                label_[v] = label_[root];
                int size = tree_size[v];
                int size_root = tree_size[root];
                int size_tot = size + size_root;
                tree_size[root] = size_tot;
                for (int c = 0; c < n_feat_ch_; c++) {
                    // its features are blended into new tree, saved at new root
                    feature_[n_feat_ch_ * root + c] = (
                        feature_[n_feat_ch_ * v + c] * size +
                        feature_[n_feat_ch_ * root + c] * size_root) / size_tot;
                }
            }
        }
        assert(n_tree_ + n_mst_ == n_vert_);

        // STEP 4
        // update edges: remove intra-tree edges
        // update: n_edge_, edge_, edges_head_
        int n_edge_old = n_edge_;   // number of edges in previous iteration
        n_edge_ = 0;                // counting active edges
        for (int e = 0; e < n_edge_old; e++) {    // all old inter-tree edges
            Edge *edge = edge_[e];
            CHECK_EDGE(edge);
            int v0 = edge->v0;
            int v1 = edge->v1;
            int root0 = parent_[v0];        // all paths are compressed now
            int root1 = parent_[v1];
            assert(root0 == parent_[root0]);
            assert(root1 == parent_[root1]);
            if (root0 == root1) {
                // intra-tree edge: do not keep
                continue;
            }
            if (root1 < root0) {
                std::swap(v0, v1);
                std::swap(root0, root1);
            } // now root0 < root1
            int label0 = label_[v0];
            int label1 = label_[v1];
            Edge *edge0 = edges_head_[label0];
            Edge *edge1 = edges_head_[label1];
            if (edge0) {
                CHECK_EDGE(edge0);
            }
            Edge *edgefind = find_edge(edge0, root0, root1);
            if (edgefind) {
                // a link connecting these trees already exist
                // do not keep current edge
                // decrease old link's border_strength to current's
                if (edgefind->border_strength > edge->border_strength) {
                    edgefind->border_strength = edge->border_strength;
                }
            } else {
                // rewire link to connect roots
                // append to edge_ and linked lists
                edge->v0 = root0;
                edge->v1 = root1;
                assert(edge->v0 >= 0);
                assert(edge->v1 >= 0);
                edge->v0_next = edge0;
                edge->v1_next = edge1;
                edge_[n_edge_++] = edge;
                CHECK_EDGE(edge)
                edges_head_[label0] = edge;
                edges_head_[label1] = edge;
            }
        }
    }

    // allocated by build_hierarchy
    delete[] tree_root;
    delete[] tree_size;
    delete[] min_pair;
    // allocated by build_common
    delete[] border_strength_;
    delete[] feature_;
    delete[] edges_head_;
    delete[] edge_;
    delete[] edge_store_;
}


// vim: set sw=4 cin cino=\:0 sts=4 expandtab :
