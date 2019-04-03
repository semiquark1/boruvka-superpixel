# distutils: language=c++

cimport numpy as np

cdef extern from "boruvka_superpixel.h":
    cdef cppclass BoruvkaSuperpixel:
        BoruvkaSuperpixel() except +
        int dim()
        int * shape()
        int n_vert()
        int min_n_supix()
        int n_edgeless()
        # build_2d
        void build_2d(int, int, int, np.uint8_t*,    np.uint8_t*   ,
                np.int8_t *, np.int32_t*)
        void build_2d(int, int, int, np.uint16_t *,  np.uint16_t * ,
                np.int8_t *, np.int32_t *)
        void build_2d(int, int, int, np.int8_t *,    np.int8_t *   ,
                np.int8_t *, np.int32_t *)
        void build_2d(int, int, int, np.int16_t *,   np.int16_t *  ,
                np.int8_t *, np.int32_t *)
        void build_2d(int, int, int, np.int32_t *,   np.int32_t *  ,
                np.int8_t *, np.int32_t *)
        void build_2d(int, int, int, np.float32_t *, np.float32_t *,
                np.int8_t *, np.int32_t *)
        void build_2d(int, int, int, np.float64_t *, np.float64_t *,
                np.int8_t *, np.int32_t *)
        # build_3d
        void build_3d(int, int, int, int, np.uint8_t *,   np.uint8_t *  )
        void build_3d(int, int, int, int, np.uint16_t *,  np.uint16_t * )
        void build_3d(int, int, int, int, np.int8_t *,    np.int8_t *   )
        void build_3d(int, int, int, int, np.int16_t *,   np.int16_t *  )
        void build_3d(int, int, int, int, np.int32_t *,   np.int32_t *  )
        void build_3d(int, int, int, int, np.float32_t *, np.float32_t *)
        void build_3d(int, int, int, int, np.float64_t *, np.float64_t *)
        # build_3d_of
        void build_3d_of(int, int, int, int, np.uint8_t *,   np.uint8_t *  ,
                np.int16_t *, double)
        void build_3d_of(int, int, int, int, np.uint16_t *,  np.uint16_t * ,
                np.int16_t *, double)
        void build_3d_of(int, int, int, int, np.int8_t *,    np.int8_t *   ,
                np.int16_t *, double)
        void build_3d_of(int, int, int, int, np.int16_t *,   np.int16_t *  ,
                np.int16_t *, double)
        void build_3d_of(int, int, int, int, np.int32_t *,   np.int32_t *  ,
                np.int16_t *, double)
        void build_3d_of(int, int, int, int, np.float32_t *, np.float32_t *,
                np.int16_t *, double)
        void build_3d_of(int, int, int, int, np.float64_t *, np.float64_t *,
                np.int16_t *, double)
        # build_3d_of2
        void build_3d_of2(int, int, int, int, np.uint8_t *,   np.uint8_t *  ,
                np.int16_t *, np.int16_t *, double, int, double)
        void build_3d_of2(int, int, int, int, np.uint16_t *,  np.uint16_t * ,
                np.int16_t *, np.int16_t *, double, int, double)
        void build_3d_of2(int, int, int, int, np.int8_t *,    np.int8_t *   ,
                np.int16_t *, np.int16_t *, double, int, double)
        void build_3d_of2(int, int, int, int, np.int16_t *,   np.int16_t *  ,
                np.int16_t *, np.int16_t *, double, int, double)
        void build_3d_of2(int, int, int, int, np.int32_t *,   np.int32_t *  ,
                np.int16_t *, np.int16_t *, double, int, double)
        void build_3d_of2(int, int, int, int, np.float32_t *, np.float32_t *,
                np.int16_t *, np.int16_t *, double, int, double)
        void build_3d_of2(int, int, int, int, np.float64_t *, np.float64_t *,
                np.int16_t *, np.int16_t *, double, int, double)
        # 
        np.int32_t * label(int)
        void label_o(int, np.int16_t *)
        void seed_id(np.int32_t *)
        # average
        np.uint8_t *   average(int, int, np.uint8_t *  )
        np.uint16_t *  average(int, int, np.uint16_t * )
        np.int8_t *    average(int, int, np.int8_t *   )
        np.int16_t *   average(int, int, np.int16_t *  )
        np.int32_t *   average(int, int, np.int32_t *  )
        np.float32_t * average(int, int, np.float32_t *)
        np.float64_t * average(int, int, np.float64_t *)

# vim: set sw=4 sts=4 expandtab :
