# distutils: language=c++

# cython wrapper for Boruvka Superpixel

import numpy as np
cimport numpy as np

cimport boruvka_superpixel_wrap

cdef array_from_int32_ptr(shape, np.int32_t *ptr):
    res = np.zeros(shape, dtype=np.int32)
    cdef np.int32_t [:,:] view2_res
    cdef np.int32_t [:,:] view2_ptr
    cdef np.int32_t [:,:,:] view3_res
    cdef np.int32_t [:,:,:] view3_ptr
    cdef int x
    cdef int y
    cdef int z
    if len(shape) == 2:
        view2_res = res
        view2_ptr = <np.int32_t[:shape[0], :shape[1]]>ptr
        for x in range(shape[0]):
            for y in range(shape[1]):
                view2_res[x, y] = view2_ptr[x, y]
    elif len(shape) == 3:
        view3_res = res
        view3_ptr = <np.int32_t[:shape[0], :shape[1], :shape[2]]>ptr
        for x in range(shape[0]):
            for y in range(shape[1]):
                for z in range(shape[2]):
                    view3_res[x, y, z] = view3_ptr[x, y, z]
    return res


cdef class BoruvkaSuperpixel:
    cdef boruvka_superpixel_wrap.BoruvkaSuperpixel _bosupix

    def __cinit__(self):
        self._bosupix = boruvka_superpixel_wrap.BoruvkaSuperpixel()

    @property
    def dim(self):
        return self._bosupix.dim()

    @property
    def shape(self):
        cdef int * shape_ptr = self._bosupix.shape()
        if self.dim == 2:
            return (shape_ptr[0], shape_ptr[1])
        elif self.dim == 3:
            return (shape_ptr[0], shape_ptr[1], shape_ptr[2])

    def min_n_supix(self):
        return self._bosupix.min_n_supix()

    @property
    def n_edgeless(self):
        return self._bosupix.n_edgeless()

    def build_2d(self, feature, border_strength, *, edgeless=None, seed=None):
        # make explicit dim for channel, even if n_channels == 1
        if len(feature.shape) == 2:
            feature = np.expand_dims(feature, -1)
        # assert compatible shape
        cdef int shape_x = feature.shape[0]
        cdef int shape_y = feature.shape[1]
        cdef int n_feat_ch = feature.shape[2]
        assert shape_x == border_strength.shape[0]
        assert shape_y == border_strength.shape[1]
        # make both arrays c-contiguous
        if not feature.flags['C_CONTIGUOUS']:
            feature = np.ascontiguousarray(feature)
        if not border_strength.flags['C_CONTIGUOUS']:
            border_strength = np.ascontiguousarray(border_strength)
        # cdefs are only allowed at the first indent level
        cdef np.uint8_t  [:,:,:] feature_uint8
        cdef np.uint16_t [:,:,:] feature_uint16
        cdef np.int8_t   [:,:,:] feature_int8
        cdef np.int16_t  [:,:,:] feature_int16
        cdef np.int32_t  [:,:,:] feature_int32
        cdef np.float32_t[:,:,:] feature_float32
        cdef np.float64_t[:,:,:] feature_float64
        cdef np.uint8_t  [:,:] border_strength_uint8
        cdef np.uint16_t [:,:] border_strength_uint16
        cdef np.int8_t   [:,:] border_strength_int8
        cdef np.int16_t  [:,:] border_strength_int16
        cdef np.int32_t  [:,:] border_strength_int32
        cdef np.float32_t[:,:] border_strength_float32
        cdef np.float64_t[:,:] border_strength_float64
        # process edgeless
        cdef np.int8_t [:,:] edgeless_int8
        cdef np.int8_t *edgeless_ptr
        if edgeless is not None:
            assert shape_x == edgeless.shape[0]
            assert shape_y == edgeless.shape[1]
            if not edgeless.flags['C_CONTIGUOUS']:
                edgeless = np.ascontiguousarray(edgeless)
            if edgeless.dtype != np.int8:
                edgeless = edgeless.astype(np.int8)
            edgeless_int8 = edgeless
            edgeless_ptr = &edgeless_int8[0,0]
        else:
            edgeless_ptr = NULL
        # process seed
        cdef np.int32_t [:,:] seed_int32
        cdef np.int32_t *seed_ptr
        if seed is not None:
            assert shape_x == seed.shape[0]
            assert shape_y == seed.shape[1]
            if not seed.flags['C_CONTIGUOUS']:
                seed = np.ascontiguousarray(seed)
            if seed.dtype != np.uint32:
                seed = seed.astype(np.int32)
            seed_int32 = seed
            seed_ptr = &seed_int32[0,0]
        else:
            seed_ptr = NULL
        # call c++
        if   feature.dtype == np.uint8:
            feature_uint8 = feature
            border_strength_uint8 = border_strength
            self._bosupix.build_2d(shape_x, shape_y, n_feat_ch,
                    &feature_uint8[0,0,0], &border_strength_uint8[0,0],
                    edgeless_ptr, seed_ptr)
        elif feature.dtype == np.uint16:
            feature_uint16 = feature
            border_strength_uint16 = border_strength
            self._bosupix.build_2d(shape_x, shape_y, n_feat_ch,
                    &feature_uint16[0,0,0], &border_strength_uint16[0,0],
                    edgeless_ptr, seed_ptr)
        elif feature.dtype == np.int8:
            feature_int8 = feature
            border_strength_int8 = border_strength
            self._bosupix.build_2d(shape_x, shape_y, n_feat_ch,
                    &feature_int8[0,0,0], &border_strength_int8[0,0],
                    edgeless_ptr, seed_ptr)
        elif feature.dtype == np.int16:
            feature_int16 = feature
            border_strength_int16 = border_strength
            self._bosupix.build_2d(shape_x, shape_y, n_feat_ch,
                    &feature_int16[0,0,0], &border_strength_int16[0,0],
                    edgeless_ptr, seed_ptr)
        elif feature.dtype == np.int32:
            feature_int32 = feature
            border_strength_int32 = border_strength
            self._bosupix.build_2d(shape_x, shape_y, n_feat_ch,
                    &feature_int32[0,0,0], &border_strength_int32[0,0],
                    edgeless_ptr, seed_ptr)
        elif feature.dtype == np.float32:
            feature_float32 = feature
            border_strength_float32 = border_strength
            self._bosupix.build_2d(shape_x, shape_y, n_feat_ch,
                    &feature_float32[0,0,0], &border_strength_float32[0,0],
                    edgeless_ptr, seed_ptr)
        elif feature.dtype == np.float64:
            feature_float64 = feature
            border_strength_float64 = border_strength
            self._bosupix.build_2d(shape_x, shape_y, n_feat_ch,
                    &feature_float64[0,0,0], &border_strength_float64[0,0],
                    edgeless_ptr, seed_ptr)

    def build_3d(self, feature, border_strength):
        # make explicit dim for channel, even if n_channels == 1
        if len(feature.shape) == 3:
            feature = np.expand_dims(feature, -1)
        # assert compatible shape
        cdef int shape_x = feature.shape[0]
        cdef int shape_y = feature.shape[1]
        cdef int shape_z = feature.shape[2]
        cdef int n_feat_ch = feature.shape[3]
        assert shape_x == border_strength.shape[0]
        assert shape_y == border_strength.shape[1]
        assert shape_z == border_strength.shape[2]
        # make both arrays c-contiguous
        if not feature.flags['C_CONTIGUOUS']:
            feature = np.ascontiguousarray(feature)
        if not border_strength.flags['C_CONTIGUOUS']:
            border_strength = np.ascontiguousarray(border_strength)
        # cdefs are only allowed at the first indent level
        cdef np.uint8_t  [:,:,:,:] feature_uint8
        cdef np.uint16_t [:,:,:,:] feature_uint16
        cdef np.int8_t   [:,:,:,:] feature_int8
        cdef np.int16_t  [:,:,:,:] feature_int16
        cdef np.int32_t  [:,:,:,:] feature_int32
        cdef np.float32_t[:,:,:,:] feature_float32
        cdef np.float64_t[:,:,:,:] feature_float64
        cdef np.uint8_t  [:,:,:] border_strength_uint8
        cdef np.uint16_t [:,:,:] border_strength_uint16
        cdef np.int8_t   [:,:,:] border_strength_int8
        cdef np.int16_t  [:,:,:] border_strength_int16
        cdef np.int32_t  [:,:,:] border_strength_int32
        cdef np.float32_t[:,:,:] border_strength_float32
        cdef np.float64_t[:,:,:] border_strength_float64
        # call c++
        if   feature.dtype == np.uint8:
            feature_uint8 = feature
            border_strength_uint8 = border_strength
            self._bosupix.build_3d(shape_x, shape_y, shape_z, n_feat_ch,
                    &feature_uint8[0,0,0,0], &border_strength_uint8[0,0,0])
        elif feature.dtype == np.uint16:
            feature_uint16 = feature
            border_strength_uint16 = border_strength
            self._bosupix.build_3d(shape_x, shape_y, shape_z, n_feat_ch,
                    &feature_uint16[0,0,0,0], &border_strength_uint16[0,0,0])
        elif feature.dtype == np.int8:
            feature_int8 = feature
            border_strength_int8 = border_strength
            self._bosupix.build_3d(shape_x, shape_y, shape_z, n_feat_ch,
                    &feature_int8[0,0,0,0], &border_strength_int8[0,0,0])
        elif feature.dtype == np.int16:
            feature_int16 = feature
            border_strength_int16 = border_strength
            self._bosupix.build_3d(shape_x, shape_y, shape_z, n_feat_ch,
                    &feature_int16[0,0,0,0], &border_strength_int16[0,0,0])
        elif feature.dtype == np.int32:
            feature_int32 = feature
            border_strength_int32 = border_strength
            self._bosupix.build_3d(shape_x, shape_y, shape_z, n_feat_ch,
                    &feature_int32[0,0,0,0], &border_strength_int32[0,0,0])
        elif feature.dtype == np.float32:
            feature_float32 = feature
            border_strength_float32 = border_strength
            self._bosupix.build_3d(shape_x, shape_y, shape_z, n_feat_ch,
                    &feature_float32[0,0,0,0], &border_strength_float32[0,0,0])
        elif feature.dtype == np.float64:
            feature_float64 = feature
            border_strength_float64 = border_strength
            self._bosupix.build_3d(shape_x, shape_y, shape_z, n_feat_ch,
                    &feature_float64[0,0,0,0], &border_strength_float64[0,0,0])

    def build_3d_of(self, feature, border_strength, optical_flow,
            double ofedge_prefactor):
        # make explicit dim for channel, even if n_channels == 1
        if len(feature.shape) == 3:
            feature = np.expand_dims(feature, -1)
        # assert compatible shape
        cdef int shape_x = feature.shape[0]
        cdef int shape_y = feature.shape[1]
        cdef int shape_z = feature.shape[2]
        cdef int n_feat_ch = feature.shape[3]
        assert shape_x == border_strength.shape[0]
        assert shape_y == border_strength.shape[1]
        assert shape_z == border_strength.shape[2]
        assert shape_x == optical_flow.shape[0]
        assert shape_y == optical_flow.shape[1]
        assert shape_z == optical_flow.shape[2]
        assert 2       == optical_flow.shape[3]
        # make all arrays c-contiguous
        if not feature.flags['C_CONTIGUOUS']:
            feature = np.ascontiguousarray(feature)
        if not border_strength.flags['C_CONTIGUOUS']:
            border_strength = np.ascontiguousarray(border_strength)
        if not optical_flow.flags['C_CONTIGUOUS']:
            optical_flow = np.ascontiguousarray(optical_flow)
        # cdefs are only allowed at the first indent level
        cdef np.uint8_t  [:,:,:,:] feature_uint8
        cdef np.uint16_t [:,:,:,:] feature_uint16
        cdef np.int8_t   [:,:,:,:] feature_int8
        cdef np.int16_t  [:,:,:,:] feature_int16
        cdef np.int32_t  [:,:,:,:] feature_int32
        cdef np.float32_t[:,:,:,:] feature_float32
        cdef np.float64_t[:,:,:,:] feature_float64
        cdef np.uint8_t  [:,:,:] border_strength_uint8
        cdef np.uint16_t [:,:,:] border_strength_uint16
        cdef np.int8_t   [:,:,:] border_strength_int8
        cdef np.int16_t  [:,:,:] border_strength_int16
        cdef np.int32_t  [:,:,:] border_strength_int32
        cdef np.float32_t[:,:,:] border_strength_float32
        cdef np.float64_t[:,:,:] border_strength_float64
        cdef np.int16_t  [:,:,:,:] optical_flow_int16 = optical_flow #fixed type
        # call c++
        if   feature.dtype == np.uint8:
            feature_uint8 = feature
            border_strength_uint8 = border_strength
            self._bosupix.build_3d_of(shape_x, shape_y, shape_z, n_feat_ch,
                    &feature_uint8[0,0,0,0], &border_strength_uint8[0,0,0],
                    &optical_flow_int16[0,0,0,0], ofedge_prefactor)
        elif feature.dtype == np.uint16:
            feature_uint16 = feature
            border_strength_uint16 = border_strength
            self._bosupix.build_3d_of(shape_x, shape_y, shape_z, n_feat_ch,
                    &feature_uint16[0,0,0,0], &border_strength_uint16[0,0,0],
                    &optical_flow_int16[0,0,0,0], ofedge_prefactor)
        elif feature.dtype == np.int8:
            feature_int8 = feature
            border_strength_int8 = border_strength
            self._bosupix.build_3d_of(shape_x, shape_y, shape_z, n_feat_ch,
                    &feature_int8[0,0,0,0], &border_strength_int8[0,0,0],
                    &optical_flow_int16[0,0,0,0], ofedge_prefactor)
        elif feature.dtype == np.int16:
            feature_int16 = feature
            border_strength_int16 = border_strength
            self._bosupix.build_3d_of(shape_x, shape_y, shape_z, n_feat_ch,
                    &feature_int16[0,0,0,0], &border_strength_int16[0,0,0],
                    &optical_flow_int16[0,0,0,0], ofedge_prefactor)
        elif feature.dtype == np.int32:
            feature_int32 = feature
            border_strength_int32 = border_strength
            self._bosupix.build_3d_of(shape_x, shape_y, shape_z, n_feat_ch,
                    &feature_int32[0,0,0,0], &border_strength_int32[0,0,0],
                    &optical_flow_int16[0,0,0,0], ofedge_prefactor)
        elif feature.dtype == np.float32:
            feature_float32 = feature
            border_strength_float32 = border_strength
            self._bosupix.build_3d_of(shape_x, shape_y, shape_z, n_feat_ch,
                    &feature_float32[0,0,0,0], &border_strength_float32[0,0,0],
                    &optical_flow_int16[0,0,0,0], ofedge_prefactor)
        elif feature.dtype == np.float64:
            feature_float64 = feature
            border_strength_float64 = border_strength
            self._bosupix.build_3d_of(shape_x, shape_y, shape_z, n_feat_ch,
                    &feature_float64[0,0,0,0], &border_strength_float64[0,0,0],
                    &optical_flow_int16[0,0,0,0], ofedge_prefactor)

    def build_3d_of2(self, feature, border_strength, optical_flow,
            optical_flow_reverse, double ofedge_prefactor, int of_tolerance_sq,
            double of_rel_tolerance=0.):
        # make explicit dim for channel, even if n_channels == 1
        if len(feature.shape) == 3:
            feature = np.expand_dims(feature, -1)
        # assert compatible shape
        cdef int shape_x = feature.shape[0]
        cdef int shape_y = feature.shape[1]
        cdef int shape_z = feature.shape[2]
        cdef int n_feat_ch = feature.shape[3]
        assert shape_x == border_strength.shape[0]
        assert shape_y == border_strength.shape[1]
        assert shape_z == border_strength.shape[2]
        assert shape_x == optical_flow.shape[0]
        assert shape_y == optical_flow.shape[1]
        assert shape_z == optical_flow.shape[2]
        assert 2       == optical_flow.shape[3]
        # make all arrays c-contiguous
        if not feature.flags['C_CONTIGUOUS']:
            feature = np.ascontiguousarray(feature)
        if not border_strength.flags['C_CONTIGUOUS']:
            border_strength = np.ascontiguousarray(border_strength)
        if not optical_flow.flags['C_CONTIGUOUS']:
            optical_flow = np.ascontiguousarray(optical_flow)
        # cdefs are only allowed at the first indent level
        cdef np.uint8_t  [:,:,:,:] feature_uint8
        cdef np.uint16_t [:,:,:,:] feature_uint16
        cdef np.int8_t   [:,:,:,:] feature_int8
        cdef np.int16_t  [:,:,:,:] feature_int16
        cdef np.int32_t  [:,:,:,:] feature_int32
        cdef np.float32_t[:,:,:,:] feature_float32
        cdef np.float64_t[:,:,:,:] feature_float64
        cdef np.uint8_t  [:,:,:] border_strength_uint8
        cdef np.uint16_t [:,:,:] border_strength_uint16
        cdef np.int8_t   [:,:,:] border_strength_int8
        cdef np.int16_t  [:,:,:] border_strength_int16
        cdef np.int32_t  [:,:,:] border_strength_int32
        cdef np.float32_t[:,:,:] border_strength_float32
        cdef np.float64_t[:,:,:] border_strength_float64
        cdef np.int16_t  [:,:,:,:] optflow_int16 = optical_flow #fixed type
        cdef np.int16_t  [:,:,:,:] optflow_rev_int16 = optical_flow_reverse
        # call c++
        if   feature.dtype == np.uint8:
            feature_uint8 = feature
            border_strength_uint8 = border_strength
            self._bosupix.build_3d_of2(shape_x, shape_y, shape_z, n_feat_ch,
                    &feature_uint8[0,0,0,0], &border_strength_uint8[0,0,0],
                    &optflow_int16[0,0,0,0], &optflow_rev_int16[0,0,0,0],
                    ofedge_prefactor, of_tolerance_sq, of_rel_tolerance)
        elif feature.dtype == np.uint16:
            feature_uint16 = feature
            border_strength_uint16 = border_strength
            self._bosupix.build_3d_of2(shape_x, shape_y, shape_z, n_feat_ch,
                    &feature_uint16[0,0,0,0], &border_strength_uint16[0,0,0],
                    &optflow_int16[0,0,0,0], &optflow_rev_int16[0,0,0,0],
                    ofedge_prefactor, of_tolerance_sq, of_rel_tolerance)
        elif feature.dtype == np.int8:
            feature_int8 = feature
            border_strength_int8 = border_strength
            self._bosupix.build_3d_of2(shape_x, shape_y, shape_z, n_feat_ch,
                    &feature_int8[0,0,0,0], &border_strength_int8[0,0,0],
                    &optflow_int16[0,0,0,0], &optflow_rev_int16[0,0,0,0],
                    ofedge_prefactor, of_tolerance_sq, of_rel_tolerance)
        elif feature.dtype == np.int16:
            feature_int16 = feature
            border_strength_int16 = border_strength
            self._bosupix.build_3d_of2(shape_x, shape_y, shape_z, n_feat_ch,
                    &feature_int16[0,0,0,0], &border_strength_int16[0,0,0],
                    &optflow_int16[0,0,0,0], &optflow_rev_int16[0,0,0,0],
                    ofedge_prefactor, of_tolerance_sq, of_rel_tolerance)
        elif feature.dtype == np.int32:
            feature_int32 = feature
            border_strength_int32 = border_strength
            self._bosupix.build_3d_of2(shape_x, shape_y, shape_z, n_feat_ch,
                    &feature_int32[0,0,0,0], &border_strength_int32[0,0,0],
                    &optflow_int16[0,0,0,0], &optflow_rev_int16[0,0,0,0],
                    ofedge_prefactor, of_tolerance_sq, of_rel_tolerance)
        elif feature.dtype == np.float32:
            feature_float32 = feature
            border_strength_float32 = border_strength
            self._bosupix.build_3d_of2(shape_x, shape_y, shape_z, n_feat_ch,
                    &feature_float32[0,0,0,0], &border_strength_float32[0,0,0],
                    &optflow_int16[0,0,0,0], &optflow_rev_int16[0,0,0,0],
                    ofedge_prefactor, of_tolerance_sq, of_rel_tolerance)
        elif feature.dtype == np.float64:
            feature_float64 = feature
            border_strength_float64 = border_strength
            self._bosupix.build_3d_of2(shape_x, shape_y, shape_z, n_feat_ch,
                    &feature_float64[0,0,0,0], &border_strength_float64[0,0,0],
                    &optflow_int16[0,0,0,0], &optflow_rev_int16[0,0,0,0],
                    ofedge_prefactor, of_tolerance_sq, of_rel_tolerance)

    def label_old(self, int n_supix):
        cdef int dim = self._bosupix.dim()
        cdef int * shape = self._bosupix.shape()
        cdef np.int32_t * result_ptr = self._bosupix.label(n_supix)
        if dim == 2:
            return np.asarray(<np.int32_t[:shape[0], :shape[1]]> result_ptr)
        elif dim == 3:
            return np.asarray(<np.int32_t[:shape[0], :shape[1], :shape[2]]> result_ptr)

    def label(self, int n_supix):
        cdef np.int32_t *ptr = self._bosupix.label(n_supix)
        if ptr == NULL:
            raise RuntimeError('illegal n_supix')
        return array_from_int32_ptr(self.shape, ptr)


    def label_o(self, int n_supix, label):
        cdef np.int16_t  [:,:,:] label_int16
        label_int16 = label
        self._bosupix.label_o(n_supix, &label_int16[0,0,0])

    def seed_id(self):
        res = np.zeros(self.shape, dtype=np.int32)
        cdef np.int32_t [:,:] view2_res
        cdef np.int32_t [:,:,:] view3_res
        if self.dim == 2:
            view2_res = res
            self._bosupix.seed_id(&view2_res[0,0])
        else:
            view3_res = res
            self._bosupix.seed_id(&view3_res[0,0,0])
        return res
    
    def average(self, int n_supix, int n_channels, data):
        cdef int dim = self._bosupix.dim()
        cdef int * shape = self._bosupix.shape()
        # make explicit dim for channel, even if n_channels == 1
        if len(data.shape) == dim:
            data = np.expand_dims(data, -1)
        # assert compatible shape
        assert len(data.shape) == dim+1
        for d in range(self._bosupix.dim()):
            assert data.shape[d] == shape[d]
        assert data.shape[dim] == n_channels
        # make data c-contiguous
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
        # cdefs are only allowed at the first indent level
        cdef np.uint8_t  [:,:,:] data_2d_uint8
        cdef np.uint16_t [:,:,:] data_2d_uint16
        cdef np.int8_t   [:,:,:] data_2d_int8
        cdef np.int16_t  [:,:,:] data_2d_int16
        cdef np.int32_t  [:,:,:] data_2d_int32
        cdef np.float32_t[:,:,:] data_2d_float32
        cdef np.float64_t[:,:,:] data_2d_float64
        cdef np.uint8_t  [:,:,:,:] data_3d_uint8
        cdef np.uint16_t [:,:,:,:] data_3d_uint16
        cdef np.int8_t   [:,:,:,:] data_3d_int8
        cdef np.int16_t  [:,:,:,:] data_3d_int16
        cdef np.int32_t  [:,:,:,:] data_3d_int32
        cdef np.float32_t[:,:,:,:] data_3d_float32
        cdef np.float64_t[:,:,:,:] data_3d_float64
        cdef void * avg_ptr
        if dim == 2:
            if   data.dtype == np.uint8:
                data_2d_uint8 = data
                avg_ptr = self._bosupix.average(n_supix, n_channels,
                        &data_2d_uint8[0,0,0])
                return np.asarray(<np.uint8_t[
                    :shape[0], :shape[1], :n_channels]> avg_ptr)
            elif data.dtype == np.uint16:
                data_2d_uint16 = data
                avg_ptr = self._bosupix.average(n_supix, n_channels,
                        &data_2d_uint16[0,0,0])
                return np.asarray(<np.uint16_t[
                    :shape[0], :shape[1], :n_channels]> avg_ptr)
            elif data.dtype == np.int8:
                data_2d_int8 = data
                avg_ptr = self._bosupix.average(n_supix, n_channels,
                        &data_2d_int8[0,0,0])
                return np.asarray(<np.int8_t[
                    :shape[0], :shape[1], :n_channels]> avg_ptr)
            elif data.dtype == np.int16:
                data_2d_int16 = data
                avg_ptr = self._bosupix.average(n_supix, n_channels,
                        &data_2d_int16[0,0,0])
                return np.asarray(<np.int16_t[
                    :shape[0], :shape[1], :n_channels]> avg_ptr)
            elif data.dtype == np.int32:
                data_2d_int32 = data
                avg_ptr = self._bosupix.average(n_supix, n_channels,
                        &data_2d_int32[0,0,0])
                return np.asarray(<np.int32_t[
                    :shape[0], :shape[1], :n_channels]> avg_ptr)
            elif data.dtype == np.float32:
                data_2d_float32 = data
                avg_ptr = self._bosupix.average(n_supix, n_channels,
                        &data_2d_float32[0,0,0])
                return np.asarray(<np.float32_t[
                    :shape[0], :shape[1], :n_channels]> avg_ptr)
            elif data.dtype == np.float64:
                data_2d_float64 = data
                avg_ptr = self._bosupix.average(n_supix, n_channels,
                        &data_2d_float64[0,0,0])
                return np.asarray(<np.float64_t[
                    :shape[0], :shape[1], :n_channels]> avg_ptr)
        elif dim == 3:
            if   data.dtype == np.uint8:
                data_3d_uint8 = data
                avg_ptr = self._bosupix.average(n_supix, n_channels,
                        &data_3d_uint8[0,0,0,0])
                return np.asarray(<np.uint8_t[
                    :shape[0], :shape[1], :shape[2], :n_channels]> avg_ptr)
            elif data.dtype == np.uint16:
                data_3d_uint16 = data
                avg_ptr = self._bosupix.average(n_supix, n_channels,
                        &data_3d_uint16[0,0,0,0])
                return np.asarray(<np.uint16_t[
                    :shape[0], :shape[1], :shape[2], :n_channels]> avg_ptr)
            elif data.dtype == np.int8:
                data_3d_int8 = data
                avg_ptr = self._bosupix.average(n_supix, n_channels,
                        &data_3d_int8[0,0,0,0])
                return np.asarray(<np.int8_t[
                    :shape[0], :shape[1], :shape[2], :n_channels]> avg_ptr)
            elif data.dtype == np.int16:
                data_3d_int16 = data
                avg_ptr = self._bosupix.average(n_supix, n_channels,
                        &data_3d_int16[0,0,0,0])
                return np.asarray(<np.int16_t[
                    :shape[0], :shape[1], :shape[2], :n_channels]> avg_ptr)
            elif data.dtype == np.int32:
                data_3d_int32 = data
                avg_ptr = self._bosupix.average(n_supix, n_channels,
                        &data_3d_int32[0,0,0,0])
                return np.asarray(<np.int32_t[
                    :shape[0], :shape[1], :shape[2], :n_channels]> avg_ptr)
            elif data.dtype == np.float32:
                data_3d_float32 = data
                avg_ptr = self._bosupix.average(n_supix, n_channels,
                        &data_3d_float32[0,0,0,0])
                return np.asarray(<np.float32_t[
                    :shape[0], :shape[1], :shape[2], :n_channels]> avg_ptr)
            elif data.dtype == np.float64:
                data_3d_float64 = data
                avg_ptr = self._bosupix.average(n_supix, n_channels,
                        &data_3d_float64[0,0,0,0])
                return np.asarray(<np.float64_t[
                    :shape[0], :shape[1], :shape[2], :n_channels]> avg_ptr)

# vim: set sw=4 sts=4 expandtab :
