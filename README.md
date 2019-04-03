# Boruvka Superpixel

This code is a reimplementation of the superpixel segmentation based on
Boruvka's minimum spanning tree algorithm:
Xing Wei, Qingxiong Yang, Yihong Gong, Narendra Ahuja, and Ming-Hsuan Yang: 
_Superpixel Hierarchy_, IEEE Transactions on Image Processing *27*, 4838 (2018)
[journal](https://doi.org/10.1109/TIP.2018.2836300)
[arXiv](https://arxiv.org/pdf/1605.06325.pdf)

## build python module
- make module *(in root directory of the repo)*  
  this creates `pybuild/boruvka_superpixel.*.so`, which is to be imported from
  python

## test python module
- ./src/boruvkasupix.py <input_img_path> <output_img_path> <n_supix>  
   example:
	- ./src/boruvkasupix.py test.png test_out.png 100
    
## library interface
- c++ and python
- Data types supported: uint8, uint16, int8, int16, int32, float32, float64.  
  In all cases the internal data type is float32 as a compromise between
  precision and memory use.  The data type of the arrays `feature` and
  `border_strength` in `build_2d()` or `build_3d()` should be the same, but 
  it can be independent of the data type of `data` in `average()`.

## dependencies
- python3
- numpy
- cython 0.28
- gcc/g++
