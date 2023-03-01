import time
import numpy as np
cimport numpy as np
cimport cython

# Reference https://github.com/facebookresearch/Detectron/blob/main/detectron/utils/cython_nms.pyx

cdef inline np.float32_t max(np.float32_t a, np.float32_t b) nogil:
    return a if a >= b else b

cdef inline np.float32_t min(np.float32_t a, np.float32_t b) nogil:
    return a if a <= b else b


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def nms(
    np.ndarray[np.float32_t, ndim=2] xyxys,
    np.ndarray[np.float32_t, ndim=1] scores,
    np.float32_t threshold,
    np.float32_t time_limit=-1,
    np.int_t sample_idx=0
):
    cdef float s_time = time.time()
    cdef np.ndarray[np.float32_t, ndim=1] x1 = xyxys[:, 0]
    cdef np.ndarray[np.float32_t, ndim=1] y1 = xyxys[:, 1]
    cdef np.ndarray[np.float32_t, ndim=1] x2 = xyxys[:, 2]
    cdef np.ndarray[np.float32_t, ndim=1] y2 = xyxys[:, 3]
    cdef np.ndarray[np.float32_t, ndim=1] areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    cdef np.ndarray[np.int_t, ndim=1] order = scores.argsort()[::-1]

    cdef int ndets = xyxys.shape[0]
    cdef np.ndarray[np.int_t, ndim=1] suppressed = np.zeros(ndets, dtype=np.int)

    # nominal indices
    cdef int _i, _j
    # sorted indices
    cdef int i, j
    # temp variables for box i's (the box currently under consideration)
    cdef np.float32_t ix1, iy1, ix2, iy2, iarea
    # variables for computing overlap with box j (lower scoring box)
    cdef np.float32_t xx1, yy1, xx2, yy2
    cdef np.float32_t w, h
    cdef np.float32_t inter, ovr
    cdef float cur_time
    cdef list keep = []
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter + 1e-6)
            if ovr > threshold:
                suppressed[j] = 1
        cur_time = time.time()
        if 0 < time_limit < (cur_time - s_time):
            print(f"WARNING: NMS time limit {time_limit}s exceeded, Sample {sample_idx}.")
            break
    # return np.where(suppressed == 0)[0]
    return np.array(keep)
