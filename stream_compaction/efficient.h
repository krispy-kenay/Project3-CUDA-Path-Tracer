#pragma once

#include "common.h"
#include "../src/sceneStructs.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();
        void scanDevice(int n, int* dev_out, const int* dev_in);
        void scanDeviceShared(int n, int* dev_out, const int* dev_in);
        void scan(int n, int *odata, const int *idata);

        int compact(int n, int *odata, const int *idata);
        int compact(int n, PathSegment* dev_in, PathSegment* dev_out);
        void radixSort(int n, PathSegment* dev_paths, PathSegment* dev_paths_tmp, ShadeableIntersection* dev_isects, ShadeableIntersection* dev_isects_tmp);
    }
}
