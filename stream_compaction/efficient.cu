#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#include "../src/sceneStructs.h"

#define BLOCK_SIZE 256

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpsweep(int d, int n, int* data) {
            int k = blockIdx.x * blockDim.x + threadIdx.x;

            int stride = 1 << (d + 1);
            int numWorkItems = n / stride;

            if (k >= numWorkItems) return;

            int right = (k + 1) * stride - 1;
            int left = right - (1 << d);

            data[right] += data[left];
        }

        __global__ void kernDownsweep(int d, int n, int* data) {
            int k = blockIdx.x * blockDim.x + threadIdx.x;

            int stride = 1 << (d + 1);
            int numWorkItems = n / stride;

            if (k >= numWorkItems) return;

            int right = (k + 1) * stride - 1;
            int left = right - (1 << d);

            int t = data[left];
            data[left] = data[right];
            data[right] = t + data[right];
        }

        __global__ void kernExtractBit(int n, int bit, int* idata, int* odata) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) return;
            
            unsigned int ukey = (static_cast<unsigned int>(idata[idx]) ^ 0x80000000u);
            int b = (ukey >> bit) & 1u;
            odata[idx] = 1 - b;
        }

        __global__ void kernScatterByBit(int n,
            int* bitArray,
            int* scanned,
            PathSegment* pathsIn, PathSegment* pathsOut,
            ShadeableIntersection* isectsIn, ShadeableIntersection* isectsOut,
            int totalFalses)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) return;

            int bit = bitArray[idx];
            int newPos;
            if (bit == 1) {
                newPos = scanned[idx];
            }
            else {
                newPos = totalFalses + (idx - scanned[idx]);
            }

            pathsOut[newPos] = pathsIn[idx];
            isectsOut[newPos] = isectsIn[idx];
        }

        __global__ void kernScatterPartition(int n, PathSegment* inPaths, PathSegment* outPaths, int* flags, int* scanned, int totalActives)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n) return;

            if (flags[i] == 1) {
                int pos = scanned[i];
                outPaths[pos] = inPaths[i];
            }
        }

        __global__ void kernScanBlock(int n, int* odata, const int* idata, int* blockSums) {
            __shared__ int temp[2 * BLOCK_SIZE];

            int thid = threadIdx.x;
            int tileBase = 2 * blockIdx.x * blockDim.x;

            int ai_s = 2 * thid;
            int bi_s = ai_s + 1;

            int ai_g = tileBase + ai_s;
            int bi_g = ai_g + 1;

            temp[ai_s] = (ai_g < n) ? idata[ai_g] : 0;
            temp[bi_s] = (bi_g < n) ? idata[bi_g] : 0;

            int offset = 1;
            for (int d = BLOCK_SIZE; d > 0; d >>= 1) {
                __syncthreads();
                if (thid < d) {
                    int ai = offset * (2 * thid + 1) - 1;
                    int bi = offset * (2 * thid + 2) - 1;
                    temp[bi] += temp[ai];
                }
                offset <<= 1;
            }

            __syncthreads();

            if (thid == 0) {
                blockSums[blockIdx.x] = temp[2 * BLOCK_SIZE - 1];
                temp[2 * BLOCK_SIZE - 1] = 0;
            }
            __syncthreads();

            for (int d = 1; d <= BLOCK_SIZE; d <<= 1) {
                offset >>= 1;
                __syncthreads();
                if (thid < d) {
                    int ai = offset * (2 * thid + 1) - 1;
                    int bi = offset * (2 * thid + 2) - 1;
                    int t = temp[ai];
                    temp[ai] = temp[bi];
                    temp[bi] += t;
                }
            }
            __syncthreads();

            if (ai_g < n) odata[ai_g] = temp[ai_s];
            if (bi_g < n) odata[bi_g] = temp[bi_s];
        }

        __global__ void kernAddBlockSums(int n, int* data, const int* blockSumsScanned) {
            int blockOffset = blockSumsScanned[blockIdx.x];
            int ai = 2 * blockIdx.x * blockDim.x + threadIdx.x;
            int bi = ai + blockDim.x;

            if (ai < n) data[ai] += blockOffset;
            if (bi < n) data[bi] += blockOffset;
        }

        void scanDevice(int n, int* dev_out, const int* dev_in) { 
            int log2n = ilog2ceil(n);
            int m = 1 << log2n;

            int* dev_buf;
            cudaMalloc(&dev_buf, m * sizeof(int));

            cudaMemcpy(dev_buf, dev_in, n * sizeof(int), cudaMemcpyDeviceToDevice);

            if (m > n) {
                cudaMemset(dev_buf + n, 0, (m - n) * sizeof(int));
            }
            
            for (int d = 0; d < log2n; d++) {
                int numWorkItems = m >> (d + 1);
                int blocks = (numWorkItems + BLOCK_SIZE - 1) / BLOCK_SIZE;
                kernUpsweep<<<blocks, BLOCK_SIZE >>>(d, m, dev_buf);
            }
            
            cudaMemset(dev_buf + (m - 1), 0, sizeof(int));
            
            for (int d = log2n - 1; d >= 0; d--) {
                int numWorkItems = m >> (d + 1);
                int blocks = (numWorkItems + BLOCK_SIZE - 1) / BLOCK_SIZE;
                kernDownsweep<<<blocks, BLOCK_SIZE >>>(d, m, dev_buf);
            }
            
            cudaMemcpy(dev_out, dev_buf, n * sizeof(int), cudaMemcpyDeviceToDevice);
            
            cudaFree(dev_buf);
        }

        void scanDeviceShared(int n, int* dev_out, const int* dev_in) {
            int threadsPerBlock = BLOCK_SIZE;
            int elementsPerBlock = 2 * threadsPerBlock;
            int numBlocks = (n + elementsPerBlock - 1) / elementsPerBlock;

            int* dev_blockSums;
            cudaMalloc(&dev_blockSums, numBlocks * sizeof(int));

            kernScanBlock << <numBlocks, threadsPerBlock >> > (n, dev_out, dev_in, dev_blockSums);

            if (numBlocks > 1) {
                int* dev_blockSumsScanned;
                cudaMalloc(&dev_blockSumsScanned, numBlocks * sizeof(int));
                scanDevice(numBlocks, dev_blockSumsScanned, dev_blockSums);

                kernAddBlockSums << <numBlocks, threadsPerBlock >> > (n, dev_out, dev_blockSumsScanned);

                cudaFree(dev_blockSumsScanned);
            }

            cudaFree(dev_blockSums);
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            if (n <= 0) {
                return;
            }
            int* dev_in, * dev_out;
            cudaMalloc(&dev_in, n * sizeof(int));
            cudaMalloc(&dev_out, n * sizeof(int));
            cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            scanDevice(n, dev_out, dev_in);
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_out, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_in);
            cudaFree(dev_out);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, PathSegment* dev_in, PathSegment* dev_out) {
            int* dev_flags, * dev_indices;

            cudaMalloc(&dev_flags, n * sizeof(int));
            cudaMalloc(&dev_indices, n * sizeof(int));

            int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

            //timer().startGpuTimer();

            StreamCompaction::Common::kernMapToBoolean<<<gridSize, BLOCK_SIZE >>>(n, dev_flags, dev_in);
            //cudaDeviceSynchronize();

            scanDevice(n, dev_indices, dev_flags);
            //cudaDeviceSynchronize();

            int lastScan, lastFlag;
            cudaMemcpy(&lastScan, dev_indices + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastFlag, dev_flags + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
            int validCount = lastScan + lastFlag;

            kernScatterPartition<<<gridSize, BLOCK_SIZE >>>(n, dev_in, dev_out, dev_flags, dev_indices, validCount);
            //cudaDeviceSynchronize();
            //timer().endGpuTimer();

            cudaFree(dev_flags);
            cudaFree(dev_indices);

            return validCount;
        }

        void radixSort(int n, PathSegment* dev_paths, PathSegment* dev_paths_tmp, ShadeableIntersection* dev_isects, ShadeableIntersection* dev_isects_tmp) {
            int* dev_bits, * dev_scanned;
            cudaMalloc(&dev_bits, n * sizeof(int));
            cudaMalloc(&dev_scanned, n * sizeof(int));

            int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

            //timer().startGpuTimer();

            for (int bit = 0; bit < 32; bit++) {
                kernExtractBit<<<gridSize, BLOCK_SIZE >>>(n, bit, (int*)(&(dev_isects->materialId)), dev_bits);

                scanDeviceShared(n, dev_scanned, dev_bits);

                int lastScan = 0, lastIsZero = 0;
                cudaMemcpy(&lastScan, dev_scanned + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&lastIsZero, dev_bits + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
                const int totalZeros = lastScan + lastIsZero;

                kernScatterByBit<<<gridSize, BLOCK_SIZE >>>(n, dev_bits, dev_scanned, dev_paths, dev_paths_tmp, dev_isects, dev_isects_tmp, totalZeros);

                std::swap(dev_paths, dev_paths_tmp);
                std::swap(dev_isects, dev_isects_tmp);
            }

            //timer().endGpuTimer();
            cudaFree(dev_bits);
            cudaFree(dev_scanned);
        }
    }
}
