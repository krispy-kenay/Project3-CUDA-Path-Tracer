#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

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

        __global__ void kernScatterByBit(int n, int* idata, int* bitArray,
            int* scanned, int* odata, int totalFalses) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) return;

            int bit = bitArray[idx];
            if (bit == 1) {
                odata[scanned[idx]] = idata[idx];
            }
            else {
                int pos = totalFalses + (idx - scanned[idx]);
                odata[pos] = idata[idx];
            }
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
        int compact(int n, int *odata, const int *idata) {
            int* dev_in, * dev_flags, * dev_indices, * dev_out;

            cudaMalloc(&dev_in, n * sizeof(int));
            cudaMalloc(&dev_flags, n * sizeof(int));
            cudaMalloc(&dev_indices, n * sizeof(int));
            cudaMalloc(&dev_out, n * sizeof(int));

            cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

            timer().startGpuTimer();

            timer().startGpuSubTimer("map");
            StreamCompaction::Common::kernMapToBoolean<<<gridSize, BLOCK_SIZE >>>(n, dev_flags, dev_in);
            cudaDeviceSynchronize();
            timer().endGpuSubTimer();

            timer().startGpuSubTimer("scan");
            scanDevice(n, dev_indices, dev_flags);
            cudaDeviceSynchronize();
            timer().endGpuSubTimer();

            int lastScan, lastFlag;
            cudaMemcpy(&lastScan, dev_indices + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastFlag, dev_flags + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
            int validCount = lastScan + lastFlag;

            timer().startGpuSubTimer("scatter");
            StreamCompaction::Common::kernScatter<<<gridSize, BLOCK_SIZE >>>(n, dev_out, dev_in, dev_flags, dev_indices);
            cudaDeviceSynchronize();
            timer().endGpuSubTimer();

            timer().endGpuTimer();

            if (validCount > 0) {
                cudaMemcpy(odata, dev_out, validCount * sizeof(int), cudaMemcpyDeviceToHost);
            }

            cudaFree(dev_in);
            cudaFree(dev_flags);
            cudaFree(dev_indices);
            cudaFree(dev_out);

            return validCount;
        }

        void radixSort(int n, int* odata, const int* idata) {
            int* dev_in, * dev_out, * dev_bits, * dev_scanned;
            cudaMalloc(&dev_in, n * sizeof(int));
            cudaMalloc(&dev_out, n * sizeof(int));
            cudaMalloc(&dev_bits, n * sizeof(int));
            cudaMalloc(&dev_scanned, n * sizeof(int));

            cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

            timer().startGpuTimer();

            for (int bit = 0; bit < 32; bit++) {
                kernExtractBit<<<gridSize, BLOCK_SIZE >>>(n, bit, dev_in, dev_bits);

                scanDevice(n, dev_scanned, dev_bits);

                int lastScan = 0, lastIsZero = 0;
                cudaMemcpy(&lastScan, dev_scanned + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&lastIsZero, dev_bits + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
                const int totalZeros = lastScan + lastIsZero;

                kernScatterByBit<<<gridSize, BLOCK_SIZE >>>(n, dev_in, dev_bits, dev_scanned, dev_out, totalZeros);

                std::swap(dev_in, dev_out);
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_in, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_in);
            cudaFree(dev_out);
            cudaFree(dev_bits);
            cudaFree(dev_scanned);
        }
    }
}
