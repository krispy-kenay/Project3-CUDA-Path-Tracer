#pragma once

#include "sceneStructs.h"

static unsigned int expandBits(unsigned int v);
unsigned int morton3D_10bit(float x, float y, float z);

__device__ bool rayAABB(const Ray& r, const glm::vec3& bmin, const glm::vec3& bmax, float tMax);
__global__ void leafAndReduce(const Geom* geoms, const unsigned int* idx, BVHNode* nodes, unsigned int* ready, unsigned int N);
__global__ void initNodes(BVHNode* nodes, unsigned int N);
__global__ void zeroReady(unsigned int* ready, unsigned int N);
__global__ void buildInternal(const unsigned int* codes, const unsigned int* idx, BVHNode* nodes, unsigned int N);
void buildLBVH_host(const std::vector<Geom>& geoms, bool& dev_bvh_built, int& bvhLeafCount, int& bvhNodeCount, int& bvhRootIndex, unsigned int* dev_codes, unsigned int* dev_idx, unsigned int* dev_ready, Geom* dev_geoms, BVHNode* dev_bvhNodes);