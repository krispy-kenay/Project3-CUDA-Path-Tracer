#include "bvh.h"

#include <numeric>
#include <algorithm>

static unsigned int expandBits(unsigned int v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}
unsigned int morton3D_10bit(float x, float y, float z) {
    x = fminf(fmaxf(x * 1024.0f, 0.0f), 1023.0f);
    y = fminf(fmaxf(y * 1024.0f, 0.0f), 1023.0f);
    z = fminf(fmaxf(z * 1024.0f, 0.0f), 1023.0f);
    unsigned int xx = expandBits((unsigned int)x);
    unsigned int yy = expandBits((unsigned int)y);
    unsigned int zz = expandBits((unsigned int)z);
    return (xx << 2) | (yy << 1) | zz;
}

__device__ int clz64(unsigned int hi, unsigned int lo) {
    if (hi != 0u) return __clz(hi);
    if (lo != 0u) return 32 + __clz(lo);
    return 64;
}

__device__ int commonPrefix(const unsigned int* codes, int i, int j, unsigned int N) {
    if (j < 0 || j >= (int)N) return -1;
    unsigned int mort_i = codes[i];
    unsigned int mort_j = codes[j];
    unsigned int idx_i = (unsigned int)i;
    unsigned int idx_j = (unsigned int)j;

    unsigned int xhi = mort_i ^ mort_j;
    unsigned int xlo = idx_i ^ idx_j;

    return clz64(xhi, xlo);
}

__device__ bool rayAABB(const Ray& r, const glm::vec3& bmin, const glm::vec3& bmax, float tMax) {
    const float eps = 1e-8f;
    glm::vec3 invD(1.f / (fabsf(r.direction.x) > eps ? r.direction.x : copysignf(eps, r.direction.x)),
        1.f / (fabsf(r.direction.y) > eps ? r.direction.y : copysignf(eps, r.direction.y)),
        1.f / (fabsf(r.direction.z) > eps ? r.direction.z : copysignf(eps, r.direction.z)));
    glm::vec3 t0 = (bmin - r.origin) * invD;
    glm::vec3 t1 = (bmax - r.origin) * invD;
    glm::vec3 tsm = glm::min(t0, t1);
    glm::vec3 tbg = glm::max(t0, t1);
    float tmin = fmaxf(fmaxf(tsm.x, tsm.y), tsm.z);
    float tmax = fminf(fminf(tbg.x, tbg.y), tbg.z);
    return (tmax >= 0.f) && (tmin <= tmax) && (tmin <= tMax);
}

__global__ void leafAndReduce(const Geom* geoms,
    const unsigned int* idx,
    BVHNode* nodes,
    unsigned int* ready,
    unsigned int N)
{
    unsigned int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= N) return;

    unsigned int leafId = (N - 1u) + l;
    unsigned int geomIndex = idx[l];
    const Geom& g = geoms[geomIndex];

    glm::vec3 mn, mx;
    if (g.type == TRIANGLE) {
        mn = glm::vec3(
            glm::min(glm::min(g.v0.x, g.v1.x), g.v2.x),
            glm::min(glm::min(g.v0.y, g.v1.y), g.v2.y),
            glm::min(glm::min(g.v0.z, g.v1.z), g.v2.z)
        );
        mx = glm::vec3(
            glm::max(glm::max(g.v0.x, g.v1.x), g.v2.x),
            glm::max(glm::max(g.v0.y, g.v1.y), g.v2.y),
            glm::max(glm::max(g.v0.z, g.v1.z), g.v2.z)
        );
    }
    else if (g.type == SPHERE) {
        mn = g.translation - g.scale;
        mx = g.translation + g.scale;
    }
    else if (g.type == CUBE) {
        mn = glm::vec3(1e30f);
        mx = glm::vec3(-1e30f);
        for (int cx = 0; cx < 2; ++cx) {
            for (int cy = 0; cy < 2; ++cy) {
                for (int cz = 0; cz < 2; ++cz) {
                    glm::vec4 corner = g.transform *
                        glm::vec4(2 * cx - 1, 2 * cy - 1, 2 * cz - 1, 1.f);
                    mn = glm::min(mn, glm::vec3(corner));
                    mx = glm::max(mx, glm::vec3(corner));
                }
            }
        }
    }

    BVHNode& n = nodes[leafId];
    n.isLeaf = 1;
    n.geomIndex = geomIndex;
    n.bmin = mn;
    n.bmax = mx;

    __threadfence();

    unsigned int cur = leafId;
    for (;;) {
        unsigned int p = nodes[cur].parent;
        if (p == 0xFFFFFFFFu) break;
        unsigned int prev = atomicAdd(&ready[p], 1u);
        if (prev == 1u) {
            unsigned int lc = nodes[p].left;
            unsigned int rc = nodes[p].right;
            for (int k = 0; k < 3; k++) {
                nodes[p].bmin[k] = fminf(nodes[lc].bmin[k], nodes[rc].bmin[k]);
                nodes[p].bmax[k] = fmaxf(nodes[lc].bmax[k], nodes[rc].bmax[k]);
            }
            __threadfence();
            cur = p;
        }
        else break;
    }
}

__global__ void initNodes(BVHNode* nodes, unsigned int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = (N == 0u) ? 0u : (2u * N - 1u);
    if (i >= total) return;

    nodes[i].bmin[0] = nodes[i].bmin[1] = nodes[i].bmin[2] = 1e30f;
    nodes[i].bmax[0] = nodes[i].bmax[1] = nodes[i].bmax[2] = -1e30f;
    nodes[i].left = 0xFFFFFFFFu;
    nodes[i].right = 0xFFFFFFFFu;
    nodes[i].parent = 0xFFFFFFFFu;
    nodes[i].isLeaf = 0u;
    nodes[i].geomIndex = 0xFFFFFFFFu;
}

__global__ void zeroReady(unsigned int* ready, unsigned int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = (N == 0u) ? 0u : (2u * N - 1u);
    if (i < total) ready[i] = 0u;
}


__global__ void buildInternal(const unsigned int* codes, const unsigned int* idx, BVHNode* nodes, unsigned int N) {
    unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (N <= 1 || t >= N - 1) return;

    int i = (int)t;

    int lcpL = commonPrefix(codes, i, i - 1, N);
    int lcpR = commonPrefix(codes, i, i + 1, N);
    int d = (lcpR > lcpL) ? 1 : -1;

    int lcpMin = commonPrefix(codes, i, i - d, N);

    int lmax = 2;
    while (commonPrefix(codes, i, i + lmax * d, N) > lcpMin) {
        lmax <<= 1;
    }

    int l = 0;
    for (int step = lmax >> 1; step > 0; step >>= 1) {
        if (commonPrefix(codes, i, i + (l + step) * d, N) > lcpMin) {
            l += step;
        }
    }
    int j = i + l * d;

    int first = min(i, j);
    int last = max(i, j);

    int cp_first_last = commonPrefix(codes, first, last, N);

    int split = first;
    int step = last - first;
    do {
        step = (step + 1) >> 1;
        int candidate = split + step;
        if (candidate < last) {
            int cp_first_cand = commonPrefix(codes, first, candidate, N);
            if (cp_first_cand > cp_first_last) {
                split = candidate;
            }
        }
    } while (step > 1);

    int leftIdx, rightIdx;
    if (split == first)
        leftIdx = (int)N - 1 + split;
    else
        leftIdx = split;

    if (split + 1 == last)
        rightIdx = (int)N - 1 + (split + 1);
    else
        rightIdx = split + 1;

    nodes[i].left = (unsigned int)leftIdx;
    nodes[i].right = (unsigned int)rightIdx;
    nodes[i].isLeaf = 0u;
    nodes[i].geomIndex = 0xFFFFFFFFu;

    nodes[leftIdx].parent = (unsigned int)i;
    nodes[rightIdx].parent = (unsigned int)i;
}

void buildLBVH_host(const std::vector<Geom>& geoms, bool& dev_bvh_built, int& bvhLeafCount, int& bvhNodeCount, int& bvhRootIndex, unsigned int* dev_codes, unsigned int* dev_idx, unsigned int* dev_ready, Geom* dev_geoms, BVHNode* dev_bvhNodes) {
    const unsigned int N = (unsigned int)geoms.size();
    if (N == 0) { dev_bvh_built = false; return; }

    std::vector<glm::vec3> c(N);
    glm::vec3 sceneMin(1e30f), sceneMax(-1e30f);

    auto aabbOfGeom = [](const Geom& g, glm::vec3& mn, glm::vec3& mx) {
        if (g.type == TRIANGLE) {
            mn = glm::vec3(glm::min(glm::min(g.v0.x, g.v1.x), g.v2.x),
                glm::min(glm::min(g.v0.y, g.v1.y), g.v2.y),
                glm::min(glm::min(g.v0.z, g.v1.z), g.v2.z));
            mx = glm::vec3(glm::max(glm::max(g.v0.x, g.v1.x), g.v2.x),
                glm::max(glm::max(g.v0.y, g.v1.y), g.v2.y),
                glm::max(glm::max(g.v0.z, g.v1.z), g.v2.z));
        }
        else if (g.type == SPHERE) {
            mn = g.translation - g.scale;
            mx = g.translation + g.scale;
        }
        else {
            mn = glm::vec3(1e30f); mx = glm::vec3(-1e30f);
            for (int cx = 0; cx < 2; ++cx)
                for (int cy = 0; cy < 2; ++cy)
                    for (int cz = 0; cz < 2; ++cz) {
                        glm::vec4 corner = g.transform * glm::vec4(2 * cx - 1, 2 * cy - 1, 2 * cz - 1, 1.f);
                        glm::vec3 p = glm::vec3(corner);
                        mn = glm::min(mn, p); mx = glm::max(mx, p);
                    }
        }
        };

    for (unsigned int i = 0; i < N; ++i) {
        glm::vec3 mn, mx; aabbOfGeom(geoms[i], mn, mx);
        glm::vec3 ci = 0.5f * (mn + mx);
        c[i] = ci;
        sceneMin = glm::min(sceneMin, mn);
        sceneMax = glm::max(sceneMax, mx);
    }
    glm::vec3 extent = sceneMax - sceneMin;
    extent = glm::max(extent, glm::vec3(1e-6f));

    std::vector<unsigned int> codes(N), idx(N);
    for (unsigned int i = 0; i < N; ++i) {
        glm::vec3 nrm = (c[i] - sceneMin) / extent;
        codes[i] = morton3D_10bit(nrm.x, nrm.y, nrm.z);
        idx[i] = i;
    }

    std::vector<unsigned int> order(N); std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(), [&](unsigned int a, unsigned int b) { return codes[a] < codes[b]; });

    std::vector<unsigned int> codesSorted(N), idxSorted(N);
    for (unsigned int r = 0; r < N; ++r) {
        codesSorted[r] = codes[order[r]];
        idxSorted[r] = idx[order[r]];
    }

    bvhLeafCount = (int)N;
    bvhNodeCount = (N == 0) ? 0 : (int)(2u * N - 1u);

    cudaMemcpy(dev_codes, codesSorted.data(), N * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_idx, idxSorted.data(), N * sizeof(unsigned int), cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 gridInit((bvhNodeCount + 255) / 256);
    initNodes<<<gridInit, block>>>(dev_bvhNodes, N);

    dim3 gridZero((bvhNodeCount + 255) / 256);
    zeroReady<<<gridZero, block>>>(dev_ready, N);

    dim3 gridBuild(((N > 0 ? N - 1 : 0) + 255) / 256);
    if (N > 1) {
        buildInternal<<<gridBuild, block>>>(dev_codes, dev_idx, dev_bvhNodes, N);
    }

    dim3 gridLeaf((N + 255) / 256);
    if (N > 0) {
        leafAndReduce<<<gridLeaf, block>>>(dev_geoms, dev_idx, dev_bvhNodes, dev_ready, N);
    }
    cudaDeviceSynchronize();

    bvhRootIndex = (N > 0) ? 0 : -1;
    dev_bvh_built = (N > 0);
}