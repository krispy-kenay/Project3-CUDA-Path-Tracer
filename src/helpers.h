#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <cstdint>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include "scene.h"
#include "sceneStructs.h"
#include "utilities.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"

#pragma once

__host__ __device__ inline unsigned int rnghash(unsigned int a)
{
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

__device__ inline glm::vec2 concentricSampleDisk(thrust::default_random_engine& rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);
    float u = 2.f * u01(rng) - 1.f;
    float v = 2.f * u01(rng) - 1.f;

    if (u == 0 && v == 0) return glm::vec2(0, 0);

    float r, theta;
    if (fabs(u) > fabs(v)) {
        r = u;
        theta = (PI / 4.f) * (v / u);
    }
    else {
        r = v;
        theta = (PI / 2.f) - (PI / 4.f) * (u / v);
    }

    return r * glm::vec2(cosf(theta), sinf(theta));
}

__host__ __device__ inline void stratum_from_iter(int iter, int& sx, int& sy, int strata) {
    const int N = strata * strata;
    int s = iter % N;
    sx = s % strata;
    sy = s / strata;
}

__host__ __device__ inline glm::vec2 cp_rotation(int pixelIndex) {
    unsigned int h1 = rnghash(pixelIndex * 9781u + 0x68bc21ebu);
    unsigned int h2 = rnghash(pixelIndex * 6271u + 0x2c1b3c6du);
    float rx = (h1 & 0x00FFFFFF) / float(0x01000000);
    float ry = (h2 & 0x00FFFFFF) / float(0x01000000);
    return glm::vec2(rx, ry);
}

__device__ inline void atomicAddVec3(glm::vec3* image, int pixelIndex, const glm::vec3& contrib) {
    atomicAdd(&(image[pixelIndex].x), contrib.x);
    atomicAdd(&(image[pixelIndex].y), contrib.y);
    atomicAdd(&(image[pixelIndex].z), contrib.z);
}

__device__ inline void samplePointOnLight(
    const Geom& light,
    float u0, float u1, float u2,
    glm::vec3& point,
    glm::vec3& normal)
{

    if (light.type == SPHERE) {
        float z = - 2.f * u1 + 1.f;
        float phi = 2.f * PI * u2;
        float r = sqrtf(fmaxf(0.f, 1.f - z * z));
        glm::vec3 local(r * cosf(phi), r * sinf(phi), z);

        point = glm::vec3(light.transform * glm::vec4(local, 1.f));
        normal = glm::normalize(glm::vec3(light.invTranspose * glm::vec4(local, 0.f)));

        float radius = glm::length(glm::vec3(light.transform[0]));
    }
    else if (light.type == CUBE) {
        glm::vec3 sx(glm::length(glm::vec3(light.transform[0])), glm::length(glm::vec3(light.transform[1])), glm::length(glm::vec3(light.transform[2])));
        float faceAreas[6];
        faceAreas[0] = 4.f * sx.y * sx.z; // +X
        faceAreas[1] = 4.f * sx.y * sx.z; // -X
        faceAreas[2] = 4.f * sx.x * sx.z; // +Y
        faceAreas[3] = 4.f * sx.x * sx.z; // -Y
        faceAreas[4] = 4.f * sx.x * sx.y; // +Z
        faceAreas[5] = 4.f * sx.x * sx.y; // -Z

        float totalArea = 0.f;
        for (int i = 0; i < 6; i++) totalArea += faceAreas[i];
        float xi = u0 * totalArea;
        int face = 0;
        float accum = faceAreas[0];
        while (xi > accum && face < 5) {
            face++;
            accum += faceAreas[face];
        }

        float u = u1 * 2.f - 1.f;
        float v = u2 * 2.f - 1.f;
        glm::vec3 local, ln;

        switch (face) {
        case 0: local = glm::vec3(1, u, v); ln = glm::vec3(1, 0, 0); break;
        case 1: local = glm::vec3(-1, u, v); ln = glm::vec3(-1, 0, 0); break;
        case 2: local = glm::vec3(u, 1, v); ln = glm::vec3(0, 1, 0); break;
        case 3: local = glm::vec3(u, -1, v); ln = glm::vec3(0, -1, 0); break;
        case 4: local = glm::vec3(u, v, 1); ln = glm::vec3(0, 0, 1); break;
        default:local = glm::vec3(u, v, -1); ln = glm::vec3(0, 0, -1); break;
        }

        point = glm::vec3(light.transform * glm::vec4(local, 1.f));
        normal = glm::normalize(glm::vec3(light.invTranspose * glm::vec4(ln, 0.f)));
    }
    else if (light.type == TRIANGLE) {
        float sqrtU = sqrtf(u0);
        float b0 = 1.0f - sqrtU;
        float b1 = u1 * sqrtU;
        float b2 = 1.0f - b0 - b1;

        glm::vec3 local = b0 * light.v0 + b1 * light.v1 + b2 * light.v2;
        point = local;
        normal = glm::normalize(glm::cross(light.v1 - light.v0, light.v2 - light.v0));
    }
}

__device__ inline float areaOfLight(const Geom& light) {
    if (light.type == SPHERE) {
        float r = glm::length(glm::vec3(light.transform[0]));
        return 4.f * PI * r * r;
    }
    else if (light.type == CUBE) {
        glm::vec3 sx(glm::length(glm::vec3(light.transform[0])),
            glm::length(glm::vec3(light.transform[1])),
            glm::length(glm::vec3(light.transform[2])));
        return 2.f * (sx.x * sx.y + sx.y * sx.z + sx.z * sx.x);
    }
    return 1.f;
}

__device__ inline float bsdfPdf(const Material& m,
    const glm::vec3& n,
    const glm::vec3& wo,
    const glm::vec3& wi)
{
    float cosS = fmaxf(0.f, glm::dot(n, wi));
    return cosS / PI;
}

__host__ __device__ __forceinline__ uint32_t xorshift32(uint32_t& s) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    return s ? s : 0xA3C59AC3u;
}

__host__ __device__ __forceinline__ float uint_to_unit_float(uint32_t u) {
    return (u >> 8) * (1.0f / 16777216.0f);
}

__host__ __device__ __forceinline__ uint32_t seed_hash(int a, int b, int c, int d = 0) {
    uint32_t h = rnghash((a * 0x9E3779B1u) ^ (b + 0x7F4A7C15u));
    h ^= rnghash((c * 0x85EBCA6Bu) ^ (d + 0xC2B2AE35u) ^ (h << 1));
    return h ? h : 0x1234567u;
}

struct XRNG {
    uint32_t state;
    __host__ __device__ XRNG(uint32_t s) : state(s ? s : 0x1234567u) {}
    __host__ __device__ float next() {
        uint32_t u = xorshift32(state);
        return uint_to_unit_float(u);
    }
};

__device__ __constant__ inline uint32_t SOBOL_DIR[4][32] = {
    {0x80000000u,0x40000000u,0x20000000u,0x10000000u,0x08000000u,0x04000000u,0x02000000u,0x01000000u,
    0x00800000u,0x00400000u,0x00200000u,0x00100000u,0x00080000u,0x00040000u,0x00020000u,0x00010000u,
    0x00008000u,0x00004000u,0x00002000u,0x00001000u,0x00000800u,0x00000400u,0x00000200u,0x00000100u,
    0x00000080u,0x00000040u,0x00000020u,0x00000010u,0x00000008u,0x00000004u,0x00000002u,0x00000001u},
    {0x80000000u,0xC0000000u,0x60000000u,0xA0000000u,0x50000000u,0x88000000u,0x44000000u,0xE2000000u,
    0x71000000u,0xBA800000u,0x5D400000u,0x86A00000u,0x43500000u,0xE2880000u,0x71440000u,0xBAA20000u,
    0x5D510000u,0x86A88000u,0x43544000u,0xE2AA2000u,0x71551000u,0xBAAA8800u,0x5D555400u,0x86AAAAA0u,
    0x43555550u,0xE2AAAAA8u,0x71555554u,0xBAAAAAAAu,0x5D555555u,0x86AAAAAAu,0x43555555u,0xE2AAAAAAu},
    {0x80000000u,0x40000000u,0xE0000000u,0x90000000u,0x4C000000u,0xA6000000u,0xF3000000u,0x89800000u,
    0x44C00000u,0xA2600000u,0xF1300000u,0x88980000u,0x444C0000u,0xA2260000u,0xF1130000u,0x88898000u,
    0x4444C000u,0xA2226000u,0xF1113000u,0x88889800u,0x44444C00u,0xA2222600u,0xF1111300u,0x88888980u,
    0x444444C0u,0xA2222260u,0xF1111130u,0x88888898u,0x4444444Cu,0xA2222226u,0xF1111113u,0x88888889u},
    {0x80000000u,0xC0000000u,0x60000000u,0x20000000u,0xB0000000u,0xD8000000u,0xEC000000u,0x7A000000u,
    0xA9000000u,0xF4C00000u,0x7A600000u,0x2A200000u,0xB0B00000u,0xD8580000u,0xEC2C0000u,0x761A0000u,
    0xA90D0000u,0xF4868000u,0x7A434000u,0x2A20A000u,0xB0B05000u,0xD8582800u,0xEC2C1400u,0x761A0A00u,
    0xA90D0500u,0xF4868280u,0x7A434140u,0x2A20A0A0u,0xB0B05050u,0xD8582828u,0xEC2C1414u,0x761A0A0Au}
};

__device__ __forceinline__ float sobol_01(uint32_t index, int dim) {
    uint32_t x = 0u;
    for (int bit = 0; bit < 32; ++bit) {
        if (index & (1u << bit)) x ^= SOBOL_DIR[dim & 3][bit];
    }
    return (x >> 8) * (1.0f / 16777216.0f);
}

__device__ __forceinline__ float sobol_scrambled(uint32_t index, int dim, uint32_t scrambleMask) {
    uint32_t x = 0u;
    for (int bit = 0; bit < 32; ++bit) {
        if (index & (1u << bit)) x ^= SOBOL_DIR[dim & 3][bit];
    }
    x ^= scrambleMask;
    return (x >> 8) * (1.0f / 16777216.0f);
}

__host__ __device__ __forceinline__ float cp_rotate(float u, float shift) {
    float v = u + shift;
    return v - floorf(v);
}

__device__ __forceinline__ uint32_t scramble_mask(int iter, int pixelIdx, int dim) {
    return seed_hash(iter, pixelIdx, dim, 0x9E3779B9u);
}