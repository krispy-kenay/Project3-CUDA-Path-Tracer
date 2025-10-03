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
        faceAreas[0] = 4.f * sx.y * sx.z;
        faceAreas[1] = 4.f * sx.y * sx.z;
        faceAreas[2] = 4.f * sx.x * sx.z;
        faceAreas[3] = 4.f * sx.x * sx.z;
        faceAreas[4] = 4.f * sx.x * sx.y;
        faceAreas[5] = 4.f * sx.x * sx.y;

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

// Sobol sequence generated via python helper file
__device__ __constant__ inline uint32_t SOBOL_DIR[32][32] = {
{
    0x80000000u, 0x40000000u, 0x20000000u, 0x10000000u,
    0x08000000u, 0x04000000u, 0x02000000u, 0x01000000u,
    0x00800000u, 0x00400000u, 0x00200000u, 0x00100000u,
    0x00080000u, 0x00040000u, 0x00020000u, 0x00010000u,
    0x00008000u, 0x00004000u, 0x00002000u, 0x00001000u,
    0x00000800u, 0x00000400u, 0x00000200u, 0x00000100u,
    0x00000080u, 0x00000040u, 0x00000020u, 0x00000010u,
    0x00000008u, 0x00000004u, 0x00000002u, 0x00000001u
},
{
    0x80000000u, 0xC0000000u, 0xA0000000u, 0xF0000000u,
    0x88000000u, 0xCC000000u, 0xAA000000u, 0xFF000000u,
    0x80800000u, 0xC0C00000u, 0xA0A00000u, 0xF0F00000u,
    0x88880000u, 0xCCCC0000u, 0xAAAA0000u, 0xFFFF0000u,
    0x80008000u, 0xC000C000u, 0xA000A000u, 0xF000F000u,
    0x88008800u, 0xCC00CC00u, 0xAA00AA00u, 0xFF00FF00u,
    0x80808080u, 0xC0C0C0C0u, 0xA0A0A0A0u, 0xF0F0F0F0u,
    0x88888888u, 0xCCCCCCCCu, 0xAAAAAAAAu, 0xFFFFFFFFu
},
{
    0x80000000u, 0x40000000u, 0xA0000000u, 0x50000000u,
    0x88000000u, 0x44000000u, 0xAA000000u, 0x55000000u,
    0x80800000u, 0x40400000u, 0xA0A00000u, 0x50500000u,
    0x88880000u, 0x44440000u, 0xAAAA0000u, 0x55550000u,
    0x80008000u, 0x40004000u, 0xA000A000u, 0x50005000u,
    0x88008800u, 0x44004400u, 0xAA00AA00u, 0x55005500u,
    0x80808080u, 0x40404040u, 0xA0A0A0A0u, 0x50505050u,
    0x88888888u, 0x44444444u, 0xAAAAAAAAu, 0x55555555u
},
{
    0x80000000u, 0x40000000u, 0x20000000u, 0x80000000u,
    0x08000000u, 0x20000000u, 0x80000000u, 0x49000000u,
    0x00800000u, 0x90400000u, 0x08000000u, 0x04900000u,
    0x80000000u, 0x49000000u, 0x20820000u, 0x80410000u,
    0x00008000u, 0x24920000u, 0x82002000u, 0x41008000u,
    0x00800000u, 0x92002400u, 0x00208200u, 0x00804100u,
    0x80000000u, 0x40249240u, 0x20820000u, 0x80410000u,
    0x08008008u, 0x20920004u, 0x80002002u, 0x49008008u
},
{
    0x80000000u, 0x40000000u, 0x20000000u, 0x80000000u,
    0x40000000u, 0x04000000u, 0x80000000u, 0x49000000u,
    0x24800000u, 0x82400000u, 0x49000000u, 0x00800000u,
    0x80480000u, 0x40000000u, 0x20820000u, 0x80410000u,
    0x40208000u, 0x04820000u, 0x80410000u, 0x49041000u,
    0x24020000u, 0x82082400u, 0x49241200u, 0x00004900u,
    0x80002400u, 0x40008200u, 0x20004920u, 0x80000000u,
    0x40008008u, 0x04004004u, 0x80002002u, 0x49008008u
},
{
    0x80000000u, 0x40000000u, 0x20000000u, 0x10000000u,
    0x80000000u, 0x40000000u, 0x20000000u, 0x01000000u,
    0x80000000u, 0x40000000u, 0x22200000u, 0x11100000u,
    0x80000000u, 0x40440000u, 0x22200000u, 0x00010000u,
    0x80088000u, 0x40044000u, 0x20022000u, 0x10000000u,
    0x80088000u, 0x40044000u, 0x20020200u, 0x01011000u,
    0x80088000u, 0x40044440u, 0x22220020u, 0x11101100u,
    0x80088088u, 0x40404000u, 0x22222202u, 0x00000001u
},
{
    0x80000000u, 0x40000000u, 0x20000000u, 0x10000000u,
    0x80000000u, 0x04000000u, 0x20000000u, 0x01000000u,
    0x88800000u, 0x40000000u, 0x02000000u, 0x00100000u,
    0x80000000u, 0x04000000u, 0x00200000u, 0x00010000u,
    0x88008000u, 0x40404000u, 0x20022000u, 0x10000000u,
    0x88808800u, 0x00040000u, 0x22000200u, 0x00000100u,
    0x80088000u, 0x40000000u, 0x00200220u, 0x00100000u,
    0x88000800u, 0x00000400u, 0x00220002u, 0x00000001u
},
{
    0x80000000u, 0x40000000u, 0x20000000u, 0x10000000u,
    0x08000000u, 0x80000000u, 0x40000000u, 0x01000000u,
    0x00800000u, 0x08000000u, 0x84200000u, 0x40000000u,
    0x20000000u, 0x10840000u, 0x00400000u, 0x84200000u,
    0x42100000u, 0x00080000u, 0x00842000u, 0x00400000u,
    0x80200800u, 0x40108000u, 0x20004200u, 0x10842100u,
    0x08421080u, 0x80000000u, 0x40008020u, 0x01004010u,
    0x00002000u, 0x08401000u, 0x84000800u, 0x40108021u
},
{
    0x80000000u, 0x40000000u, 0x20000000u, 0x10000000u,
    0x08000000u, 0x80000000u, 0x40000000u, 0x20000000u,
    0x00800000u, 0x00400000u, 0x80000000u, 0x42100000u,
    0x21080000u, 0x10840000u, 0x08000000u, 0x80210000u,
    0x42100000u, 0x21004000u, 0x00040000u, 0x00020000u,
    0x80000000u, 0x40000000u, 0x20080200u, 0x10042000u,
    0x08021000u, 0x80010040u, 0x40008400u, 0x20080010u,
    0x00842108u, 0x00421000u, 0x80000840u, 0x42100401u
},
{
    0x80000000u, 0x40000000u, 0x20000000u, 0x10000000u,
    0x08000000u, 0x80000000u, 0x02000000u, 0x01000000u,
    0x00800000u, 0x08000000u, 0x80000000u, 0x40000000u,
    0x00080000u, 0x00800000u, 0x00020000u, 0x84200000u,
    0x00008000u, 0x21004000u, 0x00800000u, 0x08021000u,
    0x80000000u, 0x42100000u, 0x20000000u, 0x10000000u,
    0x00000080u, 0x84000040u, 0x00008000u, 0x00004000u,
    0x00000008u, 0x00000080u, 0x80200800u, 0x40108021u
},
{
    0x80000000u, 0x40000000u, 0x20000000u, 0x10000000u,
    0x08000000u, 0x84000000u, 0x42000000u, 0x01000000u,
    0x10800000u, 0x00400000u, 0x84200000u, 0x42100000u,
    0x20080000u, 0x10040000u, 0x00020000u, 0x80010000u,
    0x42008000u, 0x01084000u, 0x10042000u, 0x08001000u,
    0x80210800u, 0x40100400u, 0x20084200u, 0x10842100u,
    0x08400080u, 0x84000040u, 0x42100020u, 0x01000010u,
    0x10040008u, 0x00000084u, 0x84010042u, 0x42000001u
},
{
    0x80000000u, 0x40000000u, 0x20000000u, 0x10000000u,
    0x08000000u, 0x80000000u, 0x02000000u, 0x20000000u,
    0x10000000u, 0x08000000u, 0x84200000u, 0x40000000u,
    0x01000000u, 0x00040000u, 0x08000000u, 0x80210000u,
    0x02000000u, 0x01084000u, 0x10042000u, 0x08000000u,
    0x80210000u, 0x40108000u, 0x21080000u, 0x10842100u,
    0x00021080u, 0x84010840u, 0x00108400u, 0x21000200u,
    0x10000108u, 0x00400084u, 0x84200840u, 0x42100401u
},
{
    0x80000000u, 0x40000000u, 0x20000000u, 0x10000000u,
    0x08000000u, 0x04000000u, 0x80000000u, 0x01000000u,
    0x20000000u, 0x00400000u, 0x08000000u, 0x00100000u,
    0x82080000u, 0x40000000u, 0x00800000u, 0x00010000u,
    0x08208000u, 0x04000000u, 0x82002000u, 0x00001000u,
    0x00820800u, 0x00400000u, 0x08200200u, 0x00000100u,
    0x80082000u, 0x40040040u, 0x20820000u, 0x10000000u,
    0x00008208u, 0x00004000u, 0x82082080u, 0x00000001u
},
{
    0x80000000u, 0x40000000u, 0x20000000u, 0x10000000u,
    0x08000000u, 0x04000000u, 0x80000000u, 0x40000000u,
    0x00800000u, 0x10000000u, 0x08000000u, 0x04000000u,
    0x82080000u, 0x40000000u, 0x20000000u, 0x10410000u,
    0x00008000u, 0x00004000u, 0x80082000u, 0x40041000u,
    0x00820800u, 0x10000000u, 0x00200000u, 0x00104100u,
    0x80082000u, 0x40041000u, 0x20820000u, 0x10410410u,
    0x08000000u, 0x04104000u, 0x80080002u, 0x40040001u
},
{
    0x80000000u, 0x40000000u, 0x20000000u, 0x10000000u,
    0x08000000u, 0x04000000u, 0x80000000u, 0x01000000u,
    0x20000000u, 0x10000000u, 0x00200000u, 0x04000000u,
    0x82080000u, 0x41040000u, 0x00020000u, 0x00010000u,
    0x08208000u, 0x04104000u, 0x80000000u, 0x01041000u,
    0x00020000u, 0x10410000u, 0x00208200u, 0x00100000u,
    0x80080080u, 0x41000040u, 0x20020020u, 0x10010010u,
    0x00000008u, 0x00104004u, 0x82082080u, 0x00000001u
},
{
    0x80000000u, 0x40000000u, 0x20000000u, 0x10000000u,
    0x08000000u, 0x04000000u, 0x80000000u, 0x01000000u,
    0x00800000u, 0x10000000u, 0x00200000u, 0x04000000u,
    0x80000000u, 0x40000000u, 0x00020000u, 0x00400000u,
    0x00200000u, 0x00100000u, 0x80000000u, 0x01041000u,
    0x20820000u, 0x10400400u, 0x00208200u, 0x00100000u,
    0x82080000u, 0x41041040u, 0x20820000u, 0x10000410u,
    0x00000200u, 0x00000004u, 0x82080000u, 0x00001000u
},
{
    0x80000000u, 0x40000000u, 0x20000000u, 0x10000000u,
    0x08000000u, 0x04000000u, 0x02000000u, 0x80000000u,
    0x40000000u, 0x20000000u, 0x00200000u, 0x00100000u,
    0x00080000u, 0x02000000u, 0x81020000u, 0x40810000u,
    0x20000000u, 0x10000000u, 0x08000000u, 0x04081000u,
    0x00040000u, 0x81000400u, 0x40810000u, 0x20408000u,
    0x00000080u, 0x00002000u, 0x00081020u, 0x00040810u,
    0x80020008u, 0x40010004u, 0x20008002u, 0x10004080u
},
{
    0x80000000u, 0x40000000u, 0x20000000u, 0x10000000u,
    0x08000000u, 0x04000000u, 0x02000000u, 0x80000000u,
    0x40000000u, 0x20000000u, 0x10000000u, 0x00100000u,
    0x00080000u, 0x00040000u, 0x80000000u, 0x40810000u,
    0x20408000u, 0x10204000u, 0x08102000u, 0x04000000u,
    0x02000000u, 0x80020400u, 0x40810000u, 0x20400100u,
    0x10200080u, 0x00002000u, 0x00001000u, 0x00000010u,
    0x80000000u, 0x40000000u, 0x20008002u, 0x10004080u
},
{
    0x80000000u, 0x40000000u, 0x20000000u, 0x10000000u,
    0x08000000u, 0x04000000u, 0x02000000u, 0x81000000u,
    0x00800000u, 0x00400000u, 0x10200000u, 0x08100000u,
    0x00080000u, 0x00040000u, 0x81020000u, 0x40010000u,
    0x00408000u, 0x00204000u, 0x00102000u, 0x00001000u,
    0x02040800u, 0x81020400u, 0x00810200u, 0x20408100u,
    0x10004080u, 0x08000040u, 0x04000020u, 0x00000010u,
    0x81020008u, 0x40010204u, 0x20400102u, 0x10200001u
},
{
    0x80000000u, 0x40000000u, 0x20000000u, 0x10000000u,
    0x08000000u, 0x04000000u, 0x02000000u, 0x81000000u,
    0x40800000u, 0x00400000u, 0x00200000u, 0x08100000u,
    0x00080000u, 0x02040000u, 0x80020000u, 0x40810000u,
    0x20008000u, 0x10004000u, 0x00102000u, 0x00081000u,
    0x00000800u, 0x80000400u, 0x40800200u, 0x00400100u,
    0x00000080u, 0x00000040u, 0x04080020u, 0x02040810u,
    0x80000408u, 0x40800004u, 0x20408002u, 0x10004081u
},
{
    0x80000000u, 0x40000000u, 0x20000000u, 0x10000000u,
    0x08000000u, 0x04000000u, 0x02000000u, 0x81000000u,
    0x00800000u, 0x20400000u, 0x10200000u, 0x08100000u,
    0x04080000u, 0x02040000u, 0x81020000u, 0x40810000u,
    0x00008000u, 0x00204000u, 0x08002000u, 0x04001000u,
    0x02000800u, 0x81000400u, 0x00800200u, 0x00400100u,
    0x10200080u, 0x08102040u, 0x00080020u, 0x02040810u,
    0x81020408u, 0x40810204u, 0x20008102u, 0x10204081u
},
{
    0x80000000u, 0x40000000u, 0x20000000u, 0x10000000u,
    0x08000000u, 0x04000000u, 0x02000000u, 0x01000000u,
    0x80000000u, 0x00400000u, 0x20000000u, 0x10000000u,
    0x00080000u, 0x04000000u, 0x02000000u, 0x01000000u,
    0x80000000u, 0x40000000u, 0x00200000u, 0x00100000u,
    0x08080800u, 0x04040400u, 0x02020200u, 0x01000000u,
    0x80800080u, 0x00000040u, 0x00200000u, 0x10000000u,
    0x00080000u, 0x00040000u, 0x00020000u, 0x01000000u
},
{
    0x80000000u, 0x40000000u, 0x20000000u, 0x10000000u,
    0x08000000u, 0x04000000u, 0x02000000u, 0x01000000u,
    0x80800000u, 0x40400000u, 0x00200000u, 0x10100000u,
    0x08080000u, 0x04040000u, 0x00020000u, 0x01010000u,
    0x80008000u, 0x40404000u, 0x20202000u, 0x10101000u,
    0x00000800u, 0x00000400u, 0x02020200u, 0x01000100u,
    0x80808080u, 0x40400040u, 0x00200020u, 0x10101010u,
    0x00000808u, 0x00000004u, 0x02020202u, 0x01010101u
},
{
    0x80000000u, 0x40000000u, 0x20000000u, 0x10000000u,
    0x08000000u, 0x04000000u, 0x02000000u, 0x01000000u,
    0x80000000u, 0x00400000u, 0x00200000u, 0x10000000u,
    0x08000000u, 0x00040000u, 0x02000000u, 0x01000000u,
    0x80808000u, 0x40404000u, 0x00200000u, 0x00100000u,
    0x00000800u, 0x00040000u, 0x02020200u, 0x01000000u,
    0x80008080u, 0x00404040u, 0x20202000u, 0x10101000u,
    0x08000808u, 0x04000404u, 0x02000202u, 0x01000101u
},
{
    0x80000000u, 0x40000000u, 0x20000000u, 0x10000000u,
    0x08000000u, 0x04000000u, 0x02000000u, 0x01000000u,
    0x80000000u, 0x40000000u, 0x20000000u, 0x00100000u,
    0x00080000u, 0x04000000u, 0x02000000u, 0x01000000u,
    0x80808000u, 0x40000000u, 0x20202000u, 0x10000000u,
    0x08080800u, 0x04000000u, 0x00000200u, 0x00010000u,
    0x80808000u, 0x40404000u, 0x20000000u, 0x00101010u,
    0x00080000u, 0x04000000u, 0x00020000u, 0x00010000u
},
{
    0x80000000u, 0x40000000u, 0x20000000u, 0x10000000u,
    0x08000000u, 0x04000000u, 0x02000000u, 0x01000000u,
    0x00800000u, 0x80400000u, 0x40200000u, 0x00100000u,
    0x00080000u, 0x08040000u, 0x04020000u, 0x02010000u,
    0x01008000u, 0x00004000u, 0x80402000u, 0x40201000u,
    0x20100800u, 0x10080400u, 0x00000200u, 0x00020100u,
    0x00010080u, 0x00000040u, 0x00804020u, 0x80402010u,
    0x40201008u, 0x00100004u, 0x00080002u, 0x00040201u
},
{
    0x80000000u, 0x40000000u, 0x20000000u, 0x10000000u,
    0x08000000u, 0x04000000u, 0x02000000u, 0x01000000u,
    0x00800000u, 0x80000000u, 0x00200000u, 0x00100000u,
    0x00080000u, 0x08000000u, 0x00020000u, 0x00010000u,
    0x00008000u, 0x00800000u, 0x80000000u, 0x40000000u,
    0x00000800u, 0x00080000u, 0x00000200u, 0x00000100u,
    0x00000080u, 0x00008000u, 0x00800000u, 0x80002010u,
    0x00000008u, 0x20100004u, 0x00000002u, 0x08040200u
},
{
    0x80000000u, 0x40000000u, 0x20000000u, 0x10000000u,
    0x08000000u, 0x04000000u, 0x02000000u, 0x01000000u,
    0x00800000u, 0x80400000u, 0x40200000u, 0x20100000u,
    0x00080000u, 0x08040000u, 0x00020000u, 0x02010000u,
    0x00008000u, 0x00804000u, 0x80402000u, 0x40201000u,
    0x20100800u, 0x10080400u, 0x08000200u, 0x04000100u,
    0x00000080u, 0x00000040u, 0x00800020u, 0x80402010u,
    0x40001008u, 0x20000004u, 0x00080402u, 0x08040201u
},
{
    0x80000000u, 0x40000000u, 0x20000000u, 0x10000000u,
    0x08000000u, 0x04000000u, 0x02000000u, 0x01000000u,
    0x00800000u, 0x80000000u, 0x00200000u, 0x20000000u,
    0x00080000u, 0x08000000u, 0x04000000u, 0x00010000u,
    0x00008000u, 0x00004000u, 0x80000000u, 0x40201000u,
    0x00000800u, 0x00000400u, 0x08000000u, 0x04020100u,
    0x00010000u, 0x00000040u, 0x00804020u, 0x80000000u,
    0x00201008u, 0x00100804u, 0x00000002u, 0x08040001u
},
{
    0x80000000u, 0x40000000u, 0x20000000u, 0x10000000u,
    0x08000000u, 0x04000000u, 0x02000000u, 0x01000000u,
    0x00800000u, 0x80400000u, 0x00200000u, 0x00100000u,
    0x10080000u, 0x08040000u, 0x04020000u, 0x02010000u,
    0x01008000u, 0x00004000u, 0x80002000u, 0x40201000u,
    0x00000800u, 0x00000400u, 0x00040200u, 0x00020100u,
    0x02000080u, 0x01008040u, 0x00800020u, 0x80402010u,
    0x00201008u, 0x20100804u, 0x10080002u, 0x08040001u
},
{
    0x80000000u, 0x40000000u, 0x20000000u, 0x10000000u,
    0x08000000u, 0x04000000u, 0x02000000u, 0x01000000u,
    0x00800000u, 0x00400000u, 0x80200000u, 0x40100000u,
    0x20080000u, 0x00040000u, 0x00020000u, 0x00010000u,
    0x00008000u, 0x00004000u, 0x00002000u, 0x00401000u,
    0x80000800u, 0x40000400u, 0x20080200u, 0x10040100u,
    0x08020080u, 0x04010040u, 0x00000020u, 0x00000010u,
    0x00002008u, 0x00001004u, 0x80000002u, 0x40000401u
},
{
    0x80000000u, 0x40000000u, 0x20000000u, 0x10000000u,
    0x08000000u, 0x04000000u, 0x02000000u, 0x01000000u,
    0x00800000u, 0x00400000u, 0x80000000u, 0x00100000u,
    0x20000000u, 0x00040000u, 0x00020000u, 0x04000000u,
    0x02000000u, 0x00004000u, 0x00002000u, 0x00001000u,
    0x80200800u, 0x40000000u, 0x00080000u, 0x00040000u,
    0x08000000u, 0x04010040u, 0x00008000u, 0x00004000u,
    0x00002000u, 0x00001000u, 0x80000802u, 0x00100401u
}
};

__device__ __forceinline__ float sobol_01(uint32_t index, int dim) {
    uint32_t x = 0u;
    for (int bit = 0; bit < 32; ++bit) {
        if (index & (1u << bit)) x ^= SOBOL_DIR[dim % 32][bit];
    }
    return (x >> 8) * (1.0f / 16777216.0f);
}

__device__ __forceinline__ float sobol_scrambled(uint32_t index, int dim, uint32_t scrambleMask) {
    uint32_t x = 0u;
    for (int bit = 0; bit < 32; ++bit) {
        if (index & (1u << bit)) x ^= SOBOL_DIR[dim % 32][bit];
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