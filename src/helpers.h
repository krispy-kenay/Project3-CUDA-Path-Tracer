#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include "scene.h"
#include "sceneStructs.h"
#include "utilities.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"

__device__ glm::vec2 concentricSampleDisk(thrust::default_random_engine& rng) {
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
    unsigned int h1 = utilhash(pixelIndex * 9781u + 0x68bc21ebu);
    unsigned int h2 = utilhash(pixelIndex * 6271u + 0x2c1b3c6du);
    float rx = (h1 & 0x00FFFFFF) / float(0x01000000);
    float ry = (h2 & 0x00FFFFFF) / float(0x01000000);
    return glm::vec2(rx, ry);
}

__device__ void atomicAddVec3(glm::vec3* image, int pixelIndex, const glm::vec3& contrib) {
    atomicAdd(&(image[pixelIndex].x), contrib.x);
    atomicAdd(&(image[pixelIndex].y), contrib.y);
    atomicAdd(&(image[pixelIndex].z), contrib.z);
}

__device__ void samplePointOnLight(
    const Geom& light,
    thrust::default_random_engine& rng,
    glm::vec3& point,
    glm::vec3& normal)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    if (light.type == SPHERE) {
        float z = 2.f * u01(rng) - 1.f;
        float phi = 2.f * PI * u01(rng);
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
        float xi = u01(rng) * totalArea;
        int face = 0;
        float accum = faceAreas[0];
        while (xi > accum && face < 5) {
            face++;
            accum += faceAreas[face];
        }

        float u = u01(rng) * 2.f - 1.f;
        float v = u01(rng) * 2.f - 1.f;
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
}

__device__ float areaOfLight(const Geom& light) {
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

__device__ float bsdfPdf(const Material& m,
    const glm::vec3& n,
    const glm::vec3& wo,
    const glm::vec3& wi)
{
    float cosS = fmaxf(0.f, glm::dot(n, wi));
    return cosS / PI;
}