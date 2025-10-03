#pragma once

#include "sceneStructs.h"

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>


/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a)
{
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// CHECKITOUT
/**
 * Compute a point at parameter value `t` on ray `r`.
 * Falls slightly short so that it doesn't intersect the object it's hitting.
 */
__host__ __device__ inline glm::vec3 getPointOnRay(Ray r, float t)
{
    return r.origin + (t - .0001f) * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ inline glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v)
{
    return glm::vec3(m * v);
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside);

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius 0.5 and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside);


__device__ float triangleIntersectionTest(
    const Geom& triangle,
    const Ray& r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside);

__global__ void computeIntersectionsNaive(int numRays, PathSegment* rayQueue, Geom* geoms, int geoms_size, ShadeableIntersection* intersectionQueue);
__global__ void computeIntersectionsBVH(int numRays, PathSegment* rayQueue, Geom* geoms, int geoms_size, ShadeableIntersection* intersectionQueue, BVHNode* dev_bvhNodes, int bvhRootIndex);
__global__ void dispatchQueue(int num_paths,
    PathSegment* dev_queue_rays,
    ShadeableIntersection* dev_queue_isect,
    Material* materials,
    PathSegment* dev_queue_rays_emissive,
    ShadeableIntersection* dev_queue_isect_emissive,
    PathSegment* dev_queue_rays_diffuse,
    ShadeableIntersection* dev_queue_isect_diffuse,
    PathSegment* dev_queue_rays_specular,
    ShadeableIntersection* dev_queue_isect_specular,
    PathSegment* dev_queue_rays_refractive,
    ShadeableIntersection* dev_queue_isect_refractive,
    bool use_shadow_rays,
    int num_shadow_rays,
    PathSegment* dev_queue_rays_shadow,
    ShadeableIntersection* dev_queue_isect_shadow,
    PathSegment* dev_queue_rays_shadow_setup,
    ShadeableIntersection* dev_queue_isect_shadow_setup,
    int* queueSizes);
__global__ void consolidatePaths(
    PathSegment* output,
    PathSegment* emissive,
    PathSegment* refractive,
    PathSegment* specular,
    PathSegment* diffuse,
    PathSegment* shadow,
    int emissiveCount,
    int refractiveCount,
    int specularCount,
    int diffuseCount,
    int shadowCount,
    int* outputCount);