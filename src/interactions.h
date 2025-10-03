#pragma once

#include "sceneStructs.h"
#include "helpers.h"
#include <glm/glm.hpp>

#include <thrust/random.h>

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal, 
    XRNG& rng);
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    float u1,
    float u2);

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 *
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 *
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */
__device__ void scatterRay(
    int iter, int depth, int num_paths,
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    XRNG& rng,
    bool useSobol);

__device__ void scatterRayDiffuse(int iter, int depth, int num_paths, PathSegment& pathSegment, glm::vec3 intersect, glm::vec3 normal, const Material& m, XRNG& rng, bool useSobol);
__device__ void scatterRaySpecular(PathSegment& pathSegment, glm::vec3 intersect, glm::vec3 normal, const Material& m, XRNG& rng);
__device__ void scatterRayRefractive(PathSegment& pathSegment, glm::vec3 intersect, glm::vec3 normal, const Material& m, XRNG& rng);
__device__ void scatterRayShadow(int iter, int depth, int num_paths, PathSegment& pathSegment, glm::vec3 intersect, glm::vec3 normal, const Material& m, const Material& lm, const Geom& l, XRNG& rng, int num_lights, int num_shadow_rays, bool useSobol);