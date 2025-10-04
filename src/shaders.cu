#include "shaders.h"
#include "helpers.h"
#include "interactions.h"
#include "intersections.h"
#include "bvh.h"

#include <thrust/random.h>

#define RR_THRESHOLD 5

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine2(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

__device__ void russianRoulette(PathSegment& path, int depth, int rrThreshold, XRNG& rng) {
    if (depth > rrThreshold) {
        float p = glm::clamp(glm::max(path.color.r, glm::max(path.color.g, path.color.b)), 0.1f, 1.0f);
        if (rng.next() > p) {
            path.remainingBounces = 0;
        }
        path.color /= p;
    }
}


// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial(int iter, int num_paths, ShadeableIntersection* shadeableIntersections, PathSegment* pathSegments, Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
            // Set up the RNG
            // LOOK: this is how you use thrust's RNG! Please look at
            // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine2(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else {
                float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
                pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
                pathSegments[idx].color *= u01(rng); // apply some noise because why not
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            pathSegments[idx].color = glm::vec3(0.0f);
        }
    }
}

// Real shader
__global__ void shadeRealMaterial(
    int iter,
    int num_paths,
    int num_shadow_rays,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    PathSegment* shadowSegments,
    Material* materials,
    Geom* lights,
    int numLights,
    Geom* geoms,
    int geoms_size,
    bool useRR,
    int depthRR,
    bool directLighting,
    bool showBSDFContrib,
    bool showShadowContrib,
    glm::vec3* dev_image,
    bool useSobol)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    PathSegment& pathSegment = pathSegments[idx];
    if (pathSegment.remainingBounces <= 0) {
        return;
    }

    if (directLighting) {
        for (int k = 0; k < num_shadow_rays; ++k) {
            int off = k * num_paths + idx;
            PathSegment dead = {};
            dead.remainingBounces = 0;
            dead.isShadowRay = true;
            dead.pixelIndex = pathSegment.pixelIndex;
            shadowSegments[off] = dead;
        }
    }

    ShadeableIntersection intersection = shadeableIntersections[idx];
    if (intersection.t < 0.0f) {
        pathSegment.remainingBounces = 0;
        return;
    }

    Material material = materials[intersection.materialId];
    glm::vec3 intersectPoint = getPointOnRay(pathSegment.ray, intersection.t);
    glm::vec3 normal = glm::normalize(intersection.surfaceNormal);

    if (material.emittance > 0.0f) {
        if (directLighting && !showBSDFContrib) {
            pathSegment.remainingBounces = 0;
            return;
        }

        glm::vec3 Le = material.color * material.emittance;

        if (pathSegment.lastWasSpecular) {
            atomicAddVec3(dev_image, pathSegment.pixelIndex, pathSegment.color * Le);
            pathSegment.remainingBounces = 0;
            return;
        }

        if (!(pathSegment.lastBsdfPdf > 0.f)) {
            atomicAddVec3(dev_image, pathSegment.pixelIndex, pathSegment.color * Le);
            pathSegment.remainingBounces = 0;
            return;
        }

        if (directLighting && showBSDFContrib) {
            glm::vec3 wi = pathSegment.ray.direction;
            float dist = intersection.t;
            float dist2 = dist * dist;
            float cosLight = glm::max(0.f, glm::dot(normal, -wi));

            int hitGeom = intersection.geomId;
            float area = areaOfLight(geoms[hitGeom]);

            if (area <= 0.f) area = 1.f;

            float pmf_light = (numLights > 0) ? (1.f / float(numLights)) : 1.f;
            float p_light_omega = (cosLight > 0.f) ? (pmf_light * (1.f / area) * (dist2 / fmaxf(cosLight, 1e-6f))) : 0.f;

            float p_bsdf_omega = fmaxf(pathSegment.lastBsdfPdf, 0.f);
            float w_bsdf = 1.f;

            if (p_light_omega > 0.f || p_bsdf_omega > 0.f) {
                float l2 = p_light_omega * p_light_omega;
                float b2 = p_bsdf_omega * p_bsdf_omega;
                w_bsdf = (b2) / (b2 + l2 + 1e-16f);
            }

            atomicAddVec3(dev_image, pathSegment.pixelIndex, pathSegment.color * Le * w_bsdf);
            pathSegment.remainingBounces = 0;
            return;
        }

        atomicAddVec3(dev_image, pathSegment.pixelIndex, pathSegment.color * Le);
        pathSegment.remainingBounces = 0;
        return;
    }

    if (directLighting && showShadowContrib && numLights > 0 && material.hasReflective < 0.5f && material.hasRefractive < 0.5f) {
        XRNG xr(seed_hash(iter, idx, pathSegment.remainingBounces));

        glm::vec3 wo = -pathSegment.ray.direction;
        glm::vec3 n = glm::normalize(normal);
        if (glm::dot(n, wo) < 0.f) n = -n;

        for (int k = 0; k < num_shadow_rays; ++k) {
            int lightIndex = (int)floorf(xr.next() * numLights);
            lightIndex = max(0, min(numLights - 1, lightIndex));

            Geom light = lights[lightIndex];
            Material lightMat = materials[light.materialid];

            glm::vec3 lightPoint, lightNormal;
            float u0, u1, u2;
            if (useSobol) {
                uint32_t sobolIndex = pathSegment.pixelIndex + iter * num_paths;
                uint32_t s0 = scramble_mask(iter, pathSegment.pixelIndex, pathSegment.remainingBounces * 2 + 4);
                uint32_t s1 = scramble_mask(iter, pathSegment.pixelIndex, pathSegment.remainingBounces * 2 + 5);
                uint32_t s2 = scramble_mask(iter, pathSegment.pixelIndex, pathSegment.remainingBounces * 2 + 6);
                u0 = sobol_scrambled((uint32_t)sobolIndex, pathSegment.remainingBounces * 2 + 4, s0);
                u1 = sobol_scrambled((uint32_t)sobolIndex, pathSegment.remainingBounces * 2 + 5, s2);
                u2 = sobol_scrambled((uint32_t)sobolIndex, pathSegment.remainingBounces * 2 + 5, s2);
            }
            else {
                u0 = xr.next();
                u1 = xr.next();
                u2 = xr.next();
            }
            samplePointOnLight(light, u0, u1, u2, lightPoint, lightNormal);

            glm::vec3 toLight = lightPoint - intersectPoint;
            float dist2 = glm::dot(toLight, toLight);
            if (dist2 <= 0.f) continue;
            glm::vec3 wi = glm::normalize(toLight);

            float cosSurface = glm::max(0.f, glm::dot(n, wi));
            float cosLight = glm::max(0.f, glm::dot(lightNormal, -wi));

            if (cosSurface <= 0.f || cosLight <= 0.f) continue;

            float area = areaOfLight(light);
            if (!(area > 0.f)) continue;

            float pdfA = (1.f / numLights) * (1.f / area);

            float p_light_A = pdfA;
            float p_bsdf_omega = bsdfPdf(material, normal, pathSegment.ray.direction, wi);
            float p_bsdf_A = p_bsdf_omega * (cosLight / dist2);

            float l2 = p_light_A * p_light_A;
            float b2 = p_bsdf_A * p_bsdf_A;
            float w_light = (l2) / (l2 + b2 + 1e-16f);

            glm::vec3 f = material.color * (1.f / PI);
            glm::vec3 Le = lightMat.color * lightMat.emittance;
            glm::vec3 L_light = f * Le * (cosSurface * cosLight) / dist2 * (1.f / pdfA);

            PathSegment shadow;
            shadow.ray.origin = intersectPoint + 0.001f * normal;
            shadow.ray.direction = wi;
            shadow.pixelIndex = pathSegment.pixelIndex;
            shadow.remainingBounces = 1;
            shadow.isShadowRay = true;
            shadow.color = pathSegment.color;
            shadow.shadowDist2 = dist2;
            shadow.lastBsdfPdf = 0.f;
            shadow.lastWasSpecular = false;
            shadow.lightContrib = (w_light * L_light) / float(num_shadow_rays);

            shadowSegments[k * num_paths + idx] = shadow;
        }
    }

    XRNG rng2(seed_hash(iter, idx, pathSegment.remainingBounces));
    scatterRay(iter, pathSegment.remainingBounces, num_paths, pathSegment, intersectPoint, normal, material, rng2, useSobol);
    if (useRR) {
        russianRoulette(pathSegment, pathSegment.remainingBounces, depthRR, rng2);
    }
}

__global__ void shadeShadow(int P_shadow, const ShadeableIntersection* shadeableIntersections, PathSegment* shadowSegments, glm::vec3* dev_image) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P_shadow) return;

    PathSegment& shadow = shadowSegments[idx];
    if (shadow.remainingBounces <= 0) {
        return;
    }

    float tLight = sqrtf(shadow.shadowDist2);
    float eps = fmaxf(1e-4f, 1e-4f * tLight);

    if (shadeableIntersections[idx].t < 0.f || shadeableIntersections[idx].t >= tLight - eps) {
        atomicAddVec3(dev_image, shadow.pixelIndex, shadow.color * shadow.lightContrib);
    }
    shadow.remainingBounces = 0;
}

__global__ void shadeEmissive(int num_paths, PathSegment* pathSegments, ShadeableIntersection* intersections, Material* materials, Geom* geoms, glm::vec3* dev_image, int num_lights, bool MIS) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    PathSegment& ps = pathSegments[idx];
    ShadeableIntersection isect = intersections[idx];
    Material mat = materials[isect.materialId];

    if (mat.emittance <= 0.f) return;
    
    glm::vec3 Le = mat.color * mat.emittance;
    if (!(ps.lastBsdfPdf > 0.f)) {
        atomicAddVec3(dev_image, ps.pixelIndex, ps.color * Le);
        ps.remainingBounces = 0;
        return;
    }

    if (MIS) {
        glm::vec3 wi = ps.ray.direction;
        glm::vec3 normal = glm::normalize(isect.surfaceNormal);
        float dist = isect.t;
        float dist2 = dist * dist;
        float cosLight = glm::max(0.f, glm::dot(normal, -wi));

        int hitGeom = isect.geomId;
        float area = areaOfLight(geoms[hitGeom]);
        if (area <= 0.f) area = 1.f;

        float pmf_light = (num_lights > 0) ? (1.f / float(num_lights)) : 1.f;
        float p_light_omega = (cosLight > 0.f) ? (pmf_light * (1.f / area) * (dist2 / fmaxf(cosLight, 1e-6f))) : 0.f;
        float p_bsdf_omega = fmaxf(ps.lastBsdfPdf, 0.f);

        float w_bsdf = 1.f;
        if (p_light_omega > 0.f || p_bsdf_omega > 0.f) {
            float l2 = p_light_omega * p_light_omega;
            float b2 = p_bsdf_omega * p_bsdf_omega;
            w_bsdf = (b2) / (b2 + l2 + 1e-16f);
        }

        atomicAddVec3(dev_image, ps.pixelIndex, ps.color * Le * w_bsdf);
        ps.remainingBounces = 0;
        return;
    }

    atomicAddVec3(dev_image, ps.pixelIndex, ps.color * Le);
    ps.remainingBounces = 0;
}

__global__ void shadeDiffuse(int iter, int depth, int num_paths, PathSegment* pathSegments, ShadeableIntersection* intersections, Material* materials, bool useRR, int depthRR, bool useSobol) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    PathSegment& ps = pathSegments[idx];
    ShadeableIntersection isect = intersections[idx];
    Material mat = materials[isect.materialId];
    glm::vec3 intersectPoint = getPointOnRay(ps.ray, isect.t);
    glm::vec3 normal = glm::normalize(isect.surfaceNormal);
    XRNG rng(seed_hash(iter, idx, ps.remainingBounces));

    scatterRayDiffuse(iter, depth, num_paths, ps, intersectPoint, normal, mat, rng, useSobol);
    if (useRR) {
        russianRoulette(ps, depth, depthRR, rng);
    }
}

__global__ void shadeSpecular(int iter, int depth, int num_paths, PathSegment* pathSegments, ShadeableIntersection* intersections, Material* materials, bool useRR, int depthRR) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    PathSegment& ps = pathSegments[idx];
    ShadeableIntersection isect = intersections[idx];
    Material& mat = materials[isect.materialId];
    glm::vec3 intersectPoint = getPointOnRay(ps.ray, isect.t);
    glm::vec3 normal = glm::normalize(isect.surfaceNormal);
    XRNG rng(seed_hash(iter, idx, ps.remainingBounces));

    scatterRaySpecular(ps, intersectPoint, normal, mat, rng);
    if (useRR) {
        russianRoulette(ps, depth, depthRR, rng);
    }
}

__global__ void shadeRefractive(int iter, int depth, int num_paths, PathSegment* pathSegments, ShadeableIntersection* intersections, Material* materials, bool useRR, int depthRR) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    PathSegment& ps = pathSegments[idx];
    ShadeableIntersection isect = intersections[idx];
    Material mat = materials[isect.materialId];
    glm::vec3 intersectPoint = getPointOnRay(ps.ray, isect.t);
    glm::vec3 normal = glm::normalize(isect.surfaceNormal);
    XRNG rng(seed_hash(iter, idx, ps.remainingBounces));

    scatterRayRefractive(ps, intersectPoint, normal, mat, rng);
    if (useRR) {
        russianRoulette(ps, depth, depthRR, rng);
    }
}

__global__ void shadeShadowSetup(int iter, int depth, int num_paths, PathSegment* pathSegments, ShadeableIntersection* intersections, Material* materials, Geom* lights, int num_lights, int num_shadow_rays, bool useSobol) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    PathSegment& ps = pathSegments[idx];
    ShadeableIntersection isect = intersections[idx];
    Material mat = materials[isect.materialId];
    glm::vec3 intersectPoint = getPointOnRay(ps.ray, isect.t);
    glm::vec3 normal = glm::normalize(isect.surfaceNormal);
    XRNG rng(seed_hash(iter, idx, ps.remainingBounces));

    int lightIndex = (int)floorf(rng.next() * num_lights);
    lightIndex = max(0, min(num_lights - 1, lightIndex));
    Geom light = lights[lightIndex];
    Material lightMat = materials[light.materialid];

    ps.isShadowRay = true;
    ps.remainingBounces = 0;
    scatterRayShadow(iter, depth, num_paths, ps, intersectPoint, normal, mat, lightMat, light, rng, num_lights, num_shadow_rays, useSobol);
}

__global__ void shadeShadowAccum(int iter, int depth, int num_paths, PathSegment* pathSegments, ShadeableIntersection* intersections, glm::vec3* dev_image) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    PathSegment& ps = pathSegments[idx];
    ShadeableIntersection& isect = intersections[idx];
    float tLight = sqrtf(ps.shadowDist2);
    float eps = fmaxf(1e-4f, 1e-4f * tLight);

    if (isect.t < 0.f || isect.t >= tLight - eps) {
        atomicAddVec3(dev_image, ps.pixelIndex, ps.color * ps.lightContrib);
    }
}