#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ inline glm::vec3 reflectDir(const glm::vec3& wo, const glm::vec3& n) {
    return glm::normalize(glm::reflect(-wo, n));
}

__host__ __device__ inline bool refractDir(const glm::vec3& wo, const glm::vec3& n, float eta, glm::vec3& wt) {
    glm::vec3 wi = -glm::normalize(wo);
    float cosi = glm::dot(n, wi);
    float sint2 = fmaxf(0.f, 1.f - cosi * cosi);
    float eta2 = eta * eta;
    float k = 1.f - eta2 * sint2;
    if (k < 0.f) return false;
    wt = glm::normalize(eta * (-wi) + (eta * cosi - sqrtf(k)) * n);
    return true;
}

__host__ __device__ inline float fresnelSchlick(float cosTheta, float iorExt, float iorInt) {
    float r0 = (iorExt - iorInt) / (iorExt + iorInt);
    r0 = r0 * r0;
    float oneMinus = 1.f - cosTheta;
    return r0 + (1.f - r0) * oneMinus * oneMinus * oneMinus * oneMinus * oneMinus;
}

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    XRNG& rng)
{
    float u1 = rng.next();
    float u2 = rng.next();

    float up = sqrt(u1); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u2 * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    float u1,
    float u2)
{
    float up = sqrt(u1); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u2 * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__device__ void scatterRay(int iter, int depth, int num_paths,
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    XRNG& rng,
    bool useSobol)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    if (m.emittance > 0.f) {
        pathSegment.remainingBounces = 0;
        pathSegment.lastBsdfPdf = 0.f;
        pathSegment.lastWasSpecular = false;
        return;
    }

    glm::vec3 n = glm::normalize(normal);
    glm::vec3 wo = pathSegment.ray.direction;
    float nDotWo = glm::dot(n, wo);
    bool entering = nDotWo < 0.f;
    if (!entering) n = -n;

    // Refract
    if (m.hasRefractive > 0.5f) {
        float etaI = 1.0f;
        float etaT = (m.indexOfRefraction > 0.f) ? m.indexOfRefraction : 1.5f;
        float eta = entering ? (etaI / etaT) : (etaT / etaI);
        float cosThetaI = fminf(1.f, fabsf(glm::dot(n, -wo)));
        float Fr = fresnelSchlick(cosThetaI, etaI, etaT);
        if (rng.next() < Fr) {
            glm::vec3 wr = reflectDir(wo, n);
            pathSegment.ray.origin = intersect + 0.001f * n;
            pathSegment.ray.direction = -wr;
            pathSegment.lastBsdfPdf = 0.f;
            pathSegment.lastWasSpecular = true;
            pathSegment.color *= (m.specular.color.x > 0.f || m.specular.color.y > 0.f || m.specular.color.z > 0.f) ? m.specular.color : m.color;
            pathSegment.remainingBounces--;
            return;
        }
        else {
            glm::vec3 wt;
            bool ok = refractDir(wo, n, eta, wt);
            if (!ok) {
                glm::vec3 wr = reflectDir(wo, n);
                pathSegment.ray.origin = intersect + 0.001f * n;
                pathSegment.ray.direction = -wr;
                pathSegment.lastBsdfPdf = 0.f;
                pathSegment.lastWasSpecular = true;
                pathSegment.color *= (m.specular.color.x > 0.f || m.specular.color.y > 0.f || m.specular.color.z > 0.f) ? m.specular.color : m.color;
                pathSegment.remainingBounces--;
                return;
            }
            pathSegment.ray.origin = intersect - 0.001f * n;
            pathSegment.ray.direction = wt;
            pathSegment.lastBsdfPdf = 0.f;
            pathSegment.lastWasSpecular = true;
            pathSegment.color *= m.color;
            pathSegment.remainingBounces--;
            return;
        }
    }

    // Reflect
    if (m.hasReflective > 0.5f) {
        glm::vec3 wr = reflectDir(wo, n);
        pathSegment.ray.origin = intersect + 0.001f * n;
        pathSegment.ray.direction = -wr;

        pathSegment.lastBsdfPdf = 0.f;
        pathSegment.lastWasSpecular = true;

        glm::vec3 specTint = (m.specular.color.x + m.specular.color.y + m.specular.color.z) > 0.f ? m.specular.color : ((m.color.x + m.color.y + m.color.z) > 0.f ? m.color : glm::vec3(1.f));
        specTint = glm::vec3(1.f);
        pathSegment.color *= specTint;
        pathSegment.remainingBounces--;
        return;
    }
    glm::vec3 newDirection;
    if (useSobol) {
        uint32_t sobolIndex = pathSegment.pixelIndex + iter * num_paths;
        uint32_t s1 = scramble_mask(iter, pathSegment.pixelIndex, depth * 2 + 4);
        uint32_t s2 = scramble_mask(iter, pathSegment.pixelIndex, depth * 2 + 5);
        float u1 = sobol_scrambled((uint32_t)sobolIndex, depth * 2 + 4, s1);
        float u2 = sobol_scrambled((uint32_t)sobolIndex, depth * 2 + 5, s2);

        newDirection = calculateRandomDirectionInHemisphere(normal, u1, u2);
    }
    else {
        newDirection = calculateRandomDirectionInHemisphere(normal, rng);
    }

    pathSegment.ray.origin = intersect + 0.001f * normal;
    pathSegment.ray.direction = glm::normalize(newDirection);

    float cosTheta = glm::max(0.f, glm::dot(normal, pathSegment.ray.direction));
    pathSegment.lastBsdfPdf = cosTheta > 0.f ? (cosTheta / PI) : 0.f;
    pathSegment.lastWasSpecular = false;

    pathSegment.color *= m.color;
    pathSegment.remainingBounces--;
}

__device__ void scatterRayDiffuse(int iter, int depth, int num_paths, PathSegment& pathSegment, glm::vec3 intersect, glm::vec3 normal, const Material& m, XRNG& rng, bool useSobol) {
    glm::vec3 wo = pathSegment.ray.direction;

    glm::vec3 newDirection;
    if (useSobol) {
        uint32_t sobolIndex = pathSegment.pixelIndex + iter * num_paths;
        uint32_t s1 = scramble_mask(iter, pathSegment.pixelIndex, depth * 2 + 4);
        uint32_t s2 = scramble_mask(iter, pathSegment.pixelIndex, depth * 2 + 5);
        float u1 = sobol_scrambled((uint32_t)sobolIndex, depth * 2 + 4, s1);
        float u2 = sobol_scrambled((uint32_t)sobolIndex, depth * 2 + 5, s2);

        newDirection = calculateRandomDirectionInHemisphere(normal, u1, u2);
    }
    else {
        newDirection = calculateRandomDirectionInHemisphere(normal, rng);
    }

    pathSegment.ray.origin = intersect + 0.001f * normal;
    pathSegment.ray.direction = glm::normalize(newDirection);
    float cosTheta = glm::max(0.f, glm::dot(normal, pathSegment.ray.direction));
    pathSegment.lastBsdfPdf = cosTheta > 0.f ? (cosTheta / PI) : 0.f;
    pathSegment.lastWasSpecular = false;
    pathSegment.color *= m.color;
    pathSegment.remainingBounces--;
}

__device__ void scatterRaySpecular(PathSegment& pathSegment, glm::vec3 intersect, glm::vec3 normal, const Material& m, XRNG& rng) {
    glm::vec3 wo = pathSegment.ray.direction;
    glm::vec3 wr = reflectDir(wo, normal);

    pathSegment.ray.origin = intersect + 0.001f * normal;
    pathSegment.ray.direction = -wr;
    pathSegment.lastBsdfPdf = 0.f;
    pathSegment.lastWasSpecular = true;
    glm::vec3 specTint = (m.specular.color.x + m.specular.color.y + m.specular.color.z) > 0.f ? m.specular.color : ((m.color.x + m.color.y + m.color.z) > 0.f ? m.color : glm::vec3(1.f));
    pathSegment.color *= specTint;
    pathSegment.remainingBounces--;
}

__device__ void scatterRayRefractive(PathSegment& pathSegment, glm::vec3 intersect, glm::vec3 normal, const Material& m, XRNG& rng) {
    if (m.hasRefractive < 0.5f) return;
    glm::vec3 wo = pathSegment.ray.direction;

    float nDotWo = glm::dot(normal, wo);
    bool entering = nDotWo < 0.f;
    if (!entering) normal = -normal;
    float etaI = 1.0f;
    float etaT = (m.indexOfRefraction > 0.f) ? m.indexOfRefraction : 1.5f;
    float eta = entering ? (etaI / etaT) : (etaT / etaI);
    float cosThetaI = fminf(1.f, fabsf(glm::dot(normal, -wo)));
    float Fr = fresnelSchlick(cosThetaI, etaI, etaT);

    glm::vec3 wt;
    bool ok = refractDir(wo, normal, eta, wt);

    if (rng.next() < Fr || !ok) {
        scatterRaySpecular(pathSegment, intersect, normal, m, rng);
        return;
    }
    
    pathSegment.ray.origin = intersect - 0.001f * normal;
    pathSegment.ray.direction = wt;
    pathSegment.lastBsdfPdf = 0.f;
    pathSegment.lastWasSpecular = true;
    pathSegment.color *= m.color;
    pathSegment.remainingBounces--;
}

__device__ void scatterRayShadow(int iter, int depth, int num_paths, PathSegment& pathSegment, glm::vec3 intersect, glm::vec3 normal, const Material& m, const Material& lm, const Geom& l, XRNG& rng, int num_lights, int num_shadow_rays, bool useSobol) {
    glm::vec3 wo = pathSegment.ray.direction;

    float nDotWo = glm::dot(normal, wo);
    bool entering = nDotWo < 0.f;
    if (!entering) normal = -normal;

    glm::vec3 lightPoint, lightNormal;
    float u0, u1, u2;
    if (useSobol) {
        uint32_t sobolIndex = pathSegment.pixelIndex + iter * num_paths;
        uint32_t s0 = scramble_mask(iter, pathSegment.pixelIndex, depth * 2 + 4);
        uint32_t s1 = scramble_mask(iter, pathSegment.pixelIndex, depth * 2 + 5);
        uint32_t s2 = scramble_mask(iter, pathSegment.pixelIndex, depth * 2 + 6);
        u0 = sobol_scrambled((uint32_t)sobolIndex, depth * 2 + 4, s0);
        u1 = sobol_scrambled((uint32_t)sobolIndex, depth * 2 + 5, s2);
        u2 = sobol_scrambled((uint32_t)sobolIndex, depth * 2 + 6, s2);
    }
    else {
        u0 = rng.next();
        u1 = rng.next();
        u2 = rng.next();
    }
    samplePointOnLight(l, u0, u1, u2, lightPoint, lightNormal);

    glm::vec3 toLight = lightPoint - intersect;
    float dist2 = glm::dot(toLight, toLight);
    if (dist2 <= 0.f) return;
    glm::vec3 wi = glm::normalize(toLight);

    float cosSurface = glm::max(0.f, glm::dot(normal, wi));
    float cosLight = glm::max(0.f, glm::dot(lightNormal, -wi));

    if (cosSurface <= 0.f || cosLight <= 0.f) return;

    float area = areaOfLight(l);
    if (!(area > 0.f)) return;

    float pdfA = (1.f / num_lights) * (1.f / area);

    float p_light_A = pdfA;
    float p_bsdf_omega = bsdfPdf(m, normal, pathSegment.ray.direction, wi);
    float p_bsdf_A = p_bsdf_omega * (cosLight / dist2);

    float l2 = p_light_A * p_light_A;
    float b2 = p_bsdf_A * p_bsdf_A;
    float w_light = (l2) / (l2 + b2 + 1e-16f);

    glm::vec3 f = m.color * (1.f / PI);
    glm::vec3 Le = lm.color * lm.emittance;
    glm::vec3 L_light = f * Le * (cosSurface * cosLight) / dist2 * (1.f / pdfA);

    pathSegment.ray.origin = intersect + 0.001f * normal;
    pathSegment.ray.direction = wi;
    pathSegment.pixelIndex = pathSegment.pixelIndex;
    pathSegment.remainingBounces = 1;
    pathSegment.color = pathSegment.color;
    pathSegment.shadowDist2 = dist2;
    pathSegment.lastBsdfPdf = 0.f;
    pathSegment.lastWasSpecular = false;
    pathSegment.lightContrib = (w_light * L_light) / float(num_shadow_rays);
}