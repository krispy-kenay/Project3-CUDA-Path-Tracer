#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

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

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    XRNG& rng)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    glm::vec3 newDirection = calculateRandomDirectionInHemisphere(normal, rng);

    pathSegment.ray.origin = intersect + 0.001f * normal;
    pathSegment.ray.direction = glm::normalize(newDirection);

    float cosTheta = glm::max(0.f, glm::dot(normal, pathSegment.ray.direction));
    pathSegment.lastBsdfPdf = cosTheta > 0.f ? (cosTheta / PI) : 0.f;
    pathSegment.lastWasSpecular = false;

    pathSegment.color *= m.color;
    pathSegment.remainingBounces--;
}
