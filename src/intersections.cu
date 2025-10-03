#include "intersections.h"
#include "bvh.h"

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside)
    {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

__device__ float triangleIntersectionTest(
    const Geom& triangle,
    const Ray& r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    Ray q;
    q.origin = multiplyMV(triangle.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(triangle.inverseTransform, glm::vec4(r.direction, 0.0f)));

    glm::vec3 v0 = triangle.v0;
    glm::vec3 v1 = triangle.v1;
    glm::vec3 v2 = triangle.v2;

    glm::vec3 e1 = v1 - v0;
    glm::vec3 e2 = v2 - v0;

    glm::vec3 pvec = glm::cross(q.direction, e2);
    float det = glm::dot(e1, pvec);

    if (fabs(det) < 1e-8f) return -1.f;

    float invDet = 1.f / det;
    glm::vec3 tvec = q.origin - v0;
    float u = glm::dot(tvec, pvec) * invDet;
    if (u < 0.f || u > 1.f) return -1.f;

    glm::vec3 qvec = glm::cross(tvec, e1);
    float v = glm::dot(q.direction, qvec) * invDet;
    if (v < 0.f || u + v > 1.f) return -1.f;

    float t = glm::dot(e2, qvec) * invDet;
    if (t <= 0.f) return -1.f;

    glm::vec3 objSpaceIntersect = getPointOnRay(q, t);
    intersectionPoint = multiplyMV(triangle.transform, glm::vec4(objSpaceIntersect, 1.f));

    glm::vec3 objNormal = glm::normalize(glm::cross(e1, e2));
    normal = glm::normalize(multiplyMV(triangle.invTranspose, glm::vec4(objNormal, 0.f)));

    outside = glm::dot(r.direction, normal) < 0.f;

    return glm::length(r.origin - intersectionPoint);
}

__global__ void computeIntersectionsNaive(int numRays, PathSegment* rayQueue, Geom* geoms, int geoms_size, ShadeableIntersection* intersectionQueue) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numRays) return;

    PathSegment& ray = rayQueue[idx];
    if (ray.remainingBounces <= 0) return;
    ShadeableIntersection intersection;
    intersection.t = -1.0f;
    intersection.materialId = -1;
    intersection.geomId = -1;

    float t_min = FLT_MAX;
    glm::vec3 intersect;
    glm::vec3 normal;
    glm::vec3 tmp_intersect;
    glm::vec3 tmp_normal;
    
    int hitGeomId = -1;
    int hitMatId = -1;
    bool outside = true;

    for (int i = 0; i < geoms_size; i++) {
        float t = -1.0f;

        if (geoms[i].type == SPHERE) {
            t = sphereIntersectionTest(geoms[i], ray.ray, tmp_intersect, tmp_normal, outside);
        }
        else if (geoms[i].type == CUBE) {
            t = boxIntersectionTest(geoms[i], ray.ray, tmp_intersect, tmp_normal, outside);
        }
        else if (geoms[i].type == TRIANGLE) {
            t = triangleIntersectionTest(geoms[i], ray.ray, tmp_intersect, tmp_normal, outside);
        }

        if (t > 0.0f && t < t_min) {
            t_min = t;
            hitGeomId = i;
            hitMatId = geoms[hitGeomId].materialid;
            normal = tmp_normal;
            intersect = tmp_intersect;
        }
    }

    if (hitGeomId != -1) {
        intersection.t = t_min;
        intersection.materialId = hitMatId;
        intersection.surfaceNormal = normal;
        intersection.geomId = hitGeomId;
    }

    intersectionQueue[idx] = intersection;
}

__global__ void computeIntersectionsBVH(int numRays, PathSegment* rayQueue, Geom* geoms, int geoms_size, ShadeableIntersection* intersectionQueue, BVHNode* dev_bvhNodes, int bvhRootIndex) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numRays) return;
    if (bvhRootIndex < 0) return;

    PathSegment& ray = rayQueue[idx];

    ShadeableIntersection intersection;
    intersection.t = -1.0f;
    intersection.materialId = -1;
    intersection.geomId = -1;

    float t_min = FLT_MAX;
    glm::vec3 intersect;
    glm::vec3 normal;
    
    int hitGeomId = -1;
    int hitMatId = -1;

    int stack[64];
    int sp = 0;
    stack[sp++] = bvhRootIndex;

    while (sp) {
        int nodeIdx = stack[--sp];
        const BVHNode& node = dev_bvhNodes[nodeIdx];

        if (!rayAABB(ray.ray, node.bmin, node.bmax, t_min)) continue;
        if (node.isLeaf) {
            int i = (int)node.geomIndex;
            float t;
            glm::vec3 tmp_intersect;
            glm::vec3 tmp_normal;
            bool outside;

            const Geom& g = geoms[i];
            if (g.type == SPHERE) {
                t = sphereIntersectionTest(geoms[i], ray.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (g.type == CUBE) {
                t = boxIntersectionTest(geoms[i], ray.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (g.type == TRIANGLE) {
                t = triangleIntersectionTest(geoms[i], ray.ray, tmp_intersect, tmp_normal, outside);
            }

            if (t > 0.f && t < t_min) {
                t_min = t;
                hitGeomId = i;
                hitMatId = geoms[hitGeomId].materialid;
                normal = tmp_normal;
                intersect = tmp_intersect;
            }
        }
        else {
            stack[sp++] = (int)node.left;
            stack[sp++] = (int)node.right;
        }
    }

    if (hitGeomId != -1) {
        intersection.t = t_min;
        intersection.materialId = hitMatId;
        intersection.surfaceNormal = normal;
        intersection.geomId = hitGeomId;
    }

    intersectionQueue[idx] = intersection;
}

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
    int* queueSizes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    PathSegment ps = dev_queue_rays[idx];
    if (ps.remainingBounces <= 0) return;
    ShadeableIntersection isect = dev_queue_isect[idx];

    if (isect.t < 0.0f) {
        return;
    }

    Material mat = materials[isect.materialId];
    if (ps.isShadowRay) {
        int pos = atomicAdd(&queueSizes[4], 1);
        dev_queue_rays_shadow[pos] = ps;
        dev_queue_isect_shadow[pos] = isect;
        return;
    }

    if (mat.emittance > 0.0f) {
        int pos = atomicAdd(&queueSizes[0], 1);
        dev_queue_rays_emissive[pos] = ps;
        dev_queue_isect_emissive[pos] = isect;
        return;
    }

    if (mat.hasRefractive > 0.5f) {
        int pos = atomicAdd(&queueSizes[1], 1);
        dev_queue_rays_refractive[pos] = ps;
        dev_queue_isect_refractive[pos] = isect;
        return;
    }

    if (mat.hasReflective > 0.5f) {
        int pos = atomicAdd(&queueSizes[2], 1);
        dev_queue_rays_specular[pos] = ps;
        dev_queue_isect_specular[pos] = isect;
        return;
    }

    int pos = atomicAdd(&queueSizes[3], 1);
    dev_queue_rays_diffuse[pos] = ps;
    dev_queue_isect_diffuse[pos] = isect;

    if (!use_shadow_rays) return;
    int posShadow = atomicAdd(&queueSizes[5], num_shadow_rays);
    for (int i = 0; i < num_shadow_rays; i++) {
        dev_queue_rays_shadow_setup[posShadow + i] = ps;
        dev_queue_isect_shadow_setup[posShadow + i] = isect;
    }
}

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
    int* outputCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < emissiveCount) {
        PathSegment ps = emissive[idx];
        if (ps.remainingBounces > 0) {
            int pos = atomicAdd(outputCount, 1);
            output[pos] = ps;
        }
        return;
    }

    int refractiveIdx = idx - emissiveCount;
    if (refractiveIdx >= 0 && refractiveIdx < refractiveCount) {
        PathSegment ps = refractive[refractiveIdx];
        if (ps.remainingBounces > 0) {
            int pos = atomicAdd(outputCount, 1);
            output[pos] = ps;
        }
        return;
    }

    int specularIdx = refractiveIdx - refractiveCount;
    if (specularIdx >= 0 && specularIdx < specularCount) {
        PathSegment ps = specular[specularIdx];
        if (ps.remainingBounces > 0) {
            int pos = atomicAdd(outputCount, 1);
            output[pos] = ps;
        }
        return;
    }

    int diffuseIdx = specularIdx - specularCount;
    if (diffuseIdx >= 0 && diffuseIdx < diffuseCount) {
        PathSegment ps = diffuse[diffuseIdx];
        if (ps.remainingBounces > 0) {
            int pos = atomicAdd(outputCount, 1);
            output[pos] = ps;
        }
        return;
    }

    int shadowIdx = diffuseIdx - diffuseCount;
    if (shadowIdx >= 0 && shadowIdx < shadowCount) {
        PathSegment ps = shadow[shadowIdx];
        if (ps.remainingBounces > 0) {
            int pos = atomicAdd(outputCount, 1);
            output[pos] = ps;
        }
        return;
    }
}