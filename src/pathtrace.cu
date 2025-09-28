#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include "../stream_compaction/efficient.h"

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "helpers.h"

#define ERRORCHECK 1
#define RR_THRESHOLD 5
#define SHADOW_RAYS 2

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
static PathSegment* dev_paths_compacted = NULL;
static PathSegment* dev_paths_tmp = NULL;
static PathSegment* dev_paths_shadow = NULL;
static PathSegment* dev_paths_shadow_compacted = NULL;
static ShadeableIntersection* dev_intersections_shadow = NULL;
static ShadeableIntersection* dev_intersections_tmp = NULL;

static Geom* dev_lights = NULL;
static int numLights = 0;

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    const int MAX_RAYS = (SHADOW_RAYS) * pixelcount;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, MAX_RAYS * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need
    cudaMalloc(&dev_paths_compacted, pixelcount * sizeof(PathSegment));
    cudaMalloc(&dev_paths_tmp, pixelcount * sizeof(PathSegment));
    cudaMalloc(&dev_paths_shadow, MAX_RAYS * sizeof(PathSegment));
    cudaMalloc(&dev_paths_shadow_compacted, MAX_RAYS * sizeof(PathSegment));
    cudaMalloc(&dev_intersections_shadow, MAX_RAYS * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections_shadow, 0, MAX_RAYS * sizeof(ShadeableIntersection));
    cudaMalloc(&dev_intersections_tmp, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections_tmp, 0, pixelcount * sizeof(ShadeableIntersection));

    std::vector<Geom> lights;
    for (Geom& g : scene->geoms) {
        const Material& m = scene->materials[g.materialid];
        if (m.emittance > 0.f) {
            lights.push_back(g);
        }
    }
    numLights = lights.size();

    if (numLights > 0) {
        cudaMalloc(&dev_lights, numLights * sizeof(Geom));
        cudaMemcpy(dev_lights, lights.data(),
            numLights * sizeof(Geom), cudaMemcpyHostToDevice);
    }
    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
    cudaFree(dev_paths_compacted);
    cudaFree(dev_paths_tmp);
    cudaFree(dev_paths_shadow);
    cudaFree(dev_paths_shadow_compacted);
    cudaFree(dev_intersections_shadow);
    cudaFree(dev_intersections_tmp);

    if (dev_lights) {
        cudaFree(dev_lights);
    }
    dev_lights = NULL;
    numLights = 0;

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, bool antiAliasing, bool stratifiedSampling, int strata)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    int lensRadius = 0;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);
        // TODO: implement antialiasing by jittering the ray
        float jitterX = 0.5f;
        float jitterY = 0.5f;

        if (antiAliasing) {
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            if (stratifiedSampling) {
                int sx, sy;
                stratum_from_iter(iter, sx, sy, strata);
                glm::vec2 rot = cp_rotation(index);
                float jx = (sx + u01(rng)) / float(strata);
                float jy = (sy + u01(rng)) / float(strata);
                jx = jx + rot.x; jx -= floorf(jx);
                jy = jy + rot.y; jy -= floorf(jy);
                jitterX = jx;
                jitterY = jy;
            }
            else {
                jitterX = u01(rng);
                jitterY = u01(rng);
            }
        }

        float px = (float)x + jitterX - (float)cam.resolution.x * 0.5f;
        float py = (float)y + jitterY - (float)cam.resolution.y * 0.5f;

        
        glm::vec3 direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * px
            - cam.up * cam.pixelLength.y * py
        );
        segment.ray.direction = direction;

        if (cam.lensRadius > 0) {
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
            glm::vec2 lensSample = concentricSampleDisk(rng) * cam.lensRadius;
            glm::vec3 lensPoint = cam.position + lensSample.x * cam.right + lensSample.y * cam.up;
            float ft = cam.focalDistance / glm::dot(direction, cam.view);
            glm::vec3 focalPoint = cam.position + direction * ft;
            segment.ray.origin = lensPoint;
            segment.ray.direction = glm::normalize(focalPoint - lensPoint);
        }
        
        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
        segment.isShadowRay = false;
        segment.shadowDist2 = 0.f;
        segment.lightContrib = glm::vec3(0.f);
        segment.lastBsdfPdf = 0.f;
        segment.lastWasSpecular = false;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
            intersections[path_index].geomId = hit_geom_index;
        }
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
__global__ void shadeFakeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials)
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
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
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
    bool russianRoulette,
    bool directLighting,
    bool showBSDFContrib,
    bool showShadowContrib,
    glm::vec3* dev_image)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    PathSegment& pathSegment = pathSegments[idx];
    for (int k = 0; k < num_shadow_rays; ++k) {
        int off = k * num_paths + idx;
        PathSegment dead = {};
        dead.remainingBounces = 0;
        dead.isShadowRay = true;
        dead.pixelIndex = pathSegment.pixelIndex;
        shadowSegments[off] = dead;
    }
    
    if (pathSegment.remainingBounces <= 0) {
        return;
    }

    ShadeableIntersection intersection = shadeableIntersections[idx];
    if (intersection.t < 0.0f) {
        pathSegment.remainingBounces = 0;
        return;
    }

    Material material = materials[intersection.materialId];
    glm::vec3 intersectPoint = getPointOnRay(pathSegment.ray, intersection.t);
    glm::vec3 normal = glm::normalize(intersection.surfaceNormal);

    if (material.emittance > 0.0f && numLights > 0) {
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

    if (directLighting && showShadowContrib && numLights > 0 && material.emittance == 0.0f) {
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegment.remainingBounces);

        glm::vec3 wo = -pathSegment.ray.direction;
        glm::vec3 n = glm::normalize(normal);
        if (glm::dot(n, wo) < 0.f) n = -n;

        for (int k = 0; k < num_shadow_rays; ++k) {
            thrust::uniform_real_distribution<float> u01(0, 1);
            int lightIndex = (int)(u01(rng) * numLights);
            Geom light = lights[lightIndex];
            Material lightMat = materials[light.materialid];

            glm::vec3 lightPoint, lightNormal;
            samplePointOnLight(light, rng, lightPoint, lightNormal);

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

    thrust::default_random_engine rng2 = makeSeededRandomEngine(iter, idx, pathSegment.remainingBounces);
    scatterRay(pathSegment, intersectPoint, normal, material, rng2);

    if (russianRoulette && pathSegment.remainingBounces > RR_THRESHOLD) {
        thrust::uniform_real_distribution<float> u01(0, 1);
        float p = glm::clamp(glm::max(pathSegment.color.r, glm::max(pathSegment.color.g, pathSegment.color.b)), 0.1f, 1.0f);
        if (u01(rng2) > p) {
            pathSegment.remainingBounces = 0;
        }
        else {
            pathSegment.color /= p;
        }
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


// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    bool antiAliasing = false;
    bool stratifiedSampling = false;
    bool sortMaterial = false;
    bool russianRoulette = false;
    bool directLighting = false;
    bool showBSDFContrib = true;
    bool showShadowContrib = true;
    int strata = 4;
    int num_shadow_rays = SHADOW_RAYS;
    if (guiData != nullptr) {
        antiAliasing = guiData->antiAliasing;
        stratifiedSampling = guiData->StratifiedSampling;
        sortMaterial = guiData->SortMaterial;
        russianRoulette = guiData->RussianRoulette;
        directLighting = guiData->DirectLighting;
        showBSDFContrib = guiData->ShowBSDFContrib;
        showShadowContrib = guiData->ShowShadowContrib;
        strata = guiData->Strata;
    }

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths, antiAliasing, stratifiedSampling, strata);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;
    int num_paths_shadow = 0;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, num_paths * sizeof(ShadeableIntersection));
        cudaMemset(dev_intersections_shadow, 0, SHADOW_RAYS * pixelcount * sizeof(ShadeableIntersection));

        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        if (numblocksPathSegmentTracing.x > 0) {
            // tracing
            computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
                depth,
                num_paths,
                dev_paths,
                dev_geoms,
                hst_scene->geoms.size(),
                dev_intersections
                );
            checkCUDAError("trace one bounce");
            cudaDeviceSynchronize();
            

            if (sortMaterial) {
                StreamCompaction::Efficient::radixSort(num_paths, dev_paths, dev_paths_tmp, dev_intersections, dev_intersections_tmp);
            }

            // TODO:
            // --- Shading Stage ---
            // Shade path segments based on intersections and generate new rays by
            // evaluating the BSDF.
            // Start off with just a big kernel that handles all the different
            // materials you have in the scenefile.
            // TODO: compare between directly shading the path segments and shading
            // path segments that have been reshuffled to be contiguous in memory.

            shadeRealMaterial<<<numblocksPathSegmentTracing, blockSize1d>>> (
                iter,
                num_paths,
                num_shadow_rays,
                dev_intersections,
                dev_paths,
                dev_paths_shadow,
                dev_materials,
                dev_lights,
                numLights,
                dev_geoms,
                hst_scene->geoms.size(),
                russianRoulette,
                directLighting,
                showBSDFContrib,
                showShadowContrib,
                dev_image
                );
            checkCUDAError("shade");
            cudaDeviceSynchronize();
        }

        if (directLighting) {
            // Compact shadow rays
            int num_paths_shadow_full = num_shadow_rays * num_paths;
            num_paths_shadow = StreamCompaction::Efficient::compact(num_paths_shadow_full, dev_paths_shadow, dev_paths_shadow_compacted);
            //num_paths_shadow = 0;
            std::swap(dev_paths_shadow, dev_paths_shadow_compacted);

            dim3 numblocksShadowSegmentTracing = (num_paths_shadow + blockSize1d - 1) / blockSize1d;
            if (numblocksShadowSegmentTracing.x > 0) {
                computeIntersections << <numblocksShadowSegmentTracing, blockSize1d >> > (depth, num_paths_shadow, dev_paths_shadow, dev_geoms, hst_scene->geoms.size(), dev_intersections_shadow);
                shadeShadow << <numblocksShadowSegmentTracing, blockSize1d >> > (num_paths_shadow, dev_intersections_shadow, dev_paths_shadow, dev_image);
            }
        }
       
        depth++;


        int newNumPaths = StreamCompaction::Efficient::compact(num_paths, dev_paths, dev_paths_compacted);
        std::swap(dev_paths, dev_paths_compacted);
        num_paths = newNumPaths;

        if (num_paths == 0 || depth >= traceDepth) {
            iterationComplete = true;
        }

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    //finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
