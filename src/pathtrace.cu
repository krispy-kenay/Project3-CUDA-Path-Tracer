#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <numeric>
#include <chrono>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include "../stream_compaction/efficient.h"

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "helpers.h"
#include "bvh.h"
#include "shaders.h"

#define ERRORCHECK 1
#define NUM_QUEUE 6

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

struct GetMaterialId {
    __host__ __device__
        int operator()(const ShadeableIntersection& isect) const {
        return isect.materialId;
    }
};

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
static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
static PathSegment* dev_paths_compacted = NULL;
static PathSegment* dev_paths_tmp = NULL;
static PathSegment* dev_paths_shadow = NULL;
static PathSegment* dev_paths_shadow_compacted = NULL;
static ShadeableIntersection* dev_intersections_shadow = NULL;
static ShadeableIntersection* dev_intersections_tmp = NULL;

// General
static PathSegment* dev_queue_rays = NULL;
static PathSegment* dev_queue_rays_next = NULL;
static ShadeableIntersection* dev_queue_isect = NULL;
// Emissive
static PathSegment* dev_queue_rays_emissive = NULL;
static ShadeableIntersection* dev_queue_isect_emissive = NULL;
// Diffuse
static PathSegment* dev_queue_rays_diffuse = NULL;
static ShadeableIntersection* dev_queue_isect_diffuse = NULL;
// Specular
static PathSegment* dev_queue_rays_specular = NULL;
static ShadeableIntersection* dev_queue_isect_specular = NULL;
// Refractive
static PathSegment* dev_queue_rays_refract = NULL;
static ShadeableIntersection* dev_queue_isect_refract = NULL;
// Shadow
static PathSegment* dev_queue_rays_shadow = NULL;
static ShadeableIntersection* dev_queue_isect_shadow = NULL;
static PathSegment* dev_queue_rays_shadow_setup = NULL;
static ShadeableIntersection* dev_queue_isect_shadow_setup = NULL;

int* dev_queue_sizes;
int* dev_output_count;

// BVH
static BVHNode* dev_bvhNodes = nullptr;
static unsigned int* dev_ready = nullptr;
static unsigned int* dev_codes = nullptr;
static unsigned int* dev_idx = nullptr;
static int bvhNodeCount = 0;
static int bvhLeafCount = 0;
static int bvhRootIndex = -1;
static bool dev_bvh_built = false;

static Geom* dev_lights = NULL;
static int numLights = 0;

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}


void pathtraceInit(Scene* scene, bool useBVH, bool wavefront, int numShadowRays)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    const int MAX_RAYS = (numShadowRays) * pixelcount;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

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

    int generalQueueLength = (wavefront && guiData->DirectLighting) ? (numShadowRays + 1) * pixelcount : pixelcount;
    cudaMalloc(&dev_queue_rays, generalQueueLength * sizeof(PathSegment));
    cudaMalloc(&dev_queue_rays_next, generalQueueLength * sizeof(PathSegment));
    cudaMalloc(&dev_queue_isect, generalQueueLength * sizeof(ShadeableIntersection));

    if (wavefront) {
        cudaMalloc(&dev_queue_rays_emissive, pixelcount * sizeof(PathSegment));
        cudaMalloc(&dev_queue_isect_emissive, pixelcount * sizeof(ShadeableIntersection));

        cudaMalloc(&dev_queue_rays_diffuse, pixelcount * sizeof(PathSegment));
        cudaMalloc(&dev_queue_isect_diffuse, pixelcount * sizeof(ShadeableIntersection));

        cudaMalloc(&dev_queue_rays_specular, pixelcount * sizeof(PathSegment));
        cudaMalloc(&dev_queue_isect_specular, pixelcount * sizeof(ShadeableIntersection));

        cudaMalloc(&dev_queue_rays_refract, pixelcount * sizeof(PathSegment));
        cudaMalloc(&dev_queue_isect_refract, pixelcount * sizeof(ShadeableIntersection));

        cudaMalloc(&dev_queue_rays_shadow, numShadowRays * pixelcount * sizeof(PathSegment));
        cudaMalloc(&dev_queue_isect_shadow, numShadowRays * pixelcount * sizeof(ShadeableIntersection));
        cudaMalloc(&dev_queue_rays_shadow_setup, numShadowRays * pixelcount * sizeof(PathSegment));
        cudaMalloc(&dev_queue_isect_shadow_setup, numShadowRays * pixelcount * sizeof(ShadeableIntersection));

        cudaMalloc(&dev_queue_sizes, NUM_QUEUE * sizeof(int));
        cudaMalloc(&dev_output_count, sizeof(int));
    }
    
    if (useBVH) {
        const unsigned int N = (unsigned int)scene->geoms.size();

        bvhLeafCount = (int)N;
        bvhNodeCount = (N == 0) ? 0 : (int)(2u * N - 1u);

        cudaMalloc(&dev_codes, N * sizeof(unsigned int));
        cudaMalloc(&dev_idx, N * sizeof(unsigned int));
        cudaMalloc(&dev_bvhNodes, bvhNodeCount * sizeof(BVHNode));
        cudaMalloc(&dev_ready, bvhNodeCount * sizeof(unsigned int));

        buildLBVH_host(hst_scene->geoms, dev_bvh_built, bvhLeafCount, bvhNodeCount, bvhRootIndex, dev_codes, dev_idx, dev_ready, dev_geoms, dev_bvhNodes);
    }

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

void pathtraceFree(bool useBVH, bool wavefront, int numShadowRays)
{
    cudaFree(dev_image);  // no-op if dev_image is null
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

    cudaFree(dev_queue_rays);
    cudaFree(dev_queue_rays_next);
    cudaFree(dev_queue_isect);

    if (wavefront) {
        cudaFree(dev_queue_rays_emissive);
        cudaFree(dev_queue_isect_emissive);

        cudaFree(dev_queue_rays_diffuse);
        cudaFree(dev_queue_isect_diffuse);

        cudaFree(dev_queue_rays_specular);
        cudaFree(dev_queue_isect_specular);

        cudaFree(dev_queue_rays_refract);
        cudaFree(dev_queue_isect_refract);

        cudaFree(dev_queue_rays_shadow);
        cudaFree(dev_queue_isect_shadow);
        cudaFree(dev_queue_rays_shadow_setup);
        cudaFree(dev_queue_isect_shadow_setup);

        cudaFree(dev_queue_sizes);
        cudaFree(dev_output_count);
    }

    if (dev_bvh_built) {
        cudaFree(dev_codes);
        cudaFree(dev_idx);
        cudaFree(dev_bvhNodes);
        cudaFree(dev_ready);
        dev_bvh_built = false; bvhNodeCount = bvhLeafCount = 0; bvhRootIndex = -1;
    }

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
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, bool antiAliasing, bool stratifiedSampling, int strata, bool sobolSampling)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        float jitterX = 0.5f;
        float jitterY = 0.5f;

        if (antiAliasing) {
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            // Sampling choice
            float u, v;
            if (sobolSampling) {
                uint32_t sobolIndex = index + iter * (cam.resolution.x * cam.resolution.y);
                u = sobol_scrambled(sobolIndex, 0, scramble_mask(iter, index, 0));
                v = sobol_scrambled(sobolIndex, 1, scramble_mask(iter, index, 1));
            }
            else {
                u = u01(rng);
                v = u01(rng);
            }
            
            float jx = u;
            float jy = v;

            if (stratifiedSampling) {
                int sx, sy;
                stratum_from_iter(iter, sx, sy, strata);
                jx = (sx + jx) / float(strata);
                jy = (sy + jy) / float(strata);

                glm::vec2 rot = cp_rotation(index);
                jx = jx + rot.x; jx -= floorf(jx);
                jy = jy + rot.y; jy -= floorf(jy);
            }

            if (sobolSampling) {
                float sx_rot = uint_to_unit_float(seed_hash(index, 17, 0));
                float sy_rot = uint_to_unit_float(seed_hash(index, 29, 0));
                jx = cp_rotate(jx, sx_rot);
                jy = cp_rotate(jy, sy_rot);
            }

            jitterX = jx;
            jitterY = jy;
        }

        float px = (float)x + jitterX - (float)cam.resolution.x * 0.5f;
        float py = (float)y + jitterY - (float)cam.resolution.y * 0.5f;

        
        glm::vec3 direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * px
            - cam.up * cam.pixelLength.y * py
        );
        segment.ray.direction = direction;

        if (cam.lensRadius > 0) {
            glm::vec2 lensSample;
            glm::vec3 lensPoint;
            if (sobolSampling) {
                uint32_t sobolIndex = index + iter * (cam.resolution.x * cam.resolution.y);
                float u = sobol_scrambled(sobolIndex, 2, scramble_mask(iter, index, 2));
                float v = sobol_scrambled(sobolIndex, 3, scramble_mask(iter, index, 3));
                float sx = 2.0f * u - 1.0f;
                float sy = 2.0f * v - 1.0f;
                float r, theta;
                if (sx == 0 && sy == 0) {
                    r = 0.0f; theta = 0.0f;
                }
                else {
                    float ax = fabsf(sx), ay = fabsf(sy);
                    if (ax > ay) { r = ax; theta = (PI / 4.0f) * (sy / sx); }
                    else { r = ay; theta = (PI / 2.0f) - (PI / 4.0f) * (sx / sy); }
                }
                float dx = r * cosf(theta);
                float dy = r * sinf(theta);
                lensSample = glm::vec2(dx, dy) * cam.lensRadius;
                lensPoint = cam.position + lensSample.x * cam.right + lensSample.y * cam.up;
            }
            else {
                thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
                lensSample = concentricSampleDisk(rng) * cam.lensRadius;
                lensPoint = cam.position + lensSample.x * cam.right + lensSample.y * cam.up;
            }
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




void pathtraceWave(uchar4* pbo, int frame, int iter) {
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    const int blockSize1d = 128;

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_queue_rays, guiData->antiAliasing, guiData->StratifiedSampling, guiData->Strata, guiData->SobolSampling);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_rays_end = dev_queue_rays + pixelcount;
    int num_rays = dev_rays_end - dev_queue_rays;

    bool iterationComplete = false;
    while (!iterationComplete) {
        
        cudaMemset(dev_queue_sizes, 0, NUM_QUEUE * sizeof(int));

        dim3 numblocksPathSegmentTracing = (num_rays + blockSize1d - 1) / blockSize1d;
        if (numblocksPathSegmentTracing.x <= 0) {
            break;
        }

        if (!guiData->useBVH) {
            computeIntersectionsNaive<<<numblocksPathSegmentTracing, blockSize1d>>>(num_rays, dev_queue_rays, dev_geoms, hst_scene->geoms.size(), dev_queue_isect);
        }
        else {
            computeIntersectionsBVH<<<numblocksPathSegmentTracing, blockSize1d>>>(num_rays, dev_queue_rays, dev_geoms, hst_scene->geoms.size(), dev_queue_isect, dev_bvhNodes, bvhRootIndex);
        }
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        
        dispatchQueue<<<numblocksPathSegmentTracing, blockSize1d>>>(num_rays, dev_queue_rays, dev_queue_isect, dev_materials, dev_queue_rays_emissive, dev_queue_isect_emissive, dev_queue_rays_diffuse, dev_queue_isect_diffuse, dev_queue_rays_specular, dev_queue_isect_specular, dev_queue_rays_refract, dev_queue_isect_refract, guiData->DirectLighting, guiData->NumShadowRays, dev_queue_rays_shadow, dev_queue_isect_shadow, dev_queue_rays_shadow_setup, dev_queue_isect_shadow_setup, dev_queue_sizes);
        checkCUDAError("dispatcher");
        cudaDeviceSynchronize();

        int h_queueSizes[NUM_QUEUE];
        cudaMemcpy(h_queueSizes, dev_queue_sizes, NUM_QUEUE * sizeof(int), cudaMemcpyDeviceToHost);
        int emissiveCount = h_queueSizes[0];
        int refractiveCount = h_queueSizes[1];
        int specularCount = h_queueSizes[2];
        int diffuseCount = h_queueSizes[3];
        int shadowCount = h_queueSizes[4];
        int shadowSetupCount = h_queueSizes[5];

        if (emissiveCount > 0 && guiData->showEmissive) {
            dim3 blocks((emissiveCount + blockSize1d - 1) / blockSize1d);
            shadeEmissive<<<blocks, blockSize1d >>>(emissiveCount, dev_queue_rays_emissive, dev_queue_isect_emissive, dev_materials, dev_geoms, dev_image, numLights, guiData->DirectLighting);
        }
        else {
            emissiveCount = 0;
        }

        if (refractiveCount > 0 && guiData->showRefractive) {
            dim3 blocks((refractiveCount + blockSize1d - 1) / blockSize1d);
            shadeRefractive<<<blocks, blockSize1d >>>(iter, depth, refractiveCount, dev_queue_rays_refract, dev_queue_isect_refract, dev_materials, guiData->RussianRoulette, guiData->depthRussianRoulette);
        }
        else {
            refractiveCount = 0;
        }

        if (specularCount > 0 && guiData->showSpecular) {
            dim3 blocks((specularCount + blockSize1d - 1) / blockSize1d);
            shadeSpecular<<<blocks, blockSize1d >>>(iter, depth, specularCount, dev_queue_rays_specular, dev_queue_isect_specular, dev_materials, guiData->RussianRoulette, guiData->depthRussianRoulette);
        }
        else {
            specularCount = 0;
        }

        if (diffuseCount > 0 && guiData->showDiffuse) {
            dim3 blocks((diffuseCount + blockSize1d - 1) / blockSize1d);
            shadeDiffuse<<<blocks, blockSize1d >>>(iter, depth, diffuseCount, dev_queue_rays_diffuse, dev_queue_isect_diffuse, dev_materials, guiData->RussianRoulette, guiData->depthRussianRoulette, guiData->SobolSampling);
        }
        else {
            diffuseCount = 0;
        }

        if (guiData->DirectLighting) {
            if (shadowCount > 0 && guiData->showShadows) {
                dim3 blocks((shadowCount + blockSize1d - 1) / blockSize1d);
                shadeShadowAccum<<<blocks, blockSize1d>>>(iter, depth, shadowCount, dev_queue_rays_shadow, dev_queue_isect_shadow, dev_image);
            }
            else {
                shadowCount = 0;
            }
            if (shadowSetupCount > 0) {
                dim3 blocks((shadowSetupCount + blockSize1d - 1) / blockSize1d);
                shadeShadowSetup<<<blocks, blockSize1d>>>(iter, depth, shadowSetupCount, dev_queue_rays_shadow_setup, dev_queue_isect_shadow_setup, dev_materials, dev_lights, numLights, guiData->NumShadowRays, guiData->SobolSampling);
            }
        }
        checkCUDAError("shader");
        cudaDeviceSynchronize();

        int totalQueued = emissiveCount + refractiveCount + specularCount + diffuseCount + shadowCount + shadowSetupCount;
        cudaMemset(dev_output_count, 0, sizeof(int));
        if (totalQueued > 0) {
            int blocks = (totalQueued + blockSize1d - 1) / blockSize1d;
            consolidatePaths<<<blocks, blockSize1d>>>(dev_queue_rays_next, dev_queue_rays_emissive, dev_queue_rays_refract, dev_queue_rays_specular, dev_queue_rays_diffuse, dev_queue_rays_shadow_setup, emissiveCount, refractiveCount, specularCount, diffuseCount, shadowSetupCount, dev_output_count);
            checkCUDAError("consolidate paths");

            int newCount;
            cudaMemcpy(&newCount, dev_output_count, sizeof(int), cudaMemcpyDeviceToHost);
            num_rays = newCount;
            std::swap(dev_queue_rays, dev_queue_rays_next);
        }
        checkCUDAError("consolidation");
        cudaDeviceSynchronize();

        depth++;
        if (num_rays == 0 || depth >= traceDepth) {
            iterationComplete = true;
        }
    }

    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}

void pathtraceNaive(uchar4* pbo, int frame, int iter)
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

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_queue_rays, guiData->antiAliasing, guiData->StratifiedSampling, guiData->Strata, guiData->SobolSampling);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_queue_rays + pixelcount;
    int num_paths = dev_path_end - dev_queue_rays;
    int num_paths_shadow = 0;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, num_paths * sizeof(ShadeableIntersection));
        cudaMemset(dev_intersections_shadow, 0, guiData->NumShadowRays * pixelcount * sizeof(ShadeableIntersection));

        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        if (numblocksPathSegmentTracing.x > 0) {
            auto startIntersect = std::chrono::high_resolution_clock::now();
            // tracing
            if (!guiData->useBVH) {
                computeIntersectionsNaive<<<numblocksPathSegmentTracing, blockSize1d>>>(num_paths, dev_queue_rays, dev_geoms, hst_scene->geoms.size(), dev_queue_isect);
                checkCUDAError("trace one bounce");
                cudaDeviceSynchronize();
            }
            else {
                computeIntersectionsBVH<<<numblocksPathSegmentTracing, blockSize1d>>>(num_paths, dev_queue_rays, dev_geoms, hst_scene->geoms.size(), dev_queue_isect, dev_bvhNodes, bvhRootIndex);
                checkCUDAError("trace one bounce");
                cudaDeviceSynchronize();
            }
            auto endIntersect = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float, std::milli> durationIntersect = endIntersect - startIntersect;
            if (iter == 1 && guiData->printDebugStats) { printf("Intersect: %f ms, ", durationIntersect.count()); }
            
            auto startSort = std::chrono::high_resolution_clock::now();
            if (guiData->SortMaterial) {
                //StreamCompaction::Efficient::radixSort(num_paths, dev_queue_rays, dev_paths_tmp, dev_queue_isect, dev_intersections_tmp);
                thrust::device_vector<int> matKeys(num_paths);
                thrust::device_ptr<ShadeableIntersection> isect_ptr(dev_queue_isect);
                thrust::device_ptr<PathSegment>           rays_ptr(dev_queue_rays);
                thrust::transform(isect_ptr, isect_ptr + num_paths, matKeys.begin(), GetMaterialId());
            }
            auto endSort = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float, std::milli> durationSort = endSort - startSort;
            if (iter == 1 && guiData->printDebugStats) { printf("Sort: %f ms, ", durationSort.count()); }

            // TODO:
            // --- Shading Stage ---
            // Shade path segments based on intersections and generate new rays by
            // evaluating the BSDF.
            // Start off with just a big kernel that handles all the different
            // materials you have in the scenefile.
            // TODO: compare between directly shading the path segments and shading
            // path segments that have been reshuffled to be contiguous in memory.

            auto startShade = std::chrono::high_resolution_clock::now();
            shadeRealMaterial<<<numblocksPathSegmentTracing, blockSize1d>>> (
                iter,
                num_paths,
                guiData->NumShadowRays,
                dev_queue_isect,
                dev_queue_rays,
                dev_paths_shadow,
                dev_materials,
                dev_lights,
                numLights,
                dev_geoms,
                hst_scene->geoms.size(),
                guiData->RussianRoulette,
                guiData->depthRussianRoulette,
                guiData->DirectLighting,
                guiData->showDiffuse,
                guiData->showShadows,
                dev_image,
                guiData->SobolSampling
                );
            checkCUDAError("shade");
            cudaDeviceSynchronize();
            auto endShade = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float, std::milli> durationShade = endShade - startShade;
            if (iter == 1 && guiData->printDebugStats) { printf("Shade: %f ms, ", durationShade.count()); }
        }

        auto startNEE = std::chrono::high_resolution_clock::now();
        if (guiData->DirectLighting) {
            int num_paths_shadow_full = guiData->NumShadowRays * num_paths;
            num_paths_shadow = StreamCompaction::Efficient::compact(num_paths_shadow_full, dev_paths_shadow, dev_paths_shadow_compacted);
            std::swap(dev_paths_shadow, dev_paths_shadow_compacted);

            dim3 numblocksShadowSegmentTracing = (num_paths_shadow + blockSize1d - 1) / blockSize1d;
            if (numblocksShadowSegmentTracing.x > 0) {
                computeIntersectionsNaive<<<numblocksShadowSegmentTracing, blockSize1d>>>(num_paths_shadow, dev_paths_shadow, dev_geoms, hst_scene->geoms.size(), dev_intersections_shadow);
                shadeShadow << <numblocksShadowSegmentTracing, blockSize1d >> > (num_paths_shadow, dev_intersections_shadow, dev_paths_shadow, dev_image);
            }
        }
        auto endNEE = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> durationNEE = endNEE - startNEE;
        if (iter == 1 && guiData->printDebugStats) { printf("NEE: %f ms, ", durationNEE.count()); }
       
        depth++;

        auto startCompact = std::chrono::high_resolution_clock::now();
        if (guiData->useCompaction) {
            int newNumPaths = StreamCompaction::Efficient::compact(num_paths, dev_queue_rays, dev_queue_rays_next);
            num_paths = newNumPaths;
            std::swap(dev_queue_rays, dev_queue_rays_next);
        }
        auto endCompact = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> durationCompact = endCompact - startCompact;
        if (iter == 1 && guiData->printDebugStats) { printf("Compact: %f ms, ", durationCompact.count()); }
        
        if (iter == 1 && guiData->printDebugStats) {
            printf("#Depth: %d,\tTracked Rays: %d,\n", depth, num_paths);
        }


        if (num_paths == 0 || depth >= traceDepth) {
            iterationComplete = true;
        }

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter, bool wavefront) {
    if (wavefront) {
        pathtraceWave(pbo, frame, iter);
    }
    else {
        pathtraceNaive(pbo, frame, iter);
    }
}