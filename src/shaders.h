#pragma once

#include "sceneStructs.h"

__global__ void shadeFakeMaterial(int iter, int num_paths, ShadeableIntersection* shadeableIntersections, PathSegment* pathSegments, Material* materials);
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
    bool useSobol);
__global__ void shadeShadow(int P_shadow, const ShadeableIntersection* shadeableIntersections, PathSegment* shadowSegments, glm::vec3* dev_image);

__global__ void shadeEmissive(int num_paths, PathSegment* pathSegments, ShadeableIntersection* intersections, Material* materials, Geom* geoms, glm::vec3* dev_image, int num_lights, bool MIS);
__global__ void shadeDiffuse(int iter, int depth, int num_paths, PathSegment* pathSegments, ShadeableIntersection* intersections, Material* materials, bool useRR, int depthRR, bool useSobol);
__global__ void shadeSpecular(int iter, int depth, int num_paths, PathSegment* pathSegments, ShadeableIntersection* intersections, Material* materials, bool useRR, int depthRR);
__global__ void shadeRefractive(int iter, int depth, int num_paths, PathSegment* pathSegments, ShadeableIntersection* intersections, Material* materials, bool useRR, int depthRR);

__global__ void shadeShadowAccum(int iter, int depth, int num_paths, PathSegment* pathSegments, ShadeableIntersection* intersections, glm::vec3* dev_image);
__global__ void shadeShadowSetup(int iter, int depth, int num_paths, PathSegment* pathSegments, ShadeableIntersection* intersections, Material* materials, Geom* lights, int num_lights, int num_shadow_rays, bool useSobol);