#pragma once

#include "glm/glm.hpp"

#include <algorithm>
#include <istream>
#include <iterator>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#define PI                3.1415926535897932384626422832795028841971f
#define TWO_PI            6.2831853071795864769252867665590057683943f
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f
#define EPSILON           0.00001f

class GuiDataContainer
{
public:
    GuiDataContainer() : TracedDepth(8), printDebugStats(false), useBVH(true), wavefront(true), antiAliasing(false), StratifiedSampling(false), Strata(4), SobolSampling(false), SortMaterial(true), useCompaction(true), RussianRoulette(false), depthRussianRoulette(5), DirectLighting(false), NumShadowRays(1), showEmissive(true), showRefractive(true), showSpecular(true), showDiffuse(true), showShadows(true), MaxSPP(1024) {}
    int TracedDepth;
    bool printDebugStats;
    bool useBVH;
    bool wavefront;
    bool antiAliasing;
    bool StratifiedSampling;
    int Strata;
    bool SobolSampling;
    bool SortMaterial;
    bool useCompaction;
    bool RussianRoulette;
    int depthRussianRoulette;
    bool DirectLighting;
    int NumShadowRays;
    bool showEmissive;
    bool showRefractive;
    bool showSpecular;
    bool showDiffuse;
    bool showShadows;
    int MaxSPP;
};

namespace utilityCore
{
    extern float clamp(float f, float min, float max);
    extern bool replaceString(std::string& str, const std::string& from, const std::string& to);
    extern glm::vec3 clampRGB(glm::vec3 color);
    extern bool epsilonCheck(float a, float b);
    extern std::vector<std::string> tokenizeString(std::string str);
    extern glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
    extern std::string convertIntToString(int number);
    extern std::istream& safeGetline(std::istream& is, std::string& t); //Thanks to http://stackoverflow.com/a/6089413
}
