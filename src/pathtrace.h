#pragma once

#include "scene.h"
#include "utilities.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene, bool useBVH = false, bool wavefront = false, int numShadowRays = 1);
void pathtraceFree(bool useBVH = false, bool wavefront = false, int numShadowRays = 1);
void pathtrace(uchar4 *pbo, int frame, int iteration, bool wavefront = false);
