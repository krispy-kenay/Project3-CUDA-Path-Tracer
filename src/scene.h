#pragma once

#include "sceneStructs.h"
#include <vector>
#include <filesystem>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void loadFromOBJ(const std::filesystem::path& objPath, int materialOverrideID, const glm::vec3& translation, const glm::vec3& rotation, const glm::vec3& scale);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
};