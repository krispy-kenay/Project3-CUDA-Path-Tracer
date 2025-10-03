#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

using namespace std;
using json = nlohmann::json;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);

    std::filesystem::path jsonPath(jsonName);
    std::filesystem::path baseDir = jsonPath.parent_path();

    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};

        if (p.contains("RGB") && p["RGB"].is_array() && p["RGB"].size() == 3) {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else {
            newMaterial.color = glm::vec3(1.f);
        }


        if (p["TYPE"] == "Diffuse")
        {
            newMaterial.hasReflective = 0.0f;
            newMaterial.hasRefractive = 0.0f;
            newMaterial.emittance = 0.0f;
        }
        else if (p["TYPE"] == "Emitting")
        {
            newMaterial.emittance = p.value("EMITTANCE", 1.f);
        }
        else if (p["TYPE"] == "Specular")
        {
            newMaterial.specular.color = newMaterial.color;
            newMaterial.specular.exponent = p.value("EXPONENT", 50.0f);
            newMaterial.hasReflective = 1.0f;
        }
        else if (p["TYPE"] == "Refractive") 
        {
            newMaterial.hasRefractive = 1.0f;
            newMaterial.indexOfRefraction = p.value("IOR", 1.5f);
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else if (type == "sphere") {
            newGeom.type = SPHERE;
        }
        else if (type == "mesh") {
            std::string filename = p.value("FILE", "");
            std::string materialName = p.value("MATERIAL", "");
            if (filename.empty()) {
                printf("No filename for mesh provided!");
                continue;
            }

            glm::vec3 translation(0.f), rotation(0.f), scale(1.f);
            if (p.contains("TRANS")) {
                const auto& trans = p["TRANS"];
                translation = glm::vec3(trans[0], trans[1], trans[2]);
            }
            if (p.contains("ROTAT")) {
                const auto& rotat = p["ROTAT"];
                rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
            }
            if (p.contains("SCALE")) {
                const auto& sc = p["SCALE"];
                scale = glm::vec3(sc[0], sc[1], sc[2]);
            }

            std::filesystem::path objPath(filename);
            if (!objPath.is_absolute()) {
                objPath = baseDir / objPath;
            }

            int matID = -1;
            if (!materialName.empty()) {
                matID = MatNameToID[materialName];
            }

            loadFromOBJ(objPath, matID, translation, rotation, scale);
            continue;
        }
        else
        {
            continue;
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);
    camera.lensRadius = 0.f;
    camera.focalDistance = 10.f;
    camera.focalLength = 50;

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}


void Scene::loadFromOBJ(const std::filesystem::path& objPath, int materialOverrideID, const glm::vec3& translation, const glm::vec3& rotation, const glm::vec3& scale) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> objMaterials;
    std::string warn, err;

    std::string objName = objPath.filename().string();
    std::string baseDir = objPath.parent_path().string();
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &objMaterials,
        &warn, &err, objPath.string().c_str(), baseDir.c_str());

    if (!warn.empty()) std::cout << "OBJ load warning: " << warn << std::endl;
    if (!err.empty()) std::cerr << "OBJ load error: " << err << std::endl;
    if (!ret) exit(-1);

    std::unordered_map<int, int> ObjMatIDtoLocal;
    for (size_t i = 0; i < objMaterials.size(); i++) {
        const auto& om = objMaterials[i];
        Material m{};
        m.color = glm::vec3(om.diffuse[0], om.diffuse[1], om.diffuse[2]);

        // Emission
        if (om.emission[0] > 0 || om.emission[1] > 0 || om.emission[2] > 0) {
            m.emittance = glm::length(glm::vec3(om.emission[0], om.emission[1], om.emission[2]));
        }

        // Specular
        if (glm::length(glm::vec3(om.specular[0], om.specular[1], om.specular[2])) > 0.0f) {
            m.hasReflective = 1.0f;
            m.specular.color = glm::vec3(om.specular[0], om.specular[1], om.specular[2]);
            m.specular.exponent = om.shininess > 0 ? om.shininess : 50.0f;
        }

        // Refraction
        if (om.ior > 1.01f) {
            m.hasRefractive = 1.0f;
            m.indexOfRefraction = om.ior;
        }

        ObjMatIDtoLocal[i] = (int)materials.size();
        materials.push_back(m);
    }

    int defaultMatID = (int)materials.size();
    Material defaultMat{};
    defaultMat.color = glm::vec3(0.5f);
    defaultMat.emittance = 0.0f;
    materials.push_back(defaultMat);

    const glm::mat4 M = utilityCore::buildTransformationMatrix(translation, rotation, scale);
    const glm::mat3 N = glm::transpose(glm::inverse(glm::mat3(M)));
    const float detM = glm::determinant(glm::mat3(M));
    const bool flipWinding = detM < 0.0f;

    auto xformP = [&](const glm::vec3& p) {
        glm::vec4 hp = M * glm::vec4(p, 1.0f);
        return glm::vec3(hp);
        };
    auto xformN = [&](const glm::vec3& n) {
        return glm::normalize(N * n);
        };
    for (const auto& shape : shapes) {
        size_t index_offset = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
            int fv = shape.mesh.num_face_vertices[f];
            if (fv != 3) { index_offset += fv; continue; }

            tinyobj::index_t idx0 = shape.mesh.indices[index_offset + 0];
            tinyobj::index_t idx1 = shape.mesh.indices[index_offset + 1];
            tinyobj::index_t idx2 = shape.mesh.indices[index_offset + 2];

            if (flipWinding) std::swap(idx1, idx2);

            glm::vec3 p0(
                attrib.vertices[3 * idx0.vertex_index + 0],
                attrib.vertices[3 * idx0.vertex_index + 1],
                attrib.vertices[3 * idx0.vertex_index + 2]);
            glm::vec3 p1(
                attrib.vertices[3 * idx1.vertex_index + 0],
                attrib.vertices[3 * idx1.vertex_index + 1],
                attrib.vertices[3 * idx1.vertex_index + 2]);
            glm::vec3 p2(
                attrib.vertices[3 * idx2.vertex_index + 0],
                attrib.vertices[3 * idx2.vertex_index + 1],
                attrib.vertices[3 * idx2.vertex_index + 2]);

            glm::vec3 v0 = xformP(p0);
            glm::vec3 v1 = xformP(p1);
            glm::vec3 v2 = xformP(p2);

            Geom g{};
            g.type = TRIANGLE;
            g.v0 = v0;
            g.v1 = v1;
            g.v2 = v2;

            if (!attrib.normals.empty() &&
                idx0.normal_index >= 0 && idx1.normal_index >= 0 && idx2.normal_index >= 0) {
                g.n0 = glm::vec3(attrib.normals[3 * idx0.normal_index + 0],
                    attrib.normals[3 * idx0.normal_index + 1],
                    attrib.normals[3 * idx0.normal_index + 2]);
                g.n1 = glm::vec3(attrib.normals[3 * idx1.normal_index + 0],
                    attrib.normals[3 * idx1.normal_index + 1],
                    attrib.normals[3 * idx1.normal_index + 2]);
                g.n2 = glm::vec3(attrib.normals[3 * idx2.normal_index + 0],
                    attrib.normals[3 * idx2.normal_index + 1],
                    attrib.normals[3 * idx2.normal_index + 2]);
            }
            else {
                glm::vec3 faceNormal = glm::normalize(glm::cross(v1 - v0, v2 - v0));
                g.n0 = g.n1 = g.n2 = faceNormal;
            }

            if (materialOverrideID == -1) {
                int objMatID = shape.mesh.material_ids[f];
                if (objMatID >= 0) g.materialid = ObjMatIDtoLocal[objMatID];
                else g.materialid = defaultMatID;
            }
            else {
                g.materialid = materialOverrideID;
            }

            g.translation = glm::vec3(0);
            g.rotation = glm::vec3(0);
            g.scale = glm::vec3(1);
            g.transform = glm::mat4(1.0f);
            g.inverseTransform = glm::mat4(1.0f);
            g.invTranspose = glm::mat4(1.0f);

            geoms.push_back(g);

            index_offset += fv;
        }
    }
}