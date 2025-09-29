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
    else if (ext == ".obj") {
        loadFromOBJ(filename);
        wrapInCornellBox();
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
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
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
        else
        {
            newGeom.type = SPHERE;
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
    camera.focalDistance = 1.f;

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



void Scene::loadFromOBJ(const std::string& objName) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> objMaterials;
    std::string warn, err;

    std::string baseDir = objName.substr(0, objName.find_last_of("/\\") + 1);

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &objMaterials,
        &warn, &err, objName.c_str(), baseDir.c_str());

    if (!warn.empty()) std::cout << "OBJ load warning: " << warn << std::endl;
    if (!err.empty()) std::cerr << "OBJ load error: " << err << std::endl;
    if (!ret) exit(-1);

    std::unordered_map<int, int> ObjMatIDtoLocal;
    for (size_t i = 0; i < objMaterials.size(); i++) {
        Material m{};
        m.color = glm::vec3(objMaterials[i].diffuse[0],
            objMaterials[i].diffuse[1],
            objMaterials[i].diffuse[2]);
        m.emittance = (objMaterials[i].illum == 2) ? objMaterials[i].emission[0] : 0.f;
        ObjMatIDtoLocal[i] = (int)materials.size();
        materials.push_back(m);
    }

    int defaultMatID = (int)materials.size();
    Material defaultMat{};
    defaultMat.color = glm::vec3(0.5f);
    defaultMat.emittance = 0.0f;
    materials.push_back(defaultMat);

    for (const auto& shape : shapes) {
        size_t index_offset = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
            int fv = shape.mesh.num_face_vertices[f];
            if (fv != 3) { index_offset += fv; continue; }

            tinyobj::index_t idx0 = shape.mesh.indices[index_offset + 0];
            tinyobj::index_t idx1 = shape.mesh.indices[index_offset + 1];
            tinyobj::index_t idx2 = shape.mesh.indices[index_offset + 2];

            glm::vec3 v0(
                attrib.vertices[3 * idx0.vertex_index + 0],
                attrib.vertices[3 * idx0.vertex_index + 1],
                attrib.vertices[3 * idx0.vertex_index + 2]);
            glm::vec3 v1(
                attrib.vertices[3 * idx1.vertex_index + 0],
                attrib.vertices[3 * idx1.vertex_index + 1],
                attrib.vertices[3 * idx1.vertex_index + 2]);
            glm::vec3 v2(
                attrib.vertices[3 * idx2.vertex_index + 0],
                attrib.vertices[3 * idx2.vertex_index + 1],
                attrib.vertices[3 * idx2.vertex_index + 2]);

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

            int objMatID = shape.mesh.material_ids[f];
            if (objMatID >= 0) g.materialid = ObjMatIDtoLocal[objMatID];
            else g.materialid = defaultMatID;

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

    Camera& cam = state.camera;
    cam.position = glm::vec3(0, 0, 5);
    cam.lookAt = glm::vec3(0, 0, 0);
    cam.up = glm::vec3(0, 1, 0);
    cam.resolution = glm::ivec2(800, 800);
    cam.fov = glm::vec2(45.f, 45.f);
    state.traceDepth = 8;
    state.iterations = 100;
    state.imageName = "obj_render";
    state.image.assign(cam.resolution.x * cam.resolution.y, glm::vec3(0));
}

void Scene::wrapInCornellBox() {

    bool hasObj = false;
    glm::vec3 minB(FLT_MAX), maxB(-FLT_MAX);
    for (const auto& g : geoms) {
        if (g.type == TRIANGLE) {
            hasObj = true;
            minB = glm::min(minB, g.v0);
            minB = glm::min(minB, g.v1);
            minB = glm::min(minB, g.v2);
            maxB = glm::max(maxB, g.v0);
            maxB = glm::max(maxB, g.v1);
            maxB = glm::max(maxB, g.v2);
        }
    }

    if (hasObj) {
        glm::vec3 center = 0.5f * (minB + maxB);
        glm::vec3 extent = maxB - minB;
        float maxExtent = std::max(std::max(extent.x, extent.y), extent.z);
        float scaleFactor = 1.0f / maxExtent;

        for (auto& g : geoms) {
            if (g.type == TRIANGLE) {
                g.v0 = (g.v0 - center) * scaleFactor;
                g.v1 = (g.v1 - center) * scaleFactor;
                g.v2 = (g.v2 - center) * scaleFactor;
            }
        }
    }

    int redID = (int)materials.size();
    int greenID = redID + 1;
    int whiteID = redID + 2;
    int lightID = redID + 3;

    Material red{};
    red.color = glm::vec3(0.63f, 0.065f, 0.05f);

    Material green{};
    green.color = glm::vec3(0.14f, 0.45f, 0.091f);

    Material white{};
    white.color = glm::vec3(0.725f, 0.71f, 0.68f);

    Material light{};
    light.color = glm::vec3(1.0f);
    light.emittance = 50.0f;



    materials.push_back(red);
    materials.push_back(green);
    materials.push_back(white);
    materials.push_back(light);

    auto makeCube = [&](glm::vec3 t, glm::vec3 r, glm::vec3 s, int mat) {
        Geom g;
        g.type = CUBE;
        g.materialid = mat;
        g.translation = t;
        g.rotation = r;
        g.scale = s;
        g.transform = utilityCore::buildTransformationMatrix(g.translation, g.rotation, g.scale);
        g.inverseTransform = glm::inverse(g.transform);
        g.invTranspose = glm::inverseTranspose(g.transform);
        geoms.push_back(g);
        };
    

    makeCube(glm::vec3(0, -0.5f, 0), glm::vec3(0), glm::vec3(1, 0.01f, 1), whiteID);
    makeCube(glm::vec3(0, 0.5f, 0), glm::vec3(0), glm::vec3(1, 0.01f, 1), whiteID);
    makeCube(glm::vec3(0, 0, -0.5f), glm::vec3(0), glm::vec3(1, 1, 0.01f), whiteID);
    makeCube(glm::vec3(-0.5f, 0, 0), glm::vec3(0), glm::vec3(0.01f, 1, 1), redID);
    makeCube(glm::vec3(0.5f, 0, 0), glm::vec3(0), glm::vec3(0.01f, 1, 1), greenID);
    makeCube(glm::vec3(0, 0.49f, 0), glm::vec3(0), glm::vec3(0.25f, 0.01f, 0.25f), lightID);

    state.camera.position = glm::vec3(0, 0, 1.05f);
    state.camera.lookAt = glm::vec3(0, 0, 0);
    state.camera.up = glm::vec3(0, 1, 0);
    state.camera.resolution = glm::ivec2(800, 800);
    state.camera.fov = glm::vec2(45.0f);

    state.camera.view = glm::normalize(state.camera.lookAt - state.camera.position);
    state.camera.right = glm::normalize(glm::cross(state.camera.view, state.camera.up));
    state.camera.up = glm::normalize(glm::cross(state.camera.right, state.camera.view));

    float yscaled = tanf(45.0f * (PI / 180.0f));
    float xscaled = yscaled * (float)state.camera.resolution.x / (float)state.camera.resolution.y;
    state.camera.pixelLength = glm::vec2(2.f * xscaled / state.camera.resolution.x,
        2.f * yscaled / state.camera.resolution.y);

    state.traceDepth = 8;
    state.iterations = 500;
    state.imageName = "obj_in_cornell";
    state.image.resize(state.camera.resolution.x * state.camera.resolution.y);
}
