//
// Created by jay on 9/30/24.
//

#include <stdexcept>
#include <vector>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/Importer.hpp>
#include <glad/glad.h>

#include "object.cuh"
#include "path.cuh"
#include "gl_wrapper.cuh"

using namespace cu_sim;

object::object(const std::string &path, const size_t instances) {
  gl_wrapper::force_initialized();

  Assimp::Importer importer;
  const auto scene = importer.ReadFile(binary_path() / path, aiProcess_Triangulate | aiProcess_GenNormals | aiProcess_FlipUVs);
  if(!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
    throw std::runtime_error("Failed to load model " + path + ": " + importer.GetErrorString());
  }

  if(scene->mNumMeshes != 1) {
    throw std::runtime_error("Only one-mesh models are supported");
  }
  const auto mesh = scene->mMeshes[0];
  if((mesh->mPrimitiveTypes & aiPrimitiveType_TRIANGLE) != aiPrimitiveType_TRIANGLE) {
    throw std::runtime_error("Only triangle meshes are supported");
  }
  std::vector<float> vertices;
  std::vector<unsigned int> indices;

  vertices.reserve(mesh->mNumVertices * 8);
  indices.reserve(mesh->mNumFaces * 3);

  for(size_t i = 0; i < mesh->mNumVertices; ++i) {
    // x y z
    vertices.push_back(mesh->mVertices[i].x);
    vertices.push_back(mesh->mVertices[i].y);
    vertices.push_back(mesh->mVertices[i].z);
    // u v
    vertices.push_back(mesh->mTextureCoords[0][i].x);
    vertices.push_back(mesh->mTextureCoords[0][i].y);
    // nx ny nz
    vertices.push_back(mesh->mNormals[i].x);
    vertices.push_back(mesh->mNormals[i].y);
    vertices.push_back(mesh->mNormals[i].z);
  }

  for(size_t i = 0; i < mesh->mNumFaces; ++i) {
    // v1 v2 v3
    indices.push_back(mesh->mFaces[i].mIndices[0]);
    indices.push_back(mesh->mFaces[i].mIndices[1]);
    indices.push_back(mesh->mFaces[i].mIndices[2]);
  }

  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), nullptr);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), reinterpret_cast<void *>(3 * sizeof(float)));
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), reinterpret_cast<void *>(5 * sizeof(float)));
  glEnableVertexAttribArray(2);

  auto make_cuda_buffer = [instances](const int index, const int width) {
    unsigned int buf;
    glGenBuffers(1, &buf);
    glBindBuffer(GL_ARRAY_BUFFER, buf);
    glBufferData(GL_ARRAY_BUFFER, instances * width * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(index, width, GL_FLOAT, GL_FALSE, width * sizeof(float), nullptr);
    glVertexAttribDivisor(index, 1);
    glEnableVertexAttribArray(index);

    return buf;
  };

  cuda.pos = make_cuda_buffer(3, 3);
  cuda.radius = make_cuda_buffer(4, 1);
  cuda.color = make_cuda_buffer(5, 3);

  glGenBuffers(1, &ebo);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

  element_count = indices.size();
  instance_count = instances;
}

void object::draw() const {
  glBindVertexArray(vao);
  glDrawElementsInstanced(GL_TRIANGLES, element_count, GL_UNSIGNED_INT, nullptr, instance_count);
}

object::~object() {
  glDeleteVertexArrays(1, &vao);
  glDeleteBuffers(1, &vbo);
  glDeleteBuffers(1, &ebo);
  glDeleteBuffers(1, &cuda.pos);
  glDeleteBuffers(1, &cuda.radius);
  glDeleteBuffers(1, &cuda.color);
}
