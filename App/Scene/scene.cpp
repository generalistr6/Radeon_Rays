/**********************************************************************
Copyright (c) 2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
********************************************************************/
#include "Scene/scene.h"
#include "tiny_obj_loader.h"
#include "sh.h"
#include "shproject.h"
#include "OpenImageIO/imageio.h"
#include "Light/Ibl.h"

#include <algorithm>
#include <iterator>

namespace Baikal
{

using namespace RadeonRays;

static int GetTextureForemat(OIIO_NAMESPACE::ImageSpec const& spec)
{
    OIIO_NAMESPACE_USING

        if (spec.format.basetype == TypeDesc::UINT8)
            return Scene::RGBA8;
    if (spec.format.basetype == TypeDesc::HALF)
        return Scene::RGBA16;
    else
        return Scene::RGBA32;
}

static void LoadTexture(std::string const& filename, Scene::Texture& texture, std::vector<std::unique_ptr<char[]> >& data)
{
    OIIO_NAMESPACE_USING

    ImageInput* input = ImageInput::open(filename);

    if (!input)
    {
        throw std::runtime_error("Can't load " + filename + " image");
    }

    ImageSpec const& spec = input->spec();

    texture.w = spec.width;
    texture.h = spec.height;
    texture.d = spec.depth;
    texture.fmt = GetTextureForemat(spec);

    // Save old size for reading offset
    texture.dataoffset = (int)data.size();

	if (texture.fmt == Scene::RGBA8)
	{
		texture.size = spec.width * spec.height * spec.depth * 4;

		// Resize storage
		std::unique_ptr<char[]> texturedata(new char[spec.width * spec.height * spec.depth * 4]);

		// Initialize alpha value. [Manny]
		auto buffer = texturedata.get();
		for (int i = 0; i < texture.size; ++i)
		{
			buffer[i] = (char)(255);
		}

        // Read data to storage
        input->read_image(TypeDesc::UINT8, texturedata.get(), sizeof(char) * 4);

        // Close handle
        input->close();

        // Add to texture pool
        data.push_back(std::move(texturedata));
    }
    else if (texture.fmt == Scene::RGBA16)
    {
        texture.size = spec.width * spec.height * spec.depth * sizeof(float) * 2;

        // Resize storage
        std::unique_ptr<char[]> texturedata(new char[spec.width * spec.height * spec.depth * sizeof(float) * 2]);

        // Read data to storage
        input->read_image(TypeDesc::HALF, texturedata.get(), sizeof(float) * 2);

        // Close handle
        input->close();

        // Add to texture pool
        data.push_back(std::move(texturedata));
    }
    else
    {
        texture.size = spec.width * spec.height * spec.depth * sizeof(float3);

        // Resize storage
        std::unique_ptr<char[]> texturedata(new char[spec.width * spec.height * spec.depth * sizeof(float3)]);

        // Read data to storage
        input->read_image(TypeDesc::FLOAT, texturedata.get(), sizeof(float3));

        // Close handle
        input->close();

        // Add to texture pool
        data.push_back(std::move(texturedata));
    }

    // Cleanup
    delete input;
}

float Scene::specularRoughness_ = 0.2f; // [Manny]
float Scene::normalMapIntensity_ = 4.0f; // [Manny]

Scene* Scene::LoadFromObj(std::string const& filename, std::string const& basepath)
{
    using namespace tinyobj;

    // Loader data
    std::vector<shape_t> objshapes;
    std::vector<material_t> objmaterials;

    // Try loading file
    std::string res = LoadObj(objshapes, objmaterials, filename.c_str(), basepath.c_str());
    if (res != "")
    {
        throw std::runtime_error(res);
    }

    // Allocate scene
    Scene* scene(new Scene);

    // Texture map
    std::map<std::string, int> textures;
    std::map<int, int> matmap;

    // Enumerate and translate materials. [Manny]
    for (int i = 0; i < (int)objmaterials.size(); ++i)
    {
		// 1. kLambert - diffuse
		Material diffuseMaterial;
		{
			diffuseMaterial.kx = float3(objmaterials[i].diffuse[0], objmaterials[i].diffuse[1], objmaterials[i].diffuse[2]);
			diffuseMaterial.ni = Scene::normalMapIntensity_; // normal map intensity, when kLambert
			diffuseMaterial.type = kLambert;
			diffuseMaterial.fresnel = 0.f;

			// Load diffuse texture if needed
			if (!objmaterials[i].diffuse_texname.empty())
			{
				auto iter = textures.find(objmaterials[i].diffuse_texname);
				if (iter != textures.end())
				{
					diffuseMaterial.kxmapidx = iter->second;
				}
				else
				{
					Texture texture;

					// Load texture
					LoadTexture(basepath + objmaterials[i].diffuse_texname, texture, scene->texturedata_);

					// Add texture desc
					diffuseMaterial.kxmapidx = (int)scene->textures_.size();
					scene->textures_.push_back(texture);

					// Save in the map
					textures[objmaterials[i].diffuse_texname] = diffuseMaterial.kxmapidx;
				}
			}

			if (!objmaterials[i].normal_texname.empty())
			{
				auto iter = textures.find(objmaterials[i].normal_texname);
				if (iter != textures.end())
				{
					diffuseMaterial.nmapidx = iter->second;
				}
				else
				{
					Texture texture;

					// Load texture
					LoadTexture(basepath + objmaterials[i].normal_texname, texture, scene->texturedata_);

					// Add texture desc
					diffuseMaterial.nmapidx = (int)scene->textures_.size();
					scene->textures_.push_back(texture);

					// Save in the map
					textures[objmaterials[i].normal_texname] = diffuseMaterial.nmapidx;
				}
			}

			scene->materials_.push_back(diffuseMaterial);
			scene->material_names_.push_back(objmaterials[i].name);
		}

		// 2. Load normal map
		//float3 spec = float3(objmaterials[i].specular[0], objmaterials[i].specular[1], objmaterials[i].specular[2]);
		float3 spec = float3(1.f, 1.f, 1.f);
		if (spec.sqnorm() > 0.f)
		{
			Material specular;
			specular.kx = spec;
			specular.ni = 1.33f;//objmaterials[i].ior;
			specular.ns = Scene::specularRoughness_;//1.f - objmaterials[i].shininess;
			specular.type = kMicrofacetGGX;
			specular.nmapidx = -1;// scene->materials_.back().nmapidx;
			specular.fresnel = 5.f;

			if (!objmaterials[i].normal_texname.empty())
			{
				auto iter = textures.find(objmaterials[i].normal_texname);
				if (iter != textures.end())
				{
					specular.nmapidx = iter->second;
				}
				else
				{
					Texture texture;

					// Load texture
					LoadTexture(basepath + objmaterials[i].normal_texname, texture, scene->texturedata_);

					// Add texture desc
					specular.nmapidx = (int)scene->textures_.size();
					scene->textures_.push_back(texture);

					// Save in the map
					textures[objmaterials[i].normal_texname] = specular.nmapidx;
				}
			}

			scene->materials_.push_back(specular);
			scene->material_names_.push_back(objmaterials[i].name);

			Material layered;
			layered.ni = 1.33f;// objmaterials[i].ior;
			layered.type = kFresnelBlend;
			layered.brdftopidx = scene->materials_.size() - 1;
			layered.brdfbaseidx = scene->materials_.size() - 2;
			layered.fresnel = 1.f;
			layered.twosided = 1;

			scene->materials_.push_back(layered);
			scene->material_names_.push_back(objmaterials[i].name);
		}

		bool isTransparent = objmaterials[i].dissolve < 1.0f;

		if (!isTransparent &&
			diffuseMaterial.kxmapidx != -1 &&
			scene->textures_[diffuseMaterial.kxmapidx].fmt == RGBA8)
		{
			auto data = scene->texturedata_[diffuseMaterial.kxmapidx].get();
			for (int i = 3; i < scene->textures_[diffuseMaterial.kxmapidx].size; i += 4)
			{
				if (data[i] < (char)(255))
				{
					isTransparent = true;
					break;
				}
			}
		}


		// 3. Mix Alpha Material
		//if (objmaterials[i].dissolve < 1.0f ||
		//	(diffuseMaterial.kxmapidx != -1 && scene->textures_[diffuseMaterial.kxmapidx].fmt == RGBA8)) // has alpha channel
		if (isTransparent)
		{
			// 3-1. kPassthrough - normal
			{
				Material material;

				material.ni = 1.0f;
				material.type = kPassthrough;

				scene->materials_.push_back(material);
				scene->material_names_.push_back(objmaterials[i].name);
			}

			// 3-2. Kmix
			{
				Material material;

				material.ni = 1.0f;
				material.type = kMix;
				material.fresnel = 0.f;
				material.brdftopidx = scene->materials_.size() - 1;
				material.brdfbaseidx = scene->materials_.size() - 2;

				// alpha channel setting
				material.ns = objmaterials[i].dissolve;
				material.nsmapidx = diffuseMaterial.kxmapidx;

				scene->materials_.push_back(material);
				scene->material_names_.push_back(objmaterials[i].name);
			}
		}

		matmap[i] = scene->materials_.size() - 1;
    }

    // Enumerate all shapes in the scene
    for (int s = 0; s < (int)objshapes.size(); ++s)
    {
        // Prepare shape
        Shape shape;
        shape.startidx = (int)scene->indices_.size();
        shape.numprims = (int)objshapes[s].mesh.indices.size() / 3;
        shape.startvtx = (int)scene->vertices_.size();
        shape.numvertices = (int)objshapes[s].mesh.positions.size() / 3;
        shape.m = matrix();
        shape.linearvelocity = float3(0.0f, 0.f, 0.f);
        shape.angularvelocity = quaternion(0.f, 0.f, 0.f, 1.f);
        // Save last index to add to this shape indices
        // int baseidx = (int)scene->vertices_.size();

        int pos_count = (int)objshapes[s].mesh.positions.size() / 3;
        // Enumerate and copy vertex data
        for (int i = 0; i < pos_count; ++i)
        {
            scene->vertices_.push_back(float3(objshapes[s].mesh.positions[3 * i], objshapes[s].mesh.positions[3 * i + 1], objshapes[s].mesh.positions[3 * i + 2]));
        }

        for (int i = 0; i < (int)objshapes[s].mesh.normals.size() / 3; ++i)
        {
            scene->normals_.push_back(float3(objshapes[s].mesh.normals[3 * i], objshapes[s].mesh.normals[3 * i + 1], objshapes[s].mesh.normals[3 * i + 2]));
        }

        //check UV
        int texcoords_count = objshapes[s].mesh.texcoords.size() / 2;
        if (texcoords_count == pos_count)
        {
            for (int i = 0; i < texcoords_count; ++i)
            {
                float2 uv = float2(objshapes[s].mesh.texcoords[2 * i], objshapes[s].mesh.texcoords[2 * i + 1]);
                scene->uvs_.push_back(uv);
            }
        }
        else
        {
            for (int i = 0; i < pos_count; ++i)
            {
                scene->uvs_.push_back(float2(0, 0));
            }
        }

        // Enumerate and copy indices (accounting for base index) and material indices
        for (int i = 0; i < (int)objshapes[s].mesh.indices.size() / 3; ++i)
        {
            scene->indices_.push_back(objshapes[s].mesh.indices[3 * i]);
            scene->indices_.push_back(objshapes[s].mesh.indices[3 * i + 1]);
            scene->indices_.push_back(objshapes[s].mesh.indices[3 * i + 2]);

            int matidx = matmap[objshapes[s].mesh.material_ids[i]];
            scene->materialids_.push_back(matidx);

            if (scene->materials_[matidx].type == kEmissive)
            {
                Emissive emissive;
                emissive.shapeidx = s;
                emissive.primidx = i;
                emissive.m = matidx;
                scene->emissives_.push_back(emissive);
            }
        }

        scene->shapes_.push_back(shape);
    }

    // Check if there is no UVs
    if (scene->uvs_.empty())
    {
        scene->uvs_.resize(scene->normals_.size());
        std::fill(scene->uvs_.begin(), scene->uvs_.end(), float2(0, 0));
    }

    scene->envidx_ = -1;

    std::cout << "Loading complete\n";
    std::cout << "Number of objects: " << scene->shapes_.size() << "\n";
    std::cout << "Number of textures: " << scene->textures_.size() << "\n";
    std::cout << "Number of emissives: " << scene->emissives_.size() << "\n";

    return scene;
}

void Scene::SetEnvironment(std::string const& filename, std::string const& basepath, float envmapmul)
{
    // Save multiplier
    envmapmul_ = envmapmul;

    Texture texture;

    // Load texture
    if (!basepath.empty())
    {
        LoadTexture(basepath + filename, texture, texturedata_);
    }
    else
    {
        LoadTexture(filename, texture, texturedata_);
    }
    
    //
    //Ibl* ibl = new Ibl((float3*)(texturedata_[texture.dataoffset].get()), texture.w, texture.h);
    //ibl->Simulate("pdf.png");
    

    // Save index
    envidx_ = (int)textures_.size();

    // Add texture desc
    textures_.push_back(texture);
}

void Scene::SetBackground(std::string const& filename, std::string const& basepath)
{
    Texture texture;

    // Load texture
    if (!basepath.empty())
    {
        LoadTexture(basepath + filename, texture, texturedata_);
    }
    else
    {
        LoadTexture(filename, texture, texturedata_);
    }

    // Save index
    bgimgidx_ = (int)textures_.size();

    // Add texture desc
    textures_.push_back(texture);
}
}
