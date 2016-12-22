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
#include "OpenImageIO/imageio.h"

#ifdef __APPLE__
#include <OpenCL/OpenCL.h>
#include <OpenGL/OpenGL.h>
#elif WIN32
#define NOMINMAX
#include <Windows.h>
#include "GL/glew.h"
#include "GLUT/GLUT.h"
#include "GLUT/freeglut_ext.h"
#else
#include <CL/cl.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/glx.h>
#endif

#include <string>
#include <memory>
#include <chrono>
#include <cassert>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <thread>
#include <atomic>
#include <mutex>

#define _USE_MATH_DEFINES
#include <math.h>

#ifdef RR_EMBED_KERNELS
#include "./CL/cache/kernels.h"
#endif

#include "CLW.h"

#include "math/mathutils.h"

#include "tiny_obj_loader.h"
#include "perspective_camera.h"
#include "shader_manager.h"
#include "Scene/scene.h"
#include "PT/ptrenderer.h"
#include "AO/aorenderer.h"
#include "CLW/clwoutput.h"
#include "config_manager.h"

using namespace RadeonRays;

// Help message
char const* kHelpMessage =
"App [-p path_to_models][-f model_name][-b][-r][-ns number_of_shadow_rays][-ao ao_radius][-w window_width][-h window_height][-nb number_of_indirect_bounces]";
//char const* g_path =
//"../Resources/bmw/knit";
//char const* g_modelname = "knit.obj";
//char const* g_envmapname = "../Resources/Textures/ENV04.hdr";

struct OBJResources
{
	std::string basePath;
	std::string objFilePath;
};

std::vector<OBJResources>  objPaths;
std::vector<std::string>  envPaths;
int g_objIndex = 2;
int g_envIndex = 3;

std::unique_ptr<ShaderManager>    g_shader_manager;

GLuint g_vertex_buffer;
GLuint g_index_buffer;
GLuint g_texture;

int g_window_width = 1024;
int g_window_height = 1024;
int g_num_shadow_rays = 1;
int g_num_ao_rays = 1;
int g_ao_enabled = false;
int g_progressive = false;
int g_num_bounces = 5;
int g_num_samples = -1;
int g_samplecount = 0;
float g_ao_radius = 1.f; 
float g_envmapmul = 1.f;
float g_cspeed = 100.25f;

float3 g_camera_pos = float3(0.f, 1.f, 4.f);
float3 g_camera_at = float3(0.f, 1.f, 0.f);
float3 g_camera_up = float3(0.f, 1.f, 0.f);

float2 g_camera_sensor_size = float2(0.036f, 0.024f);  // default full frame sensor 36x24 mm
float2 g_camera_zcap = float2(0.0f, 100000.f);
float g_camera_focal_length = 0.035f; // 35mm lens
float g_camera_focus_distance = 0.f;
float g_camera_aperture = 0.f;


bool g_recording_enabled = false;
int g_frame_count = 0;
bool g_benchmark = false;
bool g_interop = true;
ConfigManager::Mode g_mode = ConfigManager::Mode::kUseSingleCpu;    

enum MaterialUpdateMode
{
	MaterialUpdateMode_specularRoughness,
	MaterialUpdateMode_normalMapIntensity,
	MaterialUpdateMode_lightAngle
};

MaterialUpdateMode g_material_update_mode = MaterialUpdateMode_specularRoughness;
bool g_is_display_material_info = true;

using namespace tinyobj;


struct OutputData
{
    Baikal::ClwOutput* output;
    std::vector<float3> fdata;
    std::vector<unsigned char> udata;
    CLWBuffer<float3> copybuffer;
};

struct ControlData
{
    std::atomic<int> clear;
    std::atomic<int> stop;
    std::atomic<int> newdata;
    std::mutex datamutex;
    int idx;
};

std::vector<ConfigManager::Config> g_cfgs;
std::vector<OutputData> g_outputs;
std::unique_ptr<ControlData[]> g_ctrl;
std::vector<std::thread> g_renderthreads;
int g_primary = -1;


std::unique_ptr<Baikal::Scene> g_scene;


static bool     g_is_left_pressed = false;
static bool     g_is_right_pressed = false;
static bool     g_is_fwd_pressed = false;
static bool     g_is_back_pressed = false;
static bool     g_is_home_pressed = false;
static bool     g_is_end_pressed = false;
static bool     g_is_mouse_tracking = false;
static bool		g_is_mouse_wheel = false;
static float2   g_mouse_pos = float2(0, 0);
static float2   g_mouse_delta = float2(0, 0);
static bool		g_material_changed = false;
static bool		g_light_changed = false;
static float	g_mouse_wheel = 10.00f;

// CLW stuff
CLWImage2D g_cl_interop_image;

char* GetCmdOption(char ** begin, char ** end, const std::string & option);
bool CmdOptionExists(char** begin, char** end, const std::string& option);
void ShowHelpAndDie();
void SaveFrameBuffer(std::string const& name, float3 const* data);

void Render()
{
    try
    {
        {
            glDisable(GL_DEPTH_TEST);
            glViewport(0, 0, g_window_width, g_window_height);

            glClear(GL_COLOR_BUFFER_BIT);

            glBindBuffer(GL_ARRAY_BUFFER, g_vertex_buffer);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_index_buffer);

            GLuint program = g_shader_manager->GetProgram("../App/simple");
            glUseProgram(program);

            GLuint texloc = glGetUniformLocation(program, "g_Texture");
            assert(texloc >= 0);

            glUniform1i(texloc, 0);

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, g_texture);

            GLuint position_attr = glGetAttribLocation(program, "inPosition");
            GLuint texcoord_attr = glGetAttribLocation(program, "inTexcoord");

            glVertexAttribPointer(position_attr, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 5, 0);
            glVertexAttribPointer(texcoord_attr, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 5, (void*)(sizeof(float) * 3));

            glEnableVertexAttribArray(position_attr);
            glEnableVertexAttribArray(texcoord_attr);

            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, nullptr);

            glDisableVertexAttribArray(texcoord_attr);
            glBindTexture(GL_TEXTURE_2D, 0);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
            glUseProgram(0);

			if (g_is_display_material_info)
			{
				glColor3f(0.6f, 0.6f, 0.6f);
				glDisable(GL_LIGHTING);

				using namespace Baikal;

				glMatrixMode(GL_PROJECTION);
				glPushMatrix();
				glLoadIdentity();
				gluOrtho2D(0, g_window_width, 0, g_window_height);

				glMatrixMode(GL_MODELVIEW);
				glPushMatrix();
				glLoadIdentity();

				// specular roughness
				{
					glRasterPos2i(10, g_window_height - 30);  // move in 10 pixels from the left and bottom edges
					std::string str;
					str.resize(256);
					sprintf(&(str[0]), "1 : Specular roughness : %f (q-decrease, w-increase)\n", Scene::specularRoughness_);
					for (int i = 0; i < str.length(); ++i)
					{
						if (g_material_update_mode == MaterialUpdateMode_specularRoughness)
							glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, str[i]);
						else
							glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, str[i]);
					}
				}

				// normal map intensity
				{
					glRasterPos2i(10, g_window_height - 50);  // move in 10 pixels from the left and bottom edges
					std::string str;
					str.resize(256);
					sprintf(&(str[0]), "2 : Normal map intensity : %f (q-decrease, w-increase)\n", Scene::normalMapIntensity_);
					for (int i = 0; i < str.length(); ++i)
					{
						if (g_material_update_mode == MaterialUpdateMode_normalMapIntensity)
							glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, str[i]);
						else
							glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, str[i]);
					}
				}

				// environment map rotation
				{
					glRasterPos2i(10, g_window_height - 70);  // move in 10 pixels from the left and bottom edges
					std::string str;
					str.resize(256);
					sprintf(&(str[0]), "3 : Environment map rotation(degree) : %d (q-decrease, w-increase)\n", (int)(g_scene->camera_->GetCameraAngle() * (180 / PI)));
					for (int i = 0; i < str.length(); ++i)
					{
						if (g_material_update_mode == MaterialUpdateMode_lightAngle)
							glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, str[i]);
						else
							glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, str[i]);
					}
				}

				// g_num_bounces
				{
					glRasterPos2i(10, g_window_height - 90);  // move in 10 pixels from the left and bottom edges
					std::string str;
					str.resize(256);
					sprintf(&(str[0]), "    Bounce number : %d ('page up'-decrease, 'page down'-increase)\n", g_num_bounces);
					for (int i = 0; i < str.length(); ++i)
					{
						glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, str[i]);
					}
				}

				// show hide toggle
				{
					glRasterPos2i(10, g_window_height - 110);  // move in 10 pixels from the left and bottom edges
					std::string str;
					str.resize(256);
					sprintf(&(str[0]), "' : show / hide text.\n");
					for (int i = 0; i < str.length(); ++i)
					{
						glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, str[i]);
					}
				}

				glPopMatrix();

				glMatrixMode(GL_PROJECTION);
				glPopMatrix();
				glMatrixMode(GL_MODELVIEW);
			}

            glFinish();
        }

        glutSwapBuffers();
    }
    catch (std::runtime_error& e)
    {
        std::cout << e.what();
        exit(-1);
    }
}

void InitGraphics()
{
    g_shader_manager.reset(new ShaderManager());

    glClearColor(0.0, 0.0, 0.0, 0.0);
    glCullFace(GL_NONE);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D);

    glGenBuffers(1, &g_vertex_buffer);
    glGenBuffers(1, &g_index_buffer);

    // create Vertex buffer
    glBindBuffer(GL_ARRAY_BUFFER, g_vertex_buffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_index_buffer);

    float quad_vdata[] =
    {
        -1, -1, 0.5, 0, 0,
        1, -1, 0.5, 1, 0,
        1, 1, 0.5, 1, 1,
        -1, 1, 0.5, 0, 1
    };

    GLshort quad_idata[] =
    {
        0, 1, 3,
        3, 1, 2
    };

    // fill data
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad_vdata), quad_vdata, GL_STATIC_DRAW);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(quad_idata), quad_idata, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);


    glGenTextures(1, &g_texture);
    glBindTexture(GL_TEXTURE_2D, g_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, g_window_width, g_window_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    glBindTexture(GL_TEXTURE_2D, 0);
}

void InitCl()
{
    bool force_disable_itnerop = false;

    try
    {
        ConfigManager::CreateConfigs(g_mode, g_interop, g_cfgs, g_num_bounces);
    }
    catch (CLWException &)
    {
        force_disable_itnerop = true;
        ConfigManager::CreateConfigs(g_mode, false, g_cfgs, g_num_bounces);
    }


    std::cout << "Running on devices: \n";

    for (int i = 0; i < g_cfgs.size(); ++i)
    {
        std::cout << i << ": " << g_cfgs[i].context.GetDevice(g_cfgs[i].devidx).GetName() << "\n";
    }

    g_interop = false;

    g_outputs.resize(g_cfgs.size());
    g_ctrl.reset(new ControlData[g_cfgs.size()]);

    for (int i = 0; i < g_cfgs.size(); ++i)
    {
        if (g_cfgs[i].type == ConfigManager::kPrimary)
        {
            g_primary = i;

            if (g_cfgs[i].caninterop)
            {
                g_cl_interop_image = g_cfgs[i].context.CreateImage2DFromGLTexture(g_texture);
                g_interop = true;
            }
        }

        g_ctrl[i].clear.store(1);
        g_ctrl[i].stop.store(0);
        g_ctrl[i].newdata.store(0);
        g_ctrl[i].idx = i;
    }

    if (force_disable_itnerop)
    {
        std::cout << "OpenGL interop is not supported, disabled, -interop flag is ignored\n";
    }
    else
    {
        if (g_interop)
        {
            std::cout << "OpenGL interop mode enabled\n";
        }
        else
        {
            std::cout << "OpenGL interop mode disabled\n";
        }
    }
}

void InitData()
{
    rand_init();

    // Load obj file
    std::string basepath = objPaths[g_objIndex].basePath;
    basepath += "/";
	std::string filename = objPaths[g_objIndex].objFilePath;

    g_scene.reset(Baikal::Scene::LoadFromObj(filename, basepath));

	float3 min = float3(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
	float3 max = float3(std::numeric_limits<float>::min(), std::numeric_limits<float>::min(), std::numeric_limits<float>::min());

	for (int i = 0; i < g_scene->vertices_.size(); ++i)
	{
		auto& v = g_scene->vertices_[i];

		min.x = std::min(min.x, v.x);
		min.y = std::min(min.y, v.y);
		min.z = std::min(min.z, v.z);

		max.x = std::max(max.x, v.x);
		max.y = std::max(max.y, v.y);
		max.z = std::max(max.z, v.z);
	}

	auto diagonalLength = std::sqrtf((max.x - min.x) * (max.x - min.x) + (max.y - min.y) * (max.y - min.y) + (max.z - min.z) * (max.z - min.z));
	g_camera_pos = float3((max.x + min.x) * 0.5f, (max.y + min.y) * 0.5f, max.z + diagonalLength);
	g_camera_at = float3((max.x + min.x) * 0.5f, (max.y + min.y) * 0.5f, (max.z + min.z) * 0.5f);
	g_camera_up = float3(0.f, 1.f, 0.f);

    g_scene->camera_.reset(new PerspectiveCamera(
        g_camera_pos
        , g_camera_at
        , g_camera_up));

    // Adjust sensor size based on current aspect ratio
    float aspect = (float)g_window_width / g_window_height;
    g_camera_sensor_size.y = g_camera_sensor_size.x / aspect;

    g_scene->camera_->SetSensorSize(g_camera_sensor_size);
    g_scene->camera_->SetDepthRange(g_camera_zcap);
    g_scene->camera_->SetFocalLength(g_camera_focal_length);
    g_scene->camera_->SetFocusDistance(g_camera_focus_distance);
    g_scene->camera_->SetAperture(g_camera_aperture);

    std::cout << "Camera type: " << (g_scene->camera_->GetAperture() > 0.f ? "Physical" : "Pinhole") << "\n";
    std::cout << "Lens focal length: " << g_scene->camera_->GetFocalLength() * 1000.f << "mm\n";
    std::cout << "Lens focus distance: " << g_scene->camera_->GetFocusDistance() << "m\n";
    std::cout << "F-Stop: " << 1.f / (g_scene->camera_->GetAperture() * 10.f) << "\n";
    std::cout << "Sensor size: " << g_camera_sensor_size.x * 1000.f << "x" << g_camera_sensor_size.y * 1000.f << "mm\n";

    g_scene->SetEnvironment(envPaths[g_envIndex], "", g_envmapmul);
    g_scene->AddDirectionalLight(RadeonRays::float3(-0.3f, -1.f, -0.4f), 2.f * RadeonRays::float3(1.f, 1.f, 1.f));
    g_scene->AddPointLight(RadeonRays::float3(-0.5f, 1.7f, 0.0f), RadeonRays::float3(1.f, 0.9f, 0.6f));
    g_scene->AddSpotLight(RadeonRays::float3(0.5f, 1.5f, 0.0f), RadeonRays::float3(-0.5f, -1.0f, 0.1f), RadeonRays::float3(1.f, 0.9f, 0.6f),
                           std::cos(M_PI_4/2), std::cos(M_PI_4));
#pragma omp parallel for
    for (int i = 0; i < g_cfgs.size(); ++i)
    {
        //g_cfgs[i].renderer->SetNumBounces(g_num_bounces);
        g_cfgs[i].renderer->Preprocess(*g_scene);

        g_outputs[i].output = (Baikal::ClwOutput*)g_cfgs[i].renderer->CreateOutput(g_window_width, g_window_height);

        g_cfgs[i].renderer->SetOutput(g_outputs[i].output);

        g_outputs[i].fdata.resize(g_window_width * g_window_height);
        g_outputs[i].udata.resize(g_window_width * g_window_height * 4);

        if (g_cfgs[i].type == ConfigManager::kPrimary)
        {
            g_outputs[i].copybuffer = g_cfgs[i].context.CreateBuffer<float3>(g_window_width * g_window_height, CL_MEM_READ_WRITE);
        }
    }

    g_cfgs[g_primary].renderer->Clear(float3(0, 0, 0), *g_outputs[g_primary].output);
}

void Reshape(GLint w, GLint h)
{
    // Disable window resize
    glutReshapeWindow(g_window_width, g_window_height);
}

void OnMouseMove(int x, int y)
{
    if (g_is_mouse_tracking)
    {
        g_mouse_delta = float2((float)x, (float)y) - g_mouse_pos;
        g_mouse_pos = float2((float)x, (float)y);
    }
}

void OnMouseButton(int btn, int state, int x, int y)
{
    if (btn == GLUT_LEFT_BUTTON)
    {
        if (state == GLUT_DOWN)
        {
            g_is_mouse_tracking = true;
            g_mouse_pos = float2((float)x, (float)y);
            g_mouse_delta = float2(0, 0);
        }
        else if (state == GLUT_UP && g_is_mouse_tracking)
        {
            g_is_mouse_tracking = true;
            g_mouse_delta = float2(0, 0);
        }
    }
}

void OnMouseWheel(int btn, int dir, int x, int y)
{
	if (dir > 0)
	{
		// Zoom in
		//printf("zoom in \n");
		g_scene->camera_->MoveForward(-g_mouse_wheel);
		g_is_mouse_wheel = true;
	}
	else
	{
		// Zoom out
		//printf("zoom out \n");
		g_scene->camera_->MoveForward(+g_mouse_wheel);
		g_is_mouse_wheel = true;
	}

	return;
}

void OnKey(int key, int x, int y)
{
    switch (key)
    {
    case GLUT_KEY_UP:
        g_is_fwd_pressed = true;
        break;
    case GLUT_KEY_DOWN:
        g_is_back_pressed = true;
        break;
    case GLUT_KEY_LEFT:
        g_is_left_pressed = true;
        break;
    case GLUT_KEY_RIGHT:
        g_is_right_pressed = true;
        break;
    case GLUT_KEY_HOME:
        g_is_home_pressed = true;
        break;
    case GLUT_KEY_END:
        g_is_end_pressed = true;
        break;
    case GLUT_KEY_F1:
        g_mouse_delta = float2(0, 0);
        break;
    case GLUT_KEY_F3:
        g_benchmark = true;
        break;
    case GLUT_KEY_F4:
        if (!g_interop)
        {
            std::ostringstream oss;
            oss << "aov_color_" << g_frame_count << ".hdr";
            SaveFrameBuffer(oss.str(), &g_outputs[g_primary].fdata[0]);
            break;
        }
    default:
        break;
    }
}

void UpdateMaterialValue(bool isPositive)
{
	using namespace Baikal;

	auto scene = g_scene.get();
	float delta = 0.0f;

	switch (g_material_update_mode)
	{
	case MaterialUpdateMode_specularRoughness:
		if (isPositive)
			delta = 0.1f;
		else
			delta = -0.1f;
		g_material_changed = true;
		break;
	case MaterialUpdateMode_normalMapIntensity:
		if (isPositive)
			delta = 0.2f;
		else
			delta = -0.2f;
		g_material_changed = true;
		break;
	case MaterialUpdateMode_lightAngle:
		if (isPositive)
			delta = 15.f * (PI / 180);
		else
			delta = -15.f * (PI / 180);
		g_light_changed = true;
		break;
	default:
		return;
	}

	// material update
	if (g_material_changed)
	{
		if (g_material_update_mode == MaterialUpdateMode_specularRoughness)
		{
			Scene::specularRoughness_ += delta;

			for (std::size_t i = 0; i < scene->materials_.size(); ++i)
			{
				auto& material = scene->materials_[i];
				if (material.type == Scene::Bxdf::kMicrofacetGGX)
				{
					material.ns = Scene::specularRoughness_;
				}
			}
		}
		else if (g_material_update_mode == MaterialUpdateMode_normalMapIntensity)
		{
			Scene::normalMapIntensity_ += delta;

			for (std::size_t i = 0; i < scene->materials_.size(); ++i)
			{
				auto& material = scene->materials_[i];
				if (material.type == Scene::Bxdf::kLambert)
				{
					material.ni = Scene::normalMapIntensity_;
				}
			}
		}
	}

	// light rotate
	if (g_light_changed)
	{
		// 1. rotate camera
		g_scene->camera_->Rotate(delta);

		// 2. rotate shapes
		auto rot = rotation_y(delta);
		auto cameraPos = g_scene->camera_->GetCameraPos();
		RadeonRays::matrix translate = translation(cameraPos);

		for (std::size_t i = 0; i < g_scene->shapes_.size(); ++i)
		{
			g_scene->shapes_[i].m *= translate * rot * inverse(translate);
		}
	}
}

void OnNormalKeys(unsigned char key, int x, int y)
{
	switch (key)
	{
	case '`':
		g_is_display_material_info = !g_is_display_material_info;
	case '1':
		g_material_update_mode = MaterialUpdateMode_specularRoughness;
		fprintf(stderr, "Changing Specular Roughness...\n");
		break;
	case '2':
		g_material_update_mode = MaterialUpdateMode_normalMapIntensity;
		fprintf(stderr, "Changing Normal Intensity...\n");
		break;
	case '3':
		g_material_update_mode = MaterialUpdateMode_lightAngle;
		fprintf(stderr, "Changing Light Angle...\n");
		break;
	case '4':
		fprintf(stderr, "");
		break;
	case '5':
		fprintf(stderr, "");
		break;
	case '6':
		fprintf(stderr, "");
		break;
	case '7':
		fprintf(stderr, "");
		break;
	case 'q':
		UpdateMaterialValue(false);
		break;
	case 'w':
		UpdateMaterialValue(true);
		break;
	case 27:
		exit(0);
	default:
		break;
	}
}

void OnNormalKeysUp(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 'q':
	case 'w':
		//g_material_changed = false;
		break;
	default:
		break;
	}
}

void OnKeyUp(int key, int x, int y)
{
    switch (key)
    {
    case GLUT_KEY_UP:
        g_is_fwd_pressed = false;
        break;
    case GLUT_KEY_DOWN:
        g_is_back_pressed = false;
        break;
    case GLUT_KEY_LEFT:
        g_is_left_pressed = false;
        break;
    case GLUT_KEY_RIGHT:
        g_is_right_pressed = false;
        break;
    case GLUT_KEY_HOME:
        g_is_home_pressed = false;
        break;
    case GLUT_KEY_END:
        g_is_end_pressed = false;
        break;
    case GLUT_KEY_PAGE_DOWN:
    {
        ++g_num_bounces;
        for (int i = 0; i < g_cfgs.size(); ++i)
        {
            g_cfgs[i].renderer->SetNumBounces(g_num_bounces);
            g_cfgs[i].renderer->Clear(float3(0, 0, 0), *g_outputs[i].output);
        }
        g_samplecount = 0;
        break;
    }
    case GLUT_KEY_PAGE_UP:
    {
        if (g_num_bounces > 1)
        {
            --g_num_bounces;
            for (int i = 0; i < g_cfgs.size(); ++i)
            {
                g_cfgs[i].renderer->SetNumBounces(g_num_bounces);
                g_cfgs[i].renderer->Clear(float3(0, 0, 0), *g_outputs[i].output);
            }
            g_samplecount = 0;
        }
        break;
    }
    default:
        break;
    }
}

void Update()
{
    static auto prevtime = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::high_resolution_clock::now();
    auto dt = std::chrono::duration_cast<std::chrono::duration<double>>(time - prevtime);
    prevtime = time;

    bool update = false;
    float camrotx = 0.f;
    float camroty = 0.f;

    const float kMouseSensitivity = 0.001125f;
    float2 delta = g_mouse_delta * float2(kMouseSensitivity, kMouseSensitivity);
    camrotx = -delta.x;
    camroty = -delta.y;

    if (std::abs(camroty) > 0.001f)
    {
        //g_scene->camera_->Tilt(camroty);
        g_scene->camera_->ArcballRotateVertically(float3(0, 0, 0), camroty);
        update = true;
    }

    if (std::abs(camrotx) > 0.001f)
    {

        //g_scene->camera_->Rotate(camrotx);
        g_scene->camera_->ArcballRotateHorizontally(float3(0, 0, 0), camrotx);
        update = true;
    }

	if (g_is_mouse_wheel) {
		update = true;
	}

    const float kMovementSpeed = g_cspeed;
    if (g_is_fwd_pressed)
    {
        g_scene->camera_->MoveForward((float)dt.count() * kMovementSpeed);
        update = true;
    }

    if (g_is_back_pressed)
    {
        g_scene->camera_->MoveForward(-(float)dt.count() * kMovementSpeed);
        update = true;
    }

    if (g_is_right_pressed)
    {
        g_scene->camera_->MoveRight((float)dt.count() * kMovementSpeed);
        update = true;
    }

    if (g_is_left_pressed)
    {
        g_scene->camera_->MoveRight(-(float)dt.count() * kMovementSpeed);
        update = true;
    }

    if (g_is_home_pressed)
    {
        g_scene->camera_->MoveUp((float)dt.count() * kMovementSpeed);
        update = true;
    }

    if (g_is_end_pressed)
    {
        g_scene->camera_->MoveUp(-(float)dt.count() * kMovementSpeed);
        update = true;
    }

	if (g_material_changed)
	{
		g_scene->set_dirty(Baikal::Scene::kMaterialInputs);
		update = true;
	}

	if (g_light_changed)
	{
		g_scene->set_dirty(Baikal::Scene::kGeometryTransform);
		update = true;
	}

    if (update)
    {
		g_is_mouse_wheel = false;

        g_scene->set_dirty(Baikal::Scene::kCamera);

        if (g_num_samples > -1)
        {
            g_samplecount = 0;
        }

        for (int i = 0; i < g_cfgs.size(); ++i)
        {
            if (i == g_primary)
                g_cfgs[i].renderer->Clear(float3(0, 0, 0), *g_outputs[i].output);
            else
                g_ctrl[i].clear.store(true);
        }

        /*        numbnc = 1;
        for (int i = 0; i < g_cfgs.size(); ++i)
        {
        g_cfgs[i].renderer->SetNumBounces(numbnc);
        }*/
    }

    if (g_num_samples == -1 || g_samplecount++ < g_num_samples)
    {
        g_cfgs[g_primary].renderer->Render(*g_scene.get());
    }

    //if (std::chrono::duration_cast<std::chrono::seconds>(time - updatetime).count() > 1)
    //{
    for (int i = 0; i < g_cfgs.size(); ++i)
    {
        if (g_cfgs[i].type == ConfigManager::kPrimary)
            continue;

        int desired = 1;
        if (std::atomic_compare_exchange_strong(&g_ctrl[i].newdata, &desired, 0))
        {
            {
                //std::unique_lock<std::mutex> lock(g_ctrl[i].datamutex);
                //std::cout << "Start updating acc buffer\n"; std::cout.flush();
                g_cfgs[g_primary].context.WriteBuffer(0, g_outputs[g_primary].copybuffer, &g_outputs[i].fdata[0], g_window_width * g_window_height);
                //std::cout << "Finished updating acc buffer\n"; std::cout.flush();
            }

            CLWKernel acckernel = g_cfgs[g_primary].renderer->GetAccumulateKernel();

            int argc = 0;
            acckernel.SetArg(argc++, g_outputs[g_primary].copybuffer);
            acckernel.SetArg(argc++, g_window_width * g_window_width);
            acckernel.SetArg(argc++, g_outputs[g_primary].output->data());

            int globalsize = g_window_width * g_window_height;
            g_cfgs[g_primary].context.Launch1D(0, ((globalsize + 63) / 64) * 64, 64, acckernel);
        }
    }

    //updatetime = time;
    //}

    if (!g_interop)
    {
        g_outputs[g_primary].output->GetData(&g_outputs[g_primary].fdata[0]);

        float gamma = 2.2f;
        for (int i = 0; i < (int)g_outputs[g_primary].fdata.size(); ++i)
        {
            g_outputs[g_primary].udata[4 * i] = (unsigned char)clamp(clamp(pow(g_outputs[g_primary].fdata[i].x / g_outputs[g_primary].fdata[i].w, 1.f / gamma), 0.f, 1.f) * 255, 0, 255);
            g_outputs[g_primary].udata[4 * i + 1] = (unsigned char)clamp(clamp(pow(g_outputs[g_primary].fdata[i].y / g_outputs[g_primary].fdata[i].w, 1.f / gamma), 0.f, 1.f) * 255, 0, 255);
            g_outputs[g_primary].udata[4 * i + 2] = (unsigned char)clamp(clamp(pow(g_outputs[g_primary].fdata[i].z / g_outputs[g_primary].fdata[i].w, 1.f / gamma), 0.f, 1.f) * 255, 0, 255);
            g_outputs[g_primary].udata[4 * i + 3] = 1;
        }


        glActiveTexture(GL_TEXTURE0);

        glBindTexture(GL_TEXTURE_2D, g_texture);

        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, g_outputs[g_primary].output->width(), g_outputs[g_primary].output->height(), GL_RGBA, GL_UNSIGNED_BYTE, &g_outputs[g_primary].udata[0]);

        glBindTexture(GL_TEXTURE_2D, 0);
    }
    else
    {
        std::vector<cl_mem> objects;
        objects.push_back(g_cl_interop_image);
        g_cfgs[g_primary].context.AcquireGLObjects(0, objects);

        CLWKernel copykernel = g_cfgs[g_primary].renderer->GetCopyKernel();

        int argc = 0;
        copykernel.SetArg(argc++, g_outputs[g_primary].output->data());
        copykernel.SetArg(argc++, g_outputs[g_primary].output->width());
        copykernel.SetArg(argc++, g_outputs[g_primary].output->height());
        copykernel.SetArg(argc++, 2.2f);
        copykernel.SetArg(argc++, g_cl_interop_image);

        int globalsize = g_outputs[g_primary].output->width() * g_outputs[g_primary].output->height();
        g_cfgs[g_primary].context.Launch1D(0, ((globalsize + 63) / 64) * 64, 64, copykernel);

        g_cfgs[g_primary].context.ReleaseGLObjects(0, objects);
        g_cfgs[g_primary].context.Finish(0);
    }


    if (g_benchmark)
    {
        auto const kNumBenchmarkPasses = 100U;

        Baikal::Renderer::BenchmarkStats stats;
        g_cfgs[g_primary].renderer->RunBenchmark(*g_scene.get(), kNumBenchmarkPasses, stats);

        auto numrays = stats.resolution.x * stats.resolution.y;
        std::cout << "Baikal renderer benchmark\n";
        std::cout << "Number of primary rays: " << numrays << "\n";
        std::cout << "Primary rays: " << (float)(numrays / (stats.primary_rays_time_in_ms * 0.001f) * 0.000001f) << "mrays/s ( " << stats.primary_rays_time_in_ms << "ms )\n";
        std::cout << "Secondary rays: " << (float)(numrays / (stats.secondary_rays_time_in_ms * 0.001f) * 0.000001f) << "mrays/s ( " << stats.secondary_rays_time_in_ms << "ms )\n";
        std::cout << "Shadow rays: " << (float)(numrays / (stats.shadow_rays_time_in_ms * 0.001f) * 0.000001f) << "mrays/s ( " << stats.shadow_rays_time_in_ms << "ms )\n";
        g_benchmark = false;
    }

    glutPostRedisplay();

	g_material_changed = false;
	g_light_changed = false;
}

void RenderThread(ControlData& cd)
{
    auto renderer = g_cfgs[cd.idx].renderer;
    auto output = g_outputs[cd.idx].output;

    auto updatetime = std::chrono::high_resolution_clock::now();

    while (!cd.stop.load())
    {
        int result = 1;
        bool update = false;

        if (std::atomic_compare_exchange_strong(&cd.clear, &result, 0))
        {
            renderer->Clear(float3(0, 0, 0), *output);
            update = true;
        }

        renderer->Render(*g_scene.get());

        auto now = std::chrono::high_resolution_clock::now();

        update = update || (std::chrono::duration_cast<std::chrono::seconds>(now - updatetime).count() > 1);

        if (update)
        {
            g_outputs[cd.idx].output->GetData(&g_outputs[cd.idx].fdata[0]);
            updatetime = now;
            cd.newdata.store(1);
        }

        g_cfgs[cd.idx].context.Finish(0);
    }
}


void StartRenderThreads()
{
    for (int i = 0; i < g_cfgs.size(); ++i)
    {
        if (i != g_primary)
        {
            g_renderthreads.push_back(std::thread(RenderThread, std::ref(g_ctrl[i])));
            g_renderthreads.back().detach();
        }
    }

    std::cout << g_cfgs.size() << " OpenCL submission threads started\n";
}

void enable_cuda_build_cache(bool enable)
{
#ifdef _MSC_VER
	if (enable)
		_putenv("CUDA_CACHE_DISABLE=0");
	else
		_putenv("CUDA_CACHE_DISABLE=1");
#else // GCC
	if (enable)
		putenv("CUDA_CACHE_DISABLE=0");
	else
		putenv("CUDA_CACHE_DISABLE=1");
#endif
}

void OBJChageMenu(int option)
{
	g_objIndex = option;
	InitData();
}

void ENVChageMenu(int option)
{
	g_envIndex = option;
	InitData();
}

void Right_menu(int)
{
}

bool dirExists(const std::string& dirName_in)
{
	DWORD ftyp = GetFileAttributesA(dirName_in.c_str());
	if (ftyp == INVALID_FILE_ATTRIBUTES)
		return false;  //something is wrong with your path!

	if (ftyp & FILE_ATTRIBUTE_DIRECTORY)
		return true;   // this is a directory!

	return false;    // this is not a directory!
}

std::string getexepath()
{
	char result[MAX_PATH];
	auto str = std::string(result, GetModuleFileName(NULL, result, MAX_PATH));
	str.erase(str.begin() + str.find_last_of('\\'), str.end());
	return str;
}

void createGLUTMenus()
{
	std::string resourcesPath;
	std::string envmapsPath;
	std::string objsPath;
	
	auto str = getexepath();
	resourcesPath = str + std::string("\\..\\resources\\");
	envmapsPath = resourcesPath + std::string("envmaps\\");
	objsPath = resourcesPath + std::string("objs\\");

	if (!dirExists(resourcesPath))
	{
		fprintf(stderr, "error! resourcesPath not exist.\n");
		exit(0);
	}
	if (!dirExists(envmapsPath))
	{
		fprintf(stderr, "error! envmapsPath not exist.\n");
		exit(0);
	}
	if (!dirExists(objsPath))
	{
		fprintf(stderr, "error! objsPath not exist.\n");
		exit(0);
	}

	// get env maps
	{
		HANDLE hFind = INVALID_HANDLE_VALUE;
		WIN32_FIND_DATA ffd;
		LARGE_INTEGER filesize;
		DWORD dwError = 0;
		std::string search_path = envmapsPath + "/*.hdr";
		hFind = FindFirstFile(search_path.c_str(), &ffd);
		do
		{
			if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
			{
				fprintf(stderr, "  %s   <DIR>\n", ffd.cFileName);
			}
			else
			{
				filesize.LowPart = ffd.nFileSizeLow;
				filesize.HighPart = ffd.nFileSizeHigh;
				fprintf(stderr, "  %s   %ld bytes\n", ffd.cFileName, filesize.QuadPart);
				envPaths.push_back(envmapsPath + ffd.cFileName);
			}
		} while (FindNextFile(hFind, &ffd) != 0);

		dwError = GetLastError();
		if (dwError != ERROR_NO_MORE_FILES)
		{
			fprintf(stderr, "FindFirstFile");
		}
		FindClose(hFind);
	}

	// get objs
	{
		HANDLE hFind = INVALID_HANDLE_VALUE;
		WIN32_FIND_DATA ffd;
		LARGE_INTEGER filesize;
		DWORD dwError = 0;
		std::string search_path = objsPath + "/*.*";
		hFind = FindFirstFile(search_path.c_str(), &ffd);
		do
		{
			if ((ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) &&
				strcmp(ffd.cFileName, ".") != 0 &&
				strcmp(ffd.cFileName, "..") != 0)
			{
				fprintf(stderr, "  %s   <DIR>\n", ffd.cFileName);

				OBJResources objres;
				objres.basePath = objsPath + ffd.cFileName + "\\";

				HANDLE hFind2 = INVALID_HANDLE_VALUE;
				WIN32_FIND_DATA ffd2;
				hFind2 = FindFirstFile((objres.basePath + "/*.obj").c_str(), &ffd2);
				if (!(ffd2.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
				{
					objres.objFilePath = objres.basePath + ffd2.cFileName;
					objPaths.push_back(objres);
				}
			}
		} while (FindNextFile(hFind, &ffd) != 0);

		dwError = GetLastError();
		if (dwError != ERROR_NO_MORE_FILES)
		{
			fprintf(stderr, "FindFirstFile");
		}
		FindClose(hFind);
	}

	int objMenu;
	int envMenu;

	// create the menu and
	// tell glut that "processMenuEvents" will
	// handle the events
	objMenu = glutCreateMenu(OBJChageMenu);
	for (int i = 0; i < objPaths.size(); ++i)
	{
		glutAddMenuEntry(objPaths[i].basePath.c_str(), i);
	}

	envMenu = glutCreateMenu(ENVChageMenu);
	for (int i = 0; i < envPaths.size(); ++i)
	{
		glutAddMenuEntry(envPaths[i].c_str(), i);
	}

	glutCreateMenu(Right_menu);
	glutAddSubMenu("Change OBJ", objMenu);
	glutAddSubMenu("Change ENV", envMenu);

	// attach the menu to the right button
	glutAttachMenu(GLUT_RIGHT_BUTTON);
}

int main(int argc, char * argv[])
{
	enable_cuda_build_cache(false);

    // Command line parsing
    //char* path = GetCmdOption(argv, argv + argc, "-p");
    //g_path = path ? path : g_path;

    //char* modelname = GetCmdOption(argv, argv + argc, "-f");
    //g_modelname = modelname ? modelname : g_modelname;

    //char* envmapname = GetCmdOption(argv, argv + argc, "-e");
    //g_envmapname = envmapname ? envmapname : g_envmapname;

    char* width = GetCmdOption(argv, argv + argc, "-w");
    g_window_width = width ? atoi(width) : g_window_width;

    char* height = GetCmdOption(argv, argv + argc, "-h");
    g_window_height = width ? atoi(height) : g_window_height;

    char* aorays = GetCmdOption(argv, argv + argc, "-ao");
    g_ao_radius = aorays ? (float)atof(aorays) : g_ao_radius;

    char* bounces = GetCmdOption(argv, argv + argc, "-nb");
    g_num_bounces = bounces ? atoi(bounces) : g_num_bounces;

    char* camposx = GetCmdOption(argv, argv + argc, "-cpx");
    g_camera_pos.x = camposx ? (float)atof(camposx) : g_camera_pos.x;

    char* camposy = GetCmdOption(argv, argv + argc, "-cpy");
    g_camera_pos.y = camposy ? (float)atof(camposy) : g_camera_pos.y;

    char* camposz = GetCmdOption(argv, argv + argc, "-cpz");
    g_camera_pos.z = camposz ? (float)atof(camposz) : g_camera_pos.z;

    char* camatx = GetCmdOption(argv, argv + argc, "-tpx");
    g_camera_at.x = camatx ? (float)atof(camatx) : g_camera_at.x;

    char* camaty = GetCmdOption(argv, argv + argc, "-tpy");
    g_camera_at.y = camaty ? (float)atof(camaty) : g_camera_at.y;

    char* camatz = GetCmdOption(argv, argv + argc, "-tpz");
    g_camera_at.z = camatz ? (float)atof(camatz) : g_camera_at.z;

    char* envmapmul = GetCmdOption(argv, argv + argc, "-em");
    g_envmapmul = envmapmul ? (float)atof(envmapmul) : g_envmapmul;

    char* numsamples = GetCmdOption(argv, argv + argc, "-ns");
    g_num_samples = numsamples ? atoi(numsamples) : g_num_samples;

    char* camera_aperture = GetCmdOption(argv, argv + argc, "-a");
    g_camera_aperture = camera_aperture ? (float)atof(camera_aperture) : g_camera_aperture;

    char* camera_dist = GetCmdOption(argv, argv + argc, "-fd");
    g_camera_focus_distance = camera_dist ? (float)atof(camera_dist) : g_camera_focus_distance;

    char* camera_focal_length = GetCmdOption(argv, argv + argc, "-fl");
    g_camera_focal_length = camera_focal_length ? (float)atof(camera_focal_length) : g_camera_focal_length;

    char* interop = GetCmdOption(argv, argv + argc, "-interop");
    g_interop = interop ? (atoi(interop) > 0) : g_interop;

    char* cspeed = GetCmdOption(argv, argv + argc, "-cs");
    g_cspeed = cspeed ? atof(cspeed) : g_cspeed;


    char* cfg = GetCmdOption(argv, argv + argc, "-config");

    if (cfg)
    {
        if (strcmp(cfg, "cpu") == 0)
            g_mode = ConfigManager::Mode::kUseSingleCpu;
        else if (strcmp(cfg, "gpu") == 0)
            g_mode = ConfigManager::Mode::kUseSingleGpu;
        else if (strcmp(cfg, "mcpu") == 0)
            g_mode = ConfigManager::Mode::kUseCpus;
        else if (strcmp(cfg, "mgpu") == 0)
            g_mode = ConfigManager::Mode::kUseGpus;
        else if (strcmp(cfg, "all") == 0)
            g_mode = ConfigManager::Mode::kUseAll;
    }

    if (aorays)
    {
        g_num_ao_rays = atoi(aorays);
        g_ao_enabled = true;
    }

    if (CmdOptionExists(argv, argv + argc, "-r"))
    {
        g_progressive = true;
    }

    // GLUT Window Initialization:
    glutInit(&argc, (char**)argv);
    glutInitWindowSize(g_window_width, g_window_height);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
    glutCreateWindow("App");

	createGLUTMenus();

#ifndef __APPLE__
    GLenum err = glewInit();
    if (err != GLEW_OK)
    {
        std::cout << "GLEW initialization failed\n";
        return -1;
    }
#endif

    try
    {
        InitGraphics();
        InitCl();
        InitData();

        // Register callbacks:
        glutDisplayFunc(Render);
        glutReshapeFunc(Reshape);

		glutKeyboardFunc(OnNormalKeys);
		glutKeyboardUpFunc(OnNormalKeysUp);
        glutSpecialFunc(OnKey);
        glutSpecialUpFunc(OnKeyUp);
        glutMouseFunc(OnMouseButton);
		glutMouseWheelFunc(OnMouseWheel);
        glutMotionFunc(OnMouseMove);
        glutIdleFunc(Update);

        StartRenderThreads();

        glutMainLoop();

        for (int i = 0; i < g_cfgs.size(); ++i)
        {
            if (i == g_primary)
                continue;

            g_ctrl[i].stop.store(true);
        }
    }
    catch (std::runtime_error& err)
    {
        std::cout << err.what();
        return -1;
    }

    return 0;
}

char* GetCmdOption(char ** begin, char ** end, const std::string & option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

bool CmdOptionExists(char** begin, char** end, const std::string& option)
{
    return std::find(begin, end, option) != end;
}

void ShowHelpAndDie()
{
    std::cout << kHelpMessage << "\n";
}

void SaveImage(std::string const& name, float3 const* data, int w, int h)
{


}

void SaveFrameBuffer(std::string const& name, float3 const* data)
{
    OIIO_NAMESPACE_USING;

    std::vector<float3> tempbuf(g_window_width * g_window_height);
    tempbuf.assign(data, data + g_window_width*g_window_height);

    ImageOutput* out = ImageOutput::create(name);

    if (!out)
    {
        throw std::runtime_error("Can't create image file on disk");
    }

    ImageSpec spec(g_window_width, g_window_height, 3, TypeDesc::FLOAT);

    out->open(name, spec);
    out->write_image(TypeDesc::FLOAT, &tempbuf[0], sizeof(float3));
    out->close();
}
