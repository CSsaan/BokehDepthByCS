/** 
 * Code taken from www.learnopengl.com
 * Contributors:
 *      * Arthur CS (author)
 **/

#pragma once

#include <iostream>
#include <string>
#include <GL/glew.h>

//#ifndef STB_IMAGE_IMPLEMENTATION
//#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
//#endif // !STB_IMAGE_IMPLEMENTATION
#include "glError.hpp"

class Texture final
{
public:
    Texture(const std::string & image_dir){
        load(image_dir);
    };
    Texture(int width, int height, GLenum format) {
        create(width, height, format);
    };
    ~Texture() {
        if(texture_id != 0) {
            glDeleteTextures(1, &texture_id);
            texture_id = 0;
        }
    }

    void bind(GLuint index = 0) const {
        if(texture_id != 0) {
            glActiveTexture(GL_TEXTURE0 + index);
            glBindTexture(GL_TEXTURE_2D, texture_id);
        }
    }

    GLuint getTexId() const {
        return texture_id;
    }

    int getWidth() const {
        return width;
    }

    int getHeight() const {
        return height;
    }

    int getComponents() const {
        return components;
    }

    bool update(const std::string & file_name) {
        if(file_name.empty()) {
            return false;
        }

        if(texture_id != 0) {
            glDeleteTextures(1, &texture_id);
            texture_id = 0;
        }  
        std::cout << "updating picture:" << file_name << std::endl;
        return load(file_name);
    }

    bool update(int width, int height, GLenum format) {

        if (texture_id != 0) {
            glDeleteTextures(1, &texture_id);
            texture_id = 0;
        }
        std::cout << "updating texture" << std::endl;
        return create(width, height, format);
    }

private:
    bool load(const std::string & file_name) {
        if(file_name.empty()) {
            return false;
        }

        unsigned char* pixels = stbi_load((file_name).c_str(), &width, &height, &components, 0);
        if(pixels == nullptr) {
            std::cout << "Could not load file " << file_name << std::endl;
            return false;
        }

        GLenum internal_format = GL_RGB;
        GLenum format = GL_RGB;
        if(components == 1) {
            internal_format = GL_R8;
            format = GL_RED;
        }
        else if(components == 2) {
            internal_format = GL_RG8;
            format = GL_RG;
        }
        else if(components == 3) {
            internal_format = GL_RGB8;
            format = GL_RGB;
        }
        else if(components == 4) {
            internal_format = GL_RGBA8;
            format = GL_RGBA;
        }
        glGenTextures(1, &texture_id);
        glBindTexture(GL_TEXTURE_2D, texture_id);
        glTexStorage2D(GL_TEXTURE_2D, 2 /* mip map levels */, internal_format, width, height);
        glTexSubImage2D(GL_TEXTURE_2D, 0 /* mip map level */, 0 /* xoffset */, 0 /* yoffset */, width, height, format, GL_UNSIGNED_BYTE, pixels);
        glGenerateMipmap(GL_TEXTURE_2D);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, use_linear ? GL_LINEAR_MIPMAP_LINEAR : GL_NEAREST_MIPMAP_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, use_linear ? GL_LINEAR : GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        std::cout << "Texture '" << file_name << "' loaded " << "(width:" << width << ", height:" << height << ")" << std::endl;

        stbi_image_free(pixels);
        return true;
    }

    bool create(int _width, int _height, GLenum format) {
        GLenum internal_format = { GL_RGB8 };
        if (format == GL_BLUE) {
            internal_format = GL_R8;
        }
        else if (format == GL_RG) {
            internal_format = GL_RG8;
        }
        else if (format == GL_BGR) {
            internal_format = GL_RGB8;
        }
        else if (format == GL_RGBA) {
            internal_format = GL_RGBA8;
        }
        //glGenTextures(1, &texture_id);
        //glBindTexture(GL_TEXTURE_2D, texture_id);
        //glTexStorage2D(GL_TEXTURE_2D, 2 /* mip map levels */, internal_format, width, height);
        //glTexImage2D(GL_TEXTURE_2D, 0 /* mip map level */, internal_format, width, height, 0, format, GL_UNSIGNED_BYTE, NULL);
        //glGenerateMipmap(GL_TEXTURE_2D);
        //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, use_linear ? GL_LINEAR_MIPMAP_LINEAR : GL_NEAREST_MIPMAP_NEAREST);
        //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, use_linear ? GL_LINEAR : GL_NEAREST);
        //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        //glBindTexture(GL_TEXTURE_2D, GL_NONE);


        width = _width;
        height = _height;
        glGenTextures(1, &texture_id);
        glBindTexture(GL_TEXTURE_2D, texture_id);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, internal_format, _width, _height, 0, format, GL_UNSIGNED_BYTE, NULL);
        glBindTexture(GL_TEXTURE_2D, GL_NONE);
        glCheckError(__FILE__, __LINE__);
        return true;
    }
    
private:
    GLuint texture_id = 0;
    bool use_linear = true;
    int width, height, components;
};
