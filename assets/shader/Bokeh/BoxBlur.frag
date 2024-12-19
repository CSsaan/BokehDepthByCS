#version 430
precision mediump float;
in vec2 o_texcoord;

layout(binding = 0) uniform sampler2D tex_sampler;

layout(location = 0) out vec4 fragColor;

uniform vec2 iResolution;
uniform int even_odd;

const float radius = 1.0; 

void main()
{   
    vec2 uv = o_texcoord;

    vec4 color = vec4(0.0);
    int count = 0;

    if(even_odd == 0)
    {
        for (float x = uv.x-radius/iResolution.x; x <= uv.x+radius/iResolution.x; x += 1.0/float(iResolution.x)) {
            color += texture(tex_sampler, vec2(x, uv.y));
            count += 1;
        }
    } 
    else {
        for (float y = uv.y-radius/iResolution.y; y <= uv.y+radius/iResolution.y; y += 1.0/float(iResolution.y)) {
            color += texture(tex_sampler, vec2(uv.x, y));
            count += 1;
        }
    }
    

    if (count == 0) {
        fragColor = texture(tex_sampler, uv);
    } else {
        fragColor = color / float(count);
    }

}
