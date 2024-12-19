#version 430
precision mediump float;
in vec2 o_texcoord;

layout(binding = 0) uniform sampler2D tex_sampler;
layout(binding = 1) uniform sampler2D tex_depth;

uniform float IGlobalTime;
uniform vec2 iResolution;

uniform float focus_dis;
uniform vec2 focus_disScale;
uniform vec3 gamma_hardness_CoC;

layout(location = 0) out vec4 fragColor;

#define USE_GAMMA // 使用gamma校正
// const float gamma = 4.2; // 4.2;
// const float hardness = 0.9; // 0.0-1.0

float getDistance(sampler2D inTex, vec2 uv)
{
	vec3 rgb = texture(inTex, uv).rgb;
	return rgb.r;
}
float distance_to_interval(float x, float a, float b) {
    if (x >= a && x <= b) {
        return 0.0;
    } else {
        return min(abs(x - a), abs(x - b));
    }
}

float intensity(vec2 p)
{
    return smoothstep(1.0, gamma_hardness_CoC.y, distance(p, vec2(0.0)));
}

vec3 blur(sampler2D tex, float size, int res, vec2 uv, float ratio)
{
    float div = 0.0;
    vec3 accumulate = vec3(0.0);
    
    for(int iy = 0; iy < res; iy++)
    {
        float y = (float(iy) / float(res))*2.0 - 1.0;
        for(int ix = 0; ix < res; ix++)
        {
            float x = (float(ix) / float(res))*2.0 - 1.0;
            vec2 p = vec2(x, y);
            float i = intensity(p);
            
            div += i;
			#ifdef USE_GAMMA
            	accumulate += pow(texture(tex, uv+p*size*vec2(1.0, ratio)).rgb, vec3(gamma_hardness_CoC.x)) * i;
			#else
				accumulate += texture(tex, uv+p*size*vec2(1.0, ratio)).rgb * i;
			#endif
        }
    }
    #ifdef USE_GAMMA
    	return pow(accumulate / vec3(div), vec3(1.0 / gamma_hardness_CoC.x));
	#else
		return accumulate / vec3(div);
	#endif
}

void main()
{   
    vec2 uv = o_texcoord;
    
    vec3 ori_color = texture(tex_sampler, uv).rgb;
    float centerDepth = getDistance(tex_depth, vec2(uv.x-(0.0/iResolution.x), uv.y));

	// float near = max(abs(sin(iTime*0.3)) - focusScale, 0.0);
	// float far = min(abs(sin(iTime*0.3)) + focusScale, 1.0);
    float mid = focus_disScale.x < 0.01 ? focus_dis : 1.0-focus_disScale.x;
    float near = mid - 0.3*focus_disScale.y;
    float far = mid + focus_disScale.y;

	float dis = distance_to_interval(centerDepth, near, far) * gamma_hardness_CoC.z; // 0.0-0.02    * 0.025
    vec3 result = blur(tex_sampler, dis, 32, uv, iResolution.x/iResolution.y);
    // result = mix(result, ori_color, rvm_mask); // 与人像抠图融合

    fragColor = vec4(result, 1.0);

    // fragColor = texture(tex_depth, uv);
    // fragColor = vec4(vec3(centerDepth), 1.0);
    // if(centerDepth > 0.7 && centerDepth < 0.75)
    // {
    //     fragColor = vec4(vec3(centerDepth), 1.0);   
    // }else{
    //     fragColor = vec4(vec3(0.0), 1.0);   
    // }
}
