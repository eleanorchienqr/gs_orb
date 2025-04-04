#version 460 core

layout(location = 0) out vec4 color;

in vec2 v_TexCoord;
// in vec4 v_Color;

uniform sampler2D u_Texture;
//uniform vec4 u_Color;

void main()
{
	vec4 texColor = texture(u_Texture, v_TexCoord);
	color = texColor;
};