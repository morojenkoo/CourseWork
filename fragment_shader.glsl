#version 330 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D rayTexture;

void main()
{
    FragColor = texture(rayTexture, TexCoord);
}