shader_type canvas_item;

uniform bool spin;
uniform float spin_speed;
uniform vec3 spin_amp;

uniform float precision;
uniform vec3 angles;
uniform vec3 rotation_offset;
uniform vec3 pt;
uniform float amplitude;
uniform bool orthogonal;
uniform float fov_amp;
uniform sampler2D shape_texture: filter_nearest;
uniform sampler2D color_texture: filter_nearest;
uniform sampler2D back_shape_texture: filter_nearest;
uniform sampler2D back_color_texture: filter_nearest;
uniform vec2 color_offset;
uniform vec2 color_scale;

vec3 rotated_point(vec3 ang, vec3 this_pt){
	float a_x = ang.x;
	float a_y = ang.y;
	float a_z = ang.z;

	mat3 Rx = mat3(vec3(1.0,0.0,0.0), vec3(0, cos(a_x), sin(a_x)), vec3(0, -sin(a_x), cos(a_x)));
	mat3 Ry = mat3(vec3(cos(a_y), 0.0, -sin(a_y)), vec3(0.0, 1.0, 0.0), vec3(sin(a_y), 0.0, cos(a_y)));
	mat3 Rz = mat3(vec3(cos(a_z), sin(a_z), 0.0), vec3(-sin(a_z), cos(a_z), 0.0), vec3(0.0, 0.0, 1.0));

	this_pt = this_pt * Rx;
	this_pt = this_pt * Ry;
	this_pt = this_pt * Rz;

	return this_pt;
}

vec4 raycasting(vec2 uv, vec3 angs) {
	float height = -1.0;
	vec3 ray_start = rotated_point(angs, vec3(uv, 5) - pt);
	vec3 ray = ray_start;
	vec3 ray_dir;
	if (orthogonal == true) {
		ray_dir = rotated_point(angs, -normalize(vec3(0,0,1))*0.01);
	} else {
		ray_dir = rotated_point(angs, -normalize(vec3(0,0,1) + -vec3(uv, 0)*fov_amp)*0.01);
	}

	while (length(ray - ray_start) < 10.0) {
		ray = ray + ray_dir;
		float test_h = texture(shape_texture, ray.xy + pt.xy).r * amplitude;
		float back_test_h = texture(back_shape_texture, ray.xy + pt.xy).r * amplitude;
		if (ray.z < test_h && ray.z >= 0.0){
			return texture(color_texture, (ray.xy + pt.xy) * color_scale + color_offset);
		}else if (ray.z > -back_test_h && ray.z < 0.0){
			return texture(back_color_texture, (ray.xy + pt.xy) * color_scale + color_offset);
		}
	}

	return vec4(1.0, 0.0, 0.0, 0.0);
}

void fragment() {
	//COLOR = raycasting(UV, vec3(angles));

	if (spin == true){
		COLOR = raycasting(UV, vec3(sin(TIME * spin_speed)*spin_amp.x, cos(TIME * spin_speed)*spin_amp.y, cos(1.0+TIME * spin_speed)*spin_amp.z) + rotation_offset);
	} else {
		COLOR = raycasting(UV, vec3(angles));
	}

}
