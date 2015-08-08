#define TWO_PI 6.2831853
#define VERTICAL_FOV 60.


vec2 normalize_pixel_coords(vec2 pixel_coords) {
    return (pixel_coords * 2. - iResolution.xy) / iResolution.y;
}

float map(vec3 p) {
    vec3 box_center = vec3(0.);
    vec3 box_radius = vec3(1., 1., 1.);
    vec3 t = min(max(p, box_center - box_radius), box_center + box_radius);
    return distance(p, t) - 0.5;
}

vec3 map_normal(vec3 p, float pval) {
    vec2 offset = vec2(0.01, 0.);
    vec3 diff = vec3(
        map(p + offset.xyy),
        map(p + offset.yxy),
        map(p + offset.yyx)) - pval;
    return normalize(diff);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec3 camera_pos = vec3(0., -1., -8.) + vec3(normalize_pixel_coords(iMouse.xy), 0.) * 10.;
    
    vec3 camera_target = vec3(0., 0., 0.);
    vec3 camera_dir = normalize(camera_target - camera_pos);
    
    vec3 camera_right = cross(vec3(0., 1., 0.), camera_dir);
    vec3 camera_up    = cross(camera_dir, camera_right);
    
    vec2 uv = normalize_pixel_coords(fragCoord);
    float ray_spread = tan((VERTICAL_FOV / 360. * TWO_PI) / 2.);
    vec3 ray_dir = camera_dir + ((uv.x * camera_right) + (uv.y * camera_up)) * ray_spread;
    ray_dir = normalize(ray_dir);
    
    vec3 bg = ray_dir * .5 + .5;
    
    float ray_len = 0.;
    float dist = 123.;
    for (int i = 0; i < 50; i++) {
        if (ray_len > 100. || dist < 0.01) continue; 
        ray_len += dist = map(camera_pos + ray_len * ray_dir);
    }
    
    vec3 col = bg;
    if (dist < 0.02) {
        vec3 normal = map_normal(camera_pos + ray_len * ray_dir, dist);
        col = vec3(-normal.z);
    }
    //col = vec3(dist);
    
	fragColor = vec4(col, 1.);
}