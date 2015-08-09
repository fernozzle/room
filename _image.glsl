#define TWO_PI 6.2831853
#define VERTICAL_FOV 40.
#define MAX_ALPHA .9

vec2 normalize_pixel_coords(vec2 pixel_coords) {
    return (pixel_coords * 2. - iResolution.xy) / iResolution.y;
}

float box_map(vec3 p, vec3 center, vec3 size, float radius) {
    vec3 lower_bound = center - size;
    vec3 upper_bound = center + size;
    vec3 temp = min(max(p, lower_bound), upper_bound);
    return distance(p, temp) - radius;
}

float sphere_map(vec3 p, vec3 center, float radius) {
    return distance(p, center) - radius;
}

// Material data: 3 channels & index
//   Index [0, 1) = smoothness; RGB = albedo
float map(in vec3 p, out vec4 material) {
    float dist = box_map(p, vec3(0.), vec3(1.), 0.5);
    material = vec4(0.4, 0.8, 0.9, 0.);
    
    float new_dist = sphere_map(p, vec3(sin(iGlobalTime * 5.) * .8 + 2., 0., 0.), 1.);
    if (new_dist < dist) {
        dist = new_dist;
        material = vec4(0.8, 0.1, 0.1, 0.);
    }
    
    new_dist = sphere_map(p, vec3(0., 3., 0.), .5);
    if (new_dist < dist) {
        dist = new_dist;
        material = vec4(0.9, 0.9, 0.3, 0.);
    }
    
    return dist;
}

vec3 map_normal(vec3 p, float epsilon) {
    vec4 mat;
    vec2 offset = vec2(epsilon, 0.);
    vec3 diff = vec3(
        map(p + offset.xyy, mat) - map(p - offset.xyy, mat),
        map(p + offset.yxy, mat) - map(p - offset.yxy, mat),
        map(p + offset.yyx, mat) - map(p - offset.yyx, mat)
    );
    return normalize(diff);
}

float coc_kernel(float width, float dist) {
    return smoothstep(width, -width, dist);
}

float soft_shadow(vec3 p, vec3 dir, float softness, float coc, float start_len) {
    float brightness = 1.;
    float len = coc + start_len;
    vec4 mat;
    for (int i = 0; i < 20; i++) {
        float map_dist = map(p + dir * len, mat);
        float coc2 = coc + len * softness;
        brightness *= 1. - coc_kernel(coc2, map_dist);
        len += map_dist + .5 * coc;
    }
    return clamp(brightness, 0., 1.);
}

float cocSize(float dist) {
    return (sin(iGlobalTime * 3.) * 4. + 5.) / iResolution.y * dist;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec3 camera_pos = vec3(0., -1., -8.) + vec3(normalize_pixel_coords(iMouse.xy), 0.) * 20.;
    
    vec3 camera_target = vec3(0., 0., 0.);
    vec3 camera_dir = normalize(camera_target - camera_pos);
    
    vec3 camera_right = cross(vec3(0., 1., 0.), camera_dir);
    vec3 camera_up    = cross(camera_dir, camera_right);
    
    vec2 uv = normalize_pixel_coords(fragCoord);
    float ray_spread = tan((VERTICAL_FOV / 360. * TWO_PI) / 2.);
    vec3 ray_dir = camera_dir + ((uv.x * camera_right) + (uv.y * camera_up)) * ray_spread;
    ray_dir = normalize(ray_dir);
    
    vec3 bg = ray_dir * .5 + .5;
    
    vec4 col = vec4(0., 1., 0., 0.);
    
    vec4 mat;
    float ray_len = 0.;
    float map_dist = 123.;
    int iters = 0;
    for (int i = 0; i < 50; i++) {
        if (ray_len > 100. || col.a > MAX_ALPHA) continue; 
        float coc = cocSize(ray_len);
        vec3 point = camera_pos + ray_len * ray_dir;
        map_dist = map(point, mat);
        
        
        if(abs(map_dist) < coc) {
            vec3 normal = map_normal(point, coc);
            float toward_camera = -dot(normal, ray_dir);
            if (toward_camera > 0.) {
                float alpha = toward_camera * coc_kernel(coc, map_dist);
                vec3 surface_color = vec3(0.);
                vec3 light_direction = normalize(vec3(-0.4, 1., -0.3));
                surface_color += mat.rgb * (.05 + .95 * ( max(dot(normal, light_direction), 0.) * soft_shadow(point, light_direction, .0, coc, .01)));
                
                // "Alpha-under"ing surface_color/alpha beneath col
                float added_coverage = alpha * (1. - col.a);
                col = vec4(
                    (col.rgb * col.a + surface_color * added_coverage) / (col.a + added_coverage),
                    mix(col.a, 1., alpha)
                );
            }
        }
        
        iters++;
        ray_len += max(map_dist - .5 * coc, .5 * coc);
    }
    
    col = vec4(sqrt(mix(bg, col.rgb, min(col.a / MAX_ALPHA, 1.))), 1.);
    
	fragColor = vec4(col);
}