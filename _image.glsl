#define TWO_PI 6.2831853
#define VERTICAL_FOV 40.

vec2 normalize_pixel_coords(vec2 pixel_coords) {
    return (pixel_coords * 2. - iResolution.xy) / iResolution.y;
}

float map(vec3 p) {
    vec3 box_center = vec3(0.);
    vec3 box_radius = vec3(1., 1., 1.);
    vec3 t = min(max(p, box_center - box_radius), box_center + box_radius);
    return distance(p, t) - 0.5;
}

vec3 map_normal(vec3 p, float epsilon) {
    vec2 offset = vec2(epsilon, 0.);
    vec3 diff = vec3(
        map(p + offset.xyy) - map(p - offset.xyy),
        map(p + offset.yxy) - map(p - offset.yxy),
        map(p + offset.yyx) - map(p - offset.yyx)
    );
    return normalize(diff);
}

float cocSize(float dist) {
    return 1. / iResolution.y * dist;
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
    
    //float coc = sin(iGlobalTime * 2.) * .4 + .5;
    float coc = 0.1;
    float ray_len = 0.;
    float map_dist = 123.;
    for (int i = 0; i < 50; i++) {
        if (ray_len > 100. || col.a > 0.9) continue; 
        map_dist = map(camera_pos + ray_len * ray_dir);
        float coc = cocSize(ray_len);
        
        if(abs(map_dist) < coc) {
            vec3 normal = map_normal(camera_pos + ray_len * ray_dir, coc);
            float toward_camera = -dot(normal, ray_dir);
            if (toward_camera > 0.) {
                float alpha = toward_camera * smoothstep(coc, 0., map_dist);
                vec3 surface_color = vec3(0.);
                surface_color += vec3(0.8, 0.1, 0.1) * max(dot(normal, normalize(vec3(-0.4, 1., -0.3))), 0.);
                
                col = vec4((col.rgb * col.a + surface_color * alpha) / (col.a + alpha), mix(col.a, 1., alpha));
            }
        }
        
        map_dist = max(map_dist - .5 * coc, .5 * coc);
        ray_len += map_dist;
    }
    
    col = vec4(sqrt(mix(bg, col.rgb, col.a)), 1.);
    
	fragColor = vec4(col);
}