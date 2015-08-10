#define PI 3.141592
#define TWO_PI 6.2831853
//#define VERTICAL_FOV 80.
#define MAX_ALPHA .9
#define NORMAL_EPSILON .01
// Compensate for distorted distance fields
#define STEP_SCALE 0.8

// iq's texture noise
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = texture2D( iChannel0, (uv+0.5)/256.0, -100.0 ).yx;
	return mix( rg.x, rg.y, f.z );
}

vec2 normalize_pixel_coords(vec2 pixel_coords) {
    return (pixel_coords * 2. - iResolution.xy) / iResolution.x;
}

float box_map(vec3 p, vec3 size, float radius) {
    size *= .5;
    vec3 temp = clamp(p, -size, size);
    return distance(p, temp) - radius;
}
float sphere_map(vec3 p, vec3 center, float radius) {
    return distance(p, center) - radius;
}

float walls_map(vec3 p, vec2 size) {
    p.xy = abs(p.xy) - size * .5;
    return -max(p.x, p.y);
}
float pillar_map(vec3 p, float radius) {
    return length(p.xy) - radius;
}
float shelf_map(vec3 p) {
    if (p.y < -p.x) p.xyz = p.yxz * vec3(-1., 1., 1.);
    
    float shelf_spacing = .33;
    float shelf_height = floor(p.z / shelf_spacing + .5) * shelf_spacing;
    
    float shelf_radius = .48 - .2 * p.z;
    float l = length(p.xy);
    vec3 shelf_point = vec3((p.xy / l) * min(l, shelf_radius), shelf_height);
    float shelf_distance = distance(p, shelf_point);
    
    float support_distance = distance(p.xy, vec2(clamp(p.x, shelf_radius - .04, shelf_radius), 0.));
    float back_distance    = distance(p.xy, vec2(min(p.x, .04), 0.));
    
    return min(shelf_distance, min(support_distance, back_distance)) - .02;
}
float couch_map(vec3 p) {
    
    // Seat
    vec2 seat_near_center = vec2(clamp(p.x, -.75, .75), -.0);
    seat_near_center.y += .2 / (pow(p.x, 2.) * 2. + 1.);
    
    vec2 p_rel = p.xy - seat_near_center;
    float l = length(p_rel);
    vec2 seat_edge = seat_near_center + (p_rel / l) * min(l, .42);
    float seat_distance = distance(p, vec3(seat_edge, min(p.z, .33 - l*l*.3))) - .02;
    
    // Back rest
    vec3 p_transf = p;
    p_transf.y += pow(p_transf.x, 2.) * .15;
    p_transf.x *= 1. - p_transf.y * .2;
    
    vec3 back_near_center = vec3(clamp(p_transf.x, -.86, .86), .6 + .15 * p_transf.z, min(p_transf.z, .72));
    p_rel = p_transf.yz - back_near_center.yz;
    l = length(p_rel);
    vec3 back_edge = vec3(back_near_center.x, back_near_center.yz + (p_rel / l) * min(l, .11));
    float back_distance = distance(p_transf, back_edge) - .03;
    
    // Back rest wrinkles
    
    p.x += .2;
    float wrinkle_skew = p.z - .6*p.x*p.x/p.z;
    float wrinkle = (sin((wrinkle_skew) * 60.) + 1.) * .005;
    wrinkle *= 1. / (pow(wrinkle_skew - .5, 2.) * 8. + 1.);
    wrinkle /= (pow(p.x, 4.)*5. + 1.);
    wrinkle *= smoothstep(.85, .7, p.z);
    wrinkle = smoothstep(-.05, .1, wrinkle) * .06;
    back_distance += wrinkle;
    
    return min(seat_distance, back_distance);
}
float curtain_map(vec3 p) {
    vec3 temp = vec3(clamp(p.x, -.6, .6), 0., p.z);
    float dist = distance(p, temp) - .2;
    dist += sin(p.x * 20. + sin(p.x * 6.2) * (5. + sin(p.z * 2.) * 3.)) * .03;
    return dist;
}
// These two are by the one and only iq
float smin( float a, float b, float k )
{
    float h = clamp( 0.5+0.5*(b-a)/k, 0.0, 1.0 );
    return mix( b, a, h ) - k*h*(1.0-h);
}
float sdCapsule( vec3 p, vec3 a, vec3 b, float r )
{
    vec3 pa = p - a, ba = b - a;
    float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
    return length( pa - ba*h ) - r;
}
//struct Person {
    
float person_map(vec3 p, out vec4 mat) {
    p.xy = -p.xy;
    p.x = abs(p.x); // Symmetrical
    
    // Head
    float dist = distance(p, vec3(.0, .01, .0)) - .08;
    dist = smin(dist, distance(p, vec3(.0, .06, -.05)) - .01, .08);
    float jaw_dist = sdCapsule(p, vec3(.05, .02, -.10), vec3(0., .08, -.11), .005);
    dist = smin(dist, jaw_dist, .1);
    float cheek_dist = distance(p, vec3(.04, .04, -.04)) - .01;
    dist = smin(cheek_dist, dist, .05);
    float nose_dist = sdCapsule(p, vec3(.0, .09, -.03), vec3(0., .11, -.06), .002);
    dist = smin(dist, nose_dist, .04);
    
    float neck_dist = sdCapsule(p, vec3(.0, .0, -.06), vec3(.0, .0, -.20), .04);
    dist = smin(dist, neck_dist, .02);
    
    float mult = 3.;
    mat = vec4(1. * mult, .77 * mult, .65 * mult, 2.);
    
    float eye_dist = length((p - vec3(.025, .10, -.02)) * vec3(1., 1., .8)) - .005;
    if (eye_dist < dist) {
        dist = eye_dist;
        mat = vec4(0., 0., 0., 0.3);
    }
    
    float body_top = -.14;
    float body_radius = (p.z - body_top) * -.15 + .045;
    float l = length(p.xy);
    vec3 body_near = vec3(p.xy / l * min(l, body_radius), clamp(p.z, -.5, body_top));
    float body_dist = distance(p, body_near) - .005;
    body_dist = smin(body_dist, sdCapsule(p, vec3(.0, .0, -.15), vec3(.2, .0, -.3), .04), .02);
    
    if (body_dist < dist) {
        dist = body_dist;
        mat = vec4(1., 1., 1., 0.9);
    }

    return dist;
}

// Material data: 3 channels & index
//   Index [0, 1) = smoothness; RGB = albedo
float map(in vec3 p, out vec4 material) {
    float dist = walls_map(p - vec3(-.55, -.6, 0.), vec2(5.5, 5.8));
    material = vec4(.77, .15, .16, 0.8);
    // Floor
    float new_dist = p.z;
    if (new_dist < dist) {
        dist = new_dist;
        material = vec4(.5, .27, .14, 0.4);
    }
    /*
    // Pillars
    new_dist = min(pillar_map(p - vec3(.7, 2.3, 0.), .12), pillar_map(p - vec3(-2.14, 2.3, 0.), .12));
    if (new_dist < dist) {
        dist = new_dist;
        material = vec4(.95, .94, .91, 0.5);
    }
    
    // Shelf
    new_dist = shelf_map(p - vec3(-3.3, 2.3, 0.));
    if (new_dist < dist) {
        dist = new_dist;
        material = vec4(.9, .52, .3, 0.8);
    }
    
    // Door
    new_dist = box_map(p - vec3(-3.3, 1.3, 1.09), vec3(0.1, .98, 2.16), .01);
    if (new_dist < dist) {
        dist = new_dist;
        material = vec4(.8, .8, .8, 0.5);
    }
    
    // Painting
    new_dist = box_map(p - vec3(-3.3, -.4, 1.5), vec3(.1, .8, .97), .01);
    if (new_dist < dist) {
        dist = new_dist;
        material = vec4(.9, .9, .9, 0.8);
    }
    
    // Couch
    new_dist = couch_map(p - vec3(.3, 1.0, 0.));
    if (new_dist < dist) {
        dist = new_dist;
        material = vec4(.76, .52, .33, 0.9);
    }
    
    // Curtains
    new_dist = curtain_map(p - vec3(-.8, 2.3, 1.5));
    if (new_dist < dist) {
        dist = new_dist;
        material = vec4(1., 1., 1., 1.);
    }*/
    
    // Person
    vec4 new_mat;
    new_dist = person_map(p - vec3(0., .5, 1.), new_mat);
    if (new_dist < dist) {
        dist = new_dist;
        material = new_mat;
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
    for (int i = 0; i < 10; i++) {
        float map_dist = map(p + dir * len, mat);
        float coc2 = coc + len * softness;
        brightness *= 1. - coc_kernel(coc2, map_dist);
        len += (map_dist + .5 * coc) * STEP_SCALE;
    }
    return clamp(brightness, 0., 1.);
}

float ao(vec3 p, vec3 normal, float coc) {
    float ao_size = .5;
    float brightness = 1.;
    float len = coc + .05;
    vec4 mat;
    for (int i = 0; i < 3; i++) {
        float map_dist = map(p + normal * len, mat);
        brightness *= clamp(map_dist / len + len * ao_size, 0., 1.);
        len += map_dist + .5 * coc;
    }
    return pow(brightness, .3);
}

vec3 shade_standard(vec3 albedo, float roughness, vec3 normal, vec3 light_dir, vec3 ray_dir) {
    
    float F0 = .5;
    float diffuse_specular_mix = .3;
    
    float nl = dot(normal, light_dir);
    float nv = dot(normal, -ray_dir);
    if (nl > 0. && nv > 0.)
    {
        vec3 haf = normalize(light_dir - ray_dir);
        float nh = dot(normal, haf); 
        float vh = dot(-ray_dir, haf);
        
        vec3 diffuse = albedo*nl;
        
        // Cook-Torrance
        float a = roughness * roughness;
        float a2 = a * a;
        float dn = nh * nh * (a2 - 1.) + 1.;
        float D = a2 / (PI * dn * dn);
        
        float k = pow(roughness + 1., 2.0) / 8.;
        float nvc = max(nv, 0.);
        float g1v = nvc / (nvc * (1. - k) + k);
        float g1l = nl  / (nl  * (1. - k) + k);
        float G = g1l * g1v;

        float F = F0 + (1. - F0) * exp2((-5.55473 * vh - 6.98316) * vh);
        
        float specular = (D * F * G) / (4. /* * nl */ * nv);
    	
        return mix(vec3(specular), diffuse, diffuse_specular_mix);
    }
    return vec3(0.);
}

float coc_size(float dist) {
    return 2. / iResolution.y * dist;
    //return (sin(iGlobalTime * 3.) * 2. + 3.) / iResolution.y * dist;
}

float length_pow(vec3 d, float p) {
    return pow(pow(d.x, p) + pow(d.y, p) + pow(d.z, p), 1. / p);
}

vec3 window_light_pos = vec3(-1., 1.8, 1.2);
vec3 light_standard(vec3 p, float coc, vec3 albedo, float roughness, vec3 normal, vec3 ray_dir) {
    vec3 surface_color = vec3(0.);
    vec3 light_pos;

    light_pos = window_light_pos;
    vec3 light_dir = normalize(light_pos - p);
    vec3 light_intensity;
    light_intensity = shade_standard(albedo, roughness, normal, light_dir, ray_dir) * soft_shadow(p, light_dir, .1, coc, .1);
    surface_color += light_intensity * vec3(0.85, 0.8, 0.9) * .8;

    light_pos = vec3(-3., -.57, 1.6);
    light_dir = normalize(light_pos - p);
    light_intensity = shade_standard(albedo, roughness, normal, light_dir, ray_dir);
    surface_color += light_intensity * vec3(.4, .6, .8) * .1;

    light_pos = vec3(2., -1.17, 1.25);
    light_dir = normalize(light_pos - p);
    light_intensity = shade_standard(albedo, roughness, normal, light_dir, ray_dir);
    surface_color += light_intensity * vec3(1., 0.7, 0.5) * .4;
    
    return surface_color;
}

vec3 color_at(vec3 p, vec3 ray_dir, vec3 normal, float coc, vec4 mat) {
    vec3 surface_color = vec3(0.);
    
    if (mat.a < 1.) {
  	 	// Standard shading

        float amb_occ = ao(p, normal, coc);
        vec3 light_sum = light_standard(p, coc, mat.rgb, mat.a, normal, ray_dir);
        return light_sum * amb_occ;
    } else if (mat.a < 2.) {
        // Curtain shading
        vec3 wall_color = vec3(.3, .1, .1) * .3;
        float shade_fac = pow(dot(normal, vec3(0., -1., 0.)), 2.);
        shade_fac *= -dot(ray_dir, normal);
        float power = 2.;
        float windowness = pow(length_pow((p - vec3(-.8, 2.2, 2.)) * vec3(2., 1., 1.), 4.), 3.);
        vec3 transmission_color = pow(vec3(.3, .25, .2), vec3(windowness)) * 2.;
        surface_color = mix(wall_color, transmission_color, shade_fac);
        
        float stripe = smoothstep(-.1, .1, sin((p.z + cos(p.x * 11.) * .02) * 200.)) * .8;
        vec3 stripe_color = vec3(.08, .05, .06) * shade_fac/* + vec3(1.) * pow(shade_fac, 10.)*/;
        stripe_color = mix(stripe_color, wall_color, .5 * pow(1. - shade_fac, 5.));
        surface_color = mix(surface_color, stripe_color, stripe);
        return surface_color;
    } else {
        vec3 light_dir = normalize(window_light_pos - p);
        vec4 mat;
        float light = 1.;
        float soft = 0.;
        for (int i = 0; i < 8; i++) {
            float dist = map(p, mat);
            light *= smoothstep(-soft, soft, dist);
            p    += light_dir * .01;
            soft += .01;
        }
        vec3 subsurface_color = pow(vec3(.7,.3,.1), vec3(pow(light, -.2)));
        
        surface_color = light_standard(p, coc, mat.rgb, .7, normal, ray_dir);

        return mix(surface_color, subsurface_color, .3);
    }
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 mouse_normalized = normalize_pixel_coords(iMouse.xy);
    vec3 camera_pos = vec3(0., 0., 4.) + vec3(mouse_normalized.x * 2., 0., mouse_normalized.y * 8.);
     /*
    camera_pos = vec3(.232, .792, 1.10);
    camera_pos = mix(camera_pos, vec3(-.67, .11, 1.12), mouse_normalized.x);
	// */ 

    vec3 camera_target = vec3(0., .5, 1.);
     /*
    camera_target = vec3(-1.82, 1.72, .84);
    camera_target = mix(camera_target, vec3(-.17, 1.31, .70), mouse_normalized.x);
	// */
    
    vec3 camera_dir = normalize(camera_target - camera_pos);
    
    vec3 camera_right = normalize(cross(camera_dir, vec3(0., 0., 1.)));
    vec3 camera_up    = normalize(cross(camera_right, camera_dir));
    
    vec2 uv = normalize_pixel_coords(fragCoord);
    float fov = 80.;
     /*
    fov = 33.4;
    fov = mix(fov, 47.3, mouse_normalized.x);
	// */
    
    float ray_spread = tan((fov / 360. * TWO_PI) / 2.);
    vec3 ray_dir = camera_dir + ((uv.x * camera_right) + (uv.y * camera_up)) * ray_spread;
    ray_dir = normalize(ray_dir);
    
    vec3 bg = ray_dir * .5 + .5;
    
    vec4 col = vec4(0., 1., 0., 0.);
    
    float ray_len = 0.;
    float map_dist = 123.;
    int iters = 0;
    
    vec3 point;
    vec3 normal;
    float coc;
    vec4 mat;
    for (int i = 0; i < 50; i++) {
        if (ray_len > 100. || col.a > MAX_ALPHA) continue; 
        coc = coc_size(ray_len);
        point = camera_pos + ray_len * ray_dir;
        map_dist = map(point, mat);
        
        
        if(abs(map_dist) < coc) {
            normal = map_normal(point, NORMAL_EPSILON);
            float toward_camera = -dot(normal, ray_dir);
            if (toward_camera > 0.) {
                float alpha = toward_camera * coc_kernel(coc, map_dist);
                
                vec3 surface_color = color_at(point, ray_dir, normal, coc, mat);
                
                // "Alpha-under"ing surface_color/alpha beneath col
                float added_coverage = alpha * (1. - col.a);
                col = vec4(
                    (col.rgb * col.a + surface_color * added_coverage) / (col.a + added_coverage),
                    mix(col.a, 1., alpha)
                );
            }
        }
        
        iters++;
        ray_len += max(map_dist - .5 * coc, .3 * coc) * STEP_SCALE/* * mix(1., 0.9, rand(fragCoord.xy))*/;
    }
    
    if (col == vec4(0., 1., 0., 0.)) {
        normal = map_normal(point, NORMAL_EPSILON);
        col = vec4(color_at(point, ray_dir, normal, coc, mat), 1.);
    }
    
    col = vec4(sqrt(col.rgb), 1.);
    
	fragColor = col;
}