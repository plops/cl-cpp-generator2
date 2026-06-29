// --- transpiled raymarching shader with smin and shadows ---

float smin (float a, float b, float k)        {
            float h; 
        (h)=(clamp((0.50F)+((0.50F)*(((b)-(a))/(k))), 0.F, 1.0F)); 
    return (mix(b, a, h))-((k)*(h)*((1.0F)-(h))); 
}
 

float sdSphere (vec3 p, float s)        {
        return (length(p))-(s);
}
 

float sdBox (vec3 p, vec3 b)        {
            vec3 q; 
        (q)=((abs(p))-(b)); 
    return (length(max(q, 0.F)))+(min(max(q.x, max(q.y, q.z)), 0.F)); 
}
 

float map (vec3 p, float smax_blend)        {
            float plane; 
    float c; 
    float s; 
    mat3 rot; 
    vec3 pRot; 
    float box; 
    float sphere; 
    float blendedObject; 
        (plane)=((p.y)+(1.0F));
    (c)=(cos(iTime));
    (s)=(sin(iTime));
    (rot)=(mat3(c, 0.F, s, 0.F, 1.0F, 0.F,  -(s), 0.F, c));
    (pRot)=((rot)*(p));
    (box)=(sdBox(pRot, vec3(0.60F)));
    (sphere)=(sdSphere((pRot)-(vec3(0.F, 0.20F, 0.F)), 0.750F));
    (blendedObject)=(smin(box, sphere, smax_blend)); 
    return min(plane, blendedObject); 
}
 

vec3 getNormal (vec3 p, float smax_blend)        {
            vec2 e; 
    float d; 
    vec3 n; 
        (e)=(vec2(1.00e-3F, 0.F));
    (d)=(map(p, smax_blend));
    (n)=((d)-(vec3(map((p)-(e.xyy), smax_blend), map((p)-(e.yxy), smax_blend), map((p)-(e.yyx), smax_blend)))); 
    return normalize(n); 
}
 

float getShadow (vec3 ro, vec3 rd, float mint, float maxt, float k, float smax_blend)        {
            float res; 
    float tVal; 
        (res)=(1.0F);
    (tVal)=(mint); 
    for ( int i = 0;(i)<(32);(i)++ ) {
                        float h; 
                (h)=(map((ro)+((tVal)*(rd)), smax_blend)); 
        if ( (h)<(1.00e-3F) ) {
                                    return 0.F; 
} 
                (res)=(min(res, ((k)*(h))/(tVal))); 
        (tVal)+=(clamp(h, 1.00e-2F, 0.20F));
        if ( (tVal)>(maxt) ) {
                                    break; 
}  
} 
    return clamp(res, 0.F, 1.0F); 
}
 

void mainImage (out vec4 fragColor, in vec2 fragCoord)        {
            vec4 state {texelFetch(iChannel0, ivec2(0, 0), 0)}; 
        float smax_blend {state.x}; 
    float shadow_k {state.y}; 
    float focused_widget {state.z}; 
    float maxDist {state.w}; 
        vec2 uv; 
    vec3 ro; 
    vec3 rd; 
    float tVal; 
    bool hit; 
    vec3 p; 
    vec3 n; 
    vec3 lightPos; 
    vec3 l; 
    float dif; 
    float shadow; 
    vec3 objectColor; 
    vec3 col; 
        (uv)=(((fragCoord)-((0.50F)*(iResolution.xy)))/(iResolution.y));
    (ro)=(vec3(0.F, 1.0F, -3.0F));
    (rd)=(normalize(vec3(uv, 1.0F)));
    (tVal)=(0.F);
    (hit)=(false); 
    for ( int i = 0;i < 80;i++ ) {
                        float d; 
                (d)=(map((ro)+((tVal)*(rd)), smax_blend)); 
        if ( (d)<(1.00e-3F) ) {
                                                (hit)=(true); 
            break; 
} 
        (tVal)+=(d);
        if ( (tVal)>(maxDist) ) {
                                    break; 
}  
} 
        (col)=(vec3(0.10F, 0.150F, 0.20F)); 
    if ( hit ) {
                                (p)=((ro)+((tVal)*(rd)));
        (n)=(getNormal(p, smax_blend));
        (lightPos)=(vec3(2.0F, 4.0F, -1.0F));
        (l)=(normalize((lightPos)-(p)));
        (dif)=(clamp(dot(n, l), 0.F, 1.0F));
        (shadow)=(getShadow((p)+((n)*(1.00e-2F)), l, 1.00e-2F, 5.0F, shadow_k, smax_blend)); 
        if ( (p.y)>(-0.990F) ) {
                                    (objectColor)=(vec3(0.90F, 0.40F, 0.10F)); 
} else {
                                    (objectColor)=(vec3(0.50F)); 
} 
                (col)=((objectColor)*(((dif)*(shadow))+(0.10F)));
        (col)=(pow(col, vec3(0.45450F)));  
} 
        vec2 scr_uv {(fragCoord)/(iResolution.xy)}; 
    vec3 bar_color {vec3(0.40F)}; 
    vec3 handle_color {vec3(0.80F)}; 
    vec3 focus_color {vec3(0.20F, 0.90F, 0.20F)}; 
        float val {(smax_blend)/(2.0F)}; 
    float y_center {0.1250F}; 
    bool is_focused {(focused_widget)==(0.F)}; 
        float hx {(5.00e-2F)+((val)*(0.350F))}; 
    if ( ((scr_uv.x)>=(5.00e-2F))&&((scr_uv.x)<=(0.40F))&&((abs((scr_uv.y)-(y_center)))<(6.00e-3F)) ) {
                                (col)=(mix(col, (is_focused) ? (focus_color) : (bar_color), 0.80F));  
} 
    if ( (length((scr_uv)-(vec2(hx, y_center))))<(1.20e-2F) ) {
                                (col)=(mix(col, (is_focused) ? (focus_color) : (handle_color), 1.0F));  
}   
        float val {((shadow_k)-(1.0F))/(99.F)}; 
    float y_center {0.2050F}; 
    bool is_focused {(focused_widget)==(1.0F)}; 
        float hx {(5.00e-2F)+((val)*(0.350F))}; 
    if ( ((scr_uv.x)>=(5.00e-2F))&&((scr_uv.x)<=(0.40F))&&((abs((scr_uv.y)-(y_center)))<(6.00e-3F)) ) {
                                (col)=(mix(col, (is_focused) ? (focus_color) : (bar_color), 0.80F));  
} 
    if ( (length((scr_uv)-(vec2(hx, y_center))))<(1.20e-2F) ) {
                                (col)=(mix(col, (is_focused) ? (focus_color) : (handle_color), 1.0F));  
}   
        float val {((maxDist)-(2.0F))/(48.F)}; 
    float y_center {0.2850F}; 
    bool is_focused {(focused_widget)==(2.0F)}; 
        float hx {(5.00e-2F)+((val)*(0.350F))}; 
    if ( ((scr_uv.x)>=(5.00e-2F))&&((scr_uv.x)<=(0.40F))&&((abs((scr_uv.y)-(y_center)))<(6.00e-3F)) ) {
                                (col)=(mix(col, (is_focused) ? (focus_color) : (bar_color), 0.80F));  
} 
    if ( (length((scr_uv)-(vec2(hx, y_center))))<(1.20e-2F) ) {
                                (col)=(mix(col, (is_focused) ? (focus_color) : (handle_color), 1.0F));  
}    
        (fragColor)=(vec4(col, 1.0F));    
}
 