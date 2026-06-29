// --- transpiled main renderer: reconstruction, normals, shading, screen-space shadows, and EDL ---

vec3 reconstructP (ivec2 pixel, float depth)        {
            vec2 uv; 
    vec2 proj; 
        (uv)=(((vec2(pixel))-((0.50F)*(iResolution.xy)))/(iResolution.y));
    (proj)=((uv)*(2.0F)); 
    return vec3((proj)*(depth), depth); 
}
 

void mainImage (out vec4 fragColor, in vec2 fragCoord)        {
            vec4 state; 
        (state)=(texelFetch(iChannel0, ivec2(0, 0), 0)); 
        float point_size; 
    float edl_strength; 
    float focused_widget; 
    float shadow_strength; 
        (point_size)=(state.x);
    (edl_strength)=(state.y);
    (focused_widget)=(state.z);
    (shadow_strength)=(state.w); 
        vec4 centerData; 
    vec3 baseColor; 
    float depth; 
        (centerData)=(texelFetch(iChannel0, ivec2(fragCoord), 0));
    (baseColor)=(centerData.rgb);
    (depth)=(centerData.w); 
        vec3 col; 
    if ( (depth)>(1.00e+4F) ) {
                        vec2 uv = ((fragCoord)-((0.50F)*(iResolution.xy)))/(iResolution.y); 
                (col)=((vec3(0.10F, 0.150F, 0.20F))-((8.00e-2F)*(length(uv))));  
} else {
                        vec3 P; 
        float depth_R; 
        float depth_U; 
        vec3 P_R; 
        vec3 P_U; 
        vec3 dPdx; 
        vec3 dPdy; 
        vec3 normal; 
        vec3 normal_cross; 
        vec3 light_pos; 
        vec3 L; 
        float dif; 
        float shadow_factor; 
        vec3 V; 
        vec3 R_ref; 
        float spec; 
                (P)=(reconstructP(ivec2(fragCoord), depth)); 
                (depth_R)=(texelFetch(iChannel0, (ivec2(fragCoord))+(ivec2(1, 0)), 0).w);
        (depth_U)=(texelFetch(iChannel0, (ivec2(fragCoord))+(ivec2(0, 1)), 0).w); 
        if ( (depth_R)>(1.00e+4F) ) {
                                    (depth_R)=(depth); 
} 
        if ( (depth_U)>(1.00e+4F) ) {
                                    (depth_U)=(depth); 
} 
                (P_R)=(reconstructP((ivec2(fragCoord))+(ivec2(1, 0)), depth_R));
        (P_U)=(reconstructP((ivec2(fragCoord))+(ivec2(0, 1)), depth_U));
        (dPdx)=((P_R)-(P));
        (dPdy)=((P_U)-(P));
        (normal_cross)=(cross(dPdx, dPdy)); 
        if ( (length(normal_cross))<(1.00e-4F) ) {
                                    (normal)=(vec3(0.F, 0.F, -1.0F)); 
} else {
                                    (normal)=(normalize(normal_cross)); 
} 
                (light_pos)=(vec3((2.50F)*(cos((iTime)*(0.50F))), 2.50F, ((2.50F)*(sin((iTime)*(0.50F))))+(4.50F)));
        (L)=(normalize((light_pos)-(P)));
        (V)=(normalize( -(P))); 
        if ( (dot(normal, V))<(0.F) ) {
                                                (normal)=( -(normal));  
} 
                (dif)=(clamp(dot(normal, L), 0.F, 1.0F));
        (R_ref)=(reflect( -(L), normal));
        (spec)=((pow(max(dot(R_ref, V), 0.F), 16.F))*(0.30F)); 
                vec3 ray_dir = normalize((light_pos)-(P)); 
        float light_dist = length((light_pos)-(P)); 
        float t_max = min(light_dist, 5.0F); 
        int steps = 24; 
        float thickness = 0.350F; 
                (shadow_factor)=(1.0F); 
        for ( int step_idx = 1;(step_idx)<=(steps);(step_idx)++ ) {
                                    float tVal; 
            vec3 P_curr; 
            vec2 proj_curr; 
            vec2 uv_curr; 
            ivec2 pixel_curr; 
            float map_depth; 
                        (tVal)=(((float(step_idx))/(float(steps)))*(t_max));
            (P_curr)=((P)+((ray_dir)*(tVal)));
            (proj_curr)=((P_curr.xy)/(P_curr.z));
            (uv_curr)=((proj_curr)*(0.50F));
            (pixel_curr)=(ivec2(((uv_curr)*(iResolution.y))+((0.50F)*(iResolution.xy)))); 
            if ( ((pixel_curr.x)<(0))||((pixel_curr.x)>=(iResolution.x))||((pixel_curr.y)<(0))||((pixel_curr.y)>=(iResolution.y)) ) {
                                                break; 
} 
                        (map_depth)=(texelFetch(iChannel0, pixel_curr, 0).w); 
            if ( ((map_depth)<(1.00e+3F))&&((P_curr.z)>((map_depth)+(8.00e-2F)))&&((P_curr.z)<((map_depth)+(thickness))) ) {
                                                                (shadow_factor)=((1.0F)-(shadow_strength)); 
                break; 
}  
}  
                (col)=(((baseColor)*(((dif)*(shadow_factor))+(0.150F)))+((vec3(spec))*(shadow_factor)));  
} 
        float sum = 0.F; 
    float edl_radius = 2.0F; 
    vec2 offsets[4] = vec2[](vec2(0.0f, 1.0f), vec2(0.0f, -1.0f), vec2(1.0f, 0.0f), vec2(-1.0f, 0.0f)); 
    for ( int idx = 0;(idx)<(4);(idx)++ ) {
                        float neighborDepth; 
                (neighborDepth)=(texelFetch(iChannel0, clamp(ivec2((ivec2(fragCoord))+(ivec2(((offsets)[(idx)])*(edl_radius)))), ivec2(0), (ivec2(iResolution.xy))-(1)), 0).w); 
        if ( (neighborDepth)>(1.00e+4F) ) {
                                                (neighborDepth)=(depth);  
} 
                (sum)=((sum)+(max(0.F, (depth)-(neighborDepth))));  
} 
        (col)=((col)*(exp(( -(sum))*(1.50e+2F)*(edl_strength))));  
        (col)=(pow(col, vec3(0.45450F))); 
        vec2 scr_uv; 
    vec3 bar_color; 
    vec3 handle_color; 
    vec3 focus_color; 
        (scr_uv)=((fragCoord)/(iResolution.xy));
    (bar_color)=(vec3(0.40F));
    (handle_color)=(vec3(0.80F));
    (focus_color)=(vec3(0.20F, 0.90F, 0.20F)); 
        float VAL_0; 
    float Y_CENTER_0; 
    bool IS_FOCUSED_0; 
        (VAL_0)=(((point_size)-(1.0F))/(14.F));
    (Y_CENTER_0)=(0.1250F);
    (IS_FOCUSED_0)=((focused_widget)==(0.F)); 
        float HX_0; 
        (HX_0)=((5.00e-2F)+((VAL_0)*(0.350F))); 
    if ( ((scr_uv.x)>=(5.00e-2F))&&((scr_uv.x)<=(0.40F))&&((abs((scr_uv.y)-(Y_CENTER_0)))<(6.00e-3F)) ) {
                                (col)=(mix(col, (IS_FOCUSED_0) ? (focus_color) : (bar_color), 0.80F));  
} 
    if ( (length((scr_uv)-(vec2(HX_0, Y_CENTER_0))))<(1.20e-2F) ) {
                                (col)=(mix(col, (IS_FOCUSED_0) ? (focus_color) : (handle_color), 1.0F));  
}   
        float VAL_1; 
    float Y_CENTER_1; 
    bool IS_FOCUSED_1; 
        (VAL_1)=(((edl_strength)-(0.F))/(5.0F));
    (Y_CENTER_1)=(0.2050F);
    (IS_FOCUSED_1)=((focused_widget)==(1.0F)); 
        float HX_1; 
        (HX_1)=((5.00e-2F)+((VAL_1)*(0.350F))); 
    if ( ((scr_uv.x)>=(5.00e-2F))&&((scr_uv.x)<=(0.40F))&&((abs((scr_uv.y)-(Y_CENTER_1)))<(6.00e-3F)) ) {
                                (col)=(mix(col, (IS_FOCUSED_1) ? (focus_color) : (bar_color), 0.80F));  
} 
    if ( (length((scr_uv)-(vec2(HX_1, Y_CENTER_1))))<(1.20e-2F) ) {
                                (col)=(mix(col, (IS_FOCUSED_1) ? (focus_color) : (handle_color), 1.0F));  
}   
        float VAL_2; 
    float Y_CENTER_2; 
    bool IS_FOCUSED_2; 
        (VAL_2)=(((shadow_strength)-(0.F))/(1.0F));
    (Y_CENTER_2)=(0.2850F);
    (IS_FOCUSED_2)=((focused_widget)==(2.0F)); 
        float HX_2; 
        (HX_2)=((5.00e-2F)+((VAL_2)*(0.350F))); 
    if ( ((scr_uv.x)>=(5.00e-2F))&&((scr_uv.x)<=(0.40F))&&((abs((scr_uv.y)-(Y_CENTER_2)))<(6.00e-3F)) ) {
                                (col)=(mix(col, (IS_FOCUSED_2) ? (focus_color) : (bar_color), 0.80F));  
} 
    if ( (length((scr_uv)-(vec2(HX_2, Y_CENTER_2))))<(1.20e-2F) ) {
                                (col)=(mix(col, (IS_FOCUSED_2) ? (focus_color) : (handle_color), 1.0F));  
}    
        (fragColor)=(vec4(col, 1.0F));     
}
 