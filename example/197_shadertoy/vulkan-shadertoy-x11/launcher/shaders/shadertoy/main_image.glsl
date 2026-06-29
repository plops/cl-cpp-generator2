// --- transpiled raymarching shader with smin and shadows ---
struct Dual { float v; vec3 d; }; 
struct Dual3 { vec3 v; mat3 d; }; 

Dual addDual (Dual a, Dual b)        {
            Dual r; 
        (r.v)=((a.v)+(b.v));
    (r.d)=((a.d)+(b.d)); 
    return r; 
}
 

Dual subDual (Dual a, Dual b)        {
            Dual r; 
        (r.v)=((a.v)-(b.v));
    (r.d)=((a.d)-(b.d)); 
    return r; 
}
 

Dual3 subDual3 (Dual3 a, vec3 b)        {
            Dual3 r; 
        (r.v)=((a.v)-(b));
    (r.d)=(a.d); 
    return r; 
}
 

Dual3 mulMat3Dual3 (mat3 m, Dual3 p)        {
            Dual3 r; 
        (r.v)=((m)*(p.v));
    (r.d)=((m)*(p.d)); 
    return r; 
}
 

Dual getX (Dual3 q)        {
        return Dual(q.v.x, (q.d)[(0)]);
}
 

Dual getY (Dual3 q)        {
        return Dual(q.v.y, (q.d)[(1)]);
}
 

Dual getZ (Dual3 q)        {
        return Dual(q.v.z, (q.d)[(2)]);
}
 

Dual maxDualDual (Dual a, Dual b)        {
        if ( (a.v)>(b.v) ) {
                return a;
} else {
                return b;
} 
}
 

Dual maxDualFloat (Dual a, float b)        {
        if ( (a.v)>(b) ) {
                return a;
} else {
                return Dual(b, vec3(0.F));
} 
}
 

Dual minDualDual (Dual a, Dual b)        {
        if ( (a.v)<(b.v) ) {
                return a;
} else {
                return b;
} 
}
 

Dual minDualFloat (Dual a, float b)        {
        if ( (a.v)<(b) ) {
                return a;
} else {
                return Dual(b, vec3(0.F));
} 
}
 

Dual3 absDual3 (Dual3 p)        {
            Dual3 r; 
        (r.v)=(abs(p.v));
    (r.d)=(mat3(((p.d)[(0)])*(((p.v.x)>=(0.F)) ? (1.0F) : (-1.0F)), ((p.d)[(1)])*(((p.v.y)>=(0.F)) ? (1.0F) : (-1.0F)), ((p.d)[(2)])*(((p.v.z)>=(0.F)) ? (1.0F) : (-1.0F)))); 
    return r; 
}
 

Dual3 maxDual3Float (Dual3 a, float b)        {
            Dual3 r; 
        (r.v)=(max(a.v, vec3(b)));
    (r.d)=(mat3(((a.v.x)>(b)) ? ((a.d)[(0)]) : (vec3(0.F)), ((a.v.y)>(b)) ? ((a.d)[(1)]) : (vec3(0.F)), ((a.v.z)>(b)) ? ((a.d)[(2)]) : (vec3(0.F)))); 
    return r; 
}
 

Dual lengthDual3 (Dual3 a)        {
            Dual r; 
    float lenVal; 
        (lenVal)=(length(a.v));
    (r.v)=(lenVal); 
    if ( (lenVal)>(0.F) ) {
                        (r.d)=((normalize(a.v))*(a.d)); 
} else {
                        (r.d)=(vec3(0.F)); 
} 
    return r; 
}
 

float sdSphere (vec3 p, float s)        {
        return (length(p))-(s);
}
 

Dual sdSphereDual (Dual3 p, float s)        {
            Dual len; 
        (len)=(lengthDual3(p)); 
    return Dual((len.v)-(s), len.d); 
}
 

float sdBox (vec3 p, vec3 b)        {
            vec3 q; 
        (q)=((abs(p))-(b)); 
    return (length(max(q, 0.F)))+(min(max(q.x, max(q.y, q.z)), 0.F)); 
}
 

Dual sdBoxDual (Dual3 p, vec3 b)        {
            Dual3 q; 
    Dual len; 
    Dual mx; 
    Dual mn; 
        (q)=(subDual3(absDual3(p), b));
    (len)=(lengthDual3(maxDual3Float(q, 0.F)));
    (mx)=(maxDualDual(getX(q), maxDualDual(getY(q), getZ(q))));
    (mn)=(minDualFloat(mx, 0.F)); 
    return Dual((len.v)+(mn.v), (len.d)+(mn.d)); 
}
 

float smin (float a, float b, float k)        {
            float h; 
        (h)=(clamp((0.50F)+((0.50F)*(((b)-(a))/(k))), 0.F, 1.0F)); 
    return (mix(b, a, h))-((k)*(h)*((1.0F)-(h))); 
}
 

Dual sminDual (Dual a, Dual b, float k)        {
            float h; 
    Dual r; 
        (h)=(clamp((0.50F)+((0.50F)*(((b.v)-(a.v))/(k))), 0.F, 1.0F));
    (r.v)=((mix(b.v, a.v, h))-((k)*(h)*((1.0F)-(h))));
    (r.d)=(mix(b.d, a.d, h)); 
    return r; 
}
 

Dual mapDual (vec3 p_val, float smax_blend)        {
            Dual3 p; 
    Dual plane; 
    float c; 
    float s; 
    mat3 rot; 
    Dual3 pRot; 
    Dual box; 
    Dual sphere; 
    Dual blendedObject; 
        (p.v)=(p_val);
    (p.d)=(mat3(1.0F));
    (plane)=(Dual((p.v.y)+(1.0F), (p.d)[(1)]));
    (c)=(cos(iTime));
    (s)=(sin(iTime));
    (rot)=(mat3(c, 0.F, s, 0.F, 1.0F, 0.F,  -(s), 0.F, c));
    (pRot)=(mulMat3Dual3(rot, p));
    (box)=(sdBoxDual(pRot, vec3(0.60F)));
    (sphere)=(sdSphereDual(subDual3(pRot, vec3(0.F, 0.20F, 0.F)), 0.750F));
    (blendedObject)=(sminDual(box, sphere, smax_blend)); 
    return minDualDual(plane, blendedObject); 
}
 

float map (vec3 p, float smax_blend)        {
        return mapDual(p, smax_blend).v;
}
 

vec3 getNormal (vec3 p, float smax_blend)        {
        return normalize(mapDual(p, smax_blend).d);
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
            vec4 state; 
        (state)=(texelFetch(iChannel0, ivec2(0, 0), 0)); 
        float smax_blend; 
    float shadow_k; 
    float focused_widget; 
    float maxDist; 
        (smax_blend)=(state.x);
    (shadow_k)=(state.y);
    (focused_widget)=(state.z);
    (maxDist)=(state.w); 
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
        (VAL_0)=(((smax_blend)-(0.F))/(2.0F));
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
        (VAL_1)=(((shadow_k)-(1.0F))/(99.F));
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
        (VAL_2)=(((maxDist)-(2.0F))/(48.F));
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
 