// --- transpiled interactive state buffer & point cloud rasterizer ---

bool is_key_down (int key)        {
        return (texelFetch(iKeyboard, ivec2(key, 0), 0).x)>(0.50F);
}
 

bool is_key_pressed (int key)        {
        return (texelFetch(iKeyboard, ivec2(key, 1), 0).x)>(0.50F);
}
 

void mainImage (out vec4 fragColor, in vec2 fragCoord)        {
            ivec2 ipx; 
    vec4 state; 
        (ipx)=(ivec2(fragCoord));
    (state)=(vec4(5.0F, 1.50F, 0.F, 0.70F)); 
    if ( (iFrame)>(0) ) {
                                (state)=(texelFetch(iChannel0, ivec2(0, 0), 0));  
} 
    if ( (ipx)==(ivec2(0, 0)) ) {
                        if ( is_key_pressed(9) ) {
                                    if ( is_key_down(16) ) {
                                                (state.z)=(mod((state.z)-(1.0F), 3.0F)); 
} else {
                                                (state.z)=(mod((state.z)+(1.0F), 3.0F)); 
}  
} 
                bool left; 
        bool right; 
                (left)=(is_key_down(37));
        (right)=(is_key_down(39)); 
        if ( left ) {
                                    if ( (state.z)==(0.F) ) {
                                                (state.x)=(max((state.x)-(0.10F), 1.0F)); 
}else if ( (state.z)==(1.0F) ) {
                                                (state.y)=(max((state.y)-(5.00e-2F), 0.F)); 
}else if ( (state.z)==(2.0F) ) {
                                                (state.w)=(max((state.w)-(2.00e-2F), 0.F)); 
}  
} 
        if ( right ) {
                                    if ( (state.z)==(0.F) ) {
                                                (state.x)=(min((state.x)+(0.10F), 15.F)); 
}else if ( (state.z)==(1.0F) ) {
                                                (state.y)=(min((state.y)+(5.00e-2F), 5.0F)); 
}else if ( (state.z)==(2.0F) ) {
                                                (state.w)=(min((state.w)+(2.00e-2F), 1.0F)); 
}  
}  
        if ( (iMouse.z)>(0.F) ) {
                                                vec2 m; 
            vec2 res; 
                        (m)=(iMouse.xy);
            (res)=(iResolution.xy); 
                        float mx; 
            float my; 
                        (mx)=((m.x)/(res.x));
            (my)=((m.y)/(res.y)); 
            if ( ((mx)>=(5.00e-2F))&&((mx)<=(0.40F)) ) {
                                                                float val; 
                                (val)=(((mx)-(5.00e-2F))/(0.350F)); 
                if ( ((my)>=(0.10F))&&((my)<=(0.150F)) ) {
                                                            (state.z)=(0.F);
                    (state.x)=((1.0F)+((val)*(14.F))); 
}else if ( ((my)>=(0.180F))&&((my)<=(0.230F)) ) {
                                                            (state.z)=(1.0F);
                    (state.y)=((0.F)+((val)*(5.0F))); 
}else if ( ((my)>=(0.260F))&&((my)<=(0.310F)) ) {
                                                            (state.z)=(2.0F);
                    (state.w)=((0.F)+((val)*(1.0F))); 
}   
}    
} 
                (fragColor)=(state);  
} 
    if ( !((ipx)==(ivec2(0, 0))) ) {
                                float point_size; 
                (point_size)=(state.x); 
                float min_depth = 1.00e+5F; 
        vec3 hitColor = vec3(0.F); 
        for ( int i = 0;(i)<(1000);(i)++ ) {
                                    float theta; 
            float phi; 
            float R; 
            float r; 
            vec3 p; 
            float cy; 
            float sy; 
            mat2 rotY; 
            mat2 rotX; 
            vec2 proj; 
            vec2 p_pixel; 
            float dist; 
            float size; 
            float depth; 
                        (theta)=((float(i))*(6.2830e-3F));
            (phi)=((float(i))*(3.00e-2F));
            (R)=(1.60F);
            (r)=(0.60F);
            (p)=(vec3(((R)+((r)*(cos((phi)*(3.0F)))))*(cos((phi)*(2.0F))), (r)*(sin((phi)*(3.0F))), ((R)+((r)*(cos((phi)*(3.0F)))))*(sin((phi)*(2.0F))))); 
                        (cy)=(cos((iTime)*(0.30F)));
            (sy)=(sin((iTime)*(0.30F)));
            (rotY)=(mat2(cy,  -(sy), sy, cy));
            (p.xz)=((rotY)*(p.xz));
            (cy)=(cos((iTime)*(0.150F)));
            (sy)=(sin((iTime)*(0.150F)));
            (rotX)=(mat2(cy,  -(sy), sy, cy));
            (p.yz)=((rotX)*(p.yz)); 
            (p.z)+=(5.0F);
                        (proj)=((p.xy)/(p.z));
            (p_pixel)=((((proj)*(0.50F))+(0.50F))*(iResolution.xy));
            (dist)=(length((fragCoord)-(p_pixel)));
            (size)=((point_size)/(p.z)); 
            if ( (dist)<(size) ) {
                                                                (depth)=(p.z); 
                if ( (depth)<(min_depth) ) {
                                                                                (min_depth)=(depth);
                    (hitColor)=((vec3(0.50F, 0.50F, 0.50F))+((0.50F)*(vec3(cos(phi), sin((phi)*(2.0F)), cos((phi)*(3.0F))))));  
}  
}  
} 
        if ( (min_depth)<(1.00e+4F) ) {
                                    (fragColor)=(vec4(hitColor, min_depth)); 
} else {
                                    (fragColor)=(vec4(0.10F, 0.150F, 0.20F, 1.00e+5F)); 
}    
}  
}
 