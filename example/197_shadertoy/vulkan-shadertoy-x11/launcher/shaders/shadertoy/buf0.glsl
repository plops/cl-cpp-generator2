// --- transpiled interactive state buffer ---

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
    (state)=(vec4(0.50F, 1.0F, 0.F, 0.40F)); 
    if ( (ipx)==(ivec2(0, 0)) ) {
                        if ( (iFrame)>(0) ) {
                                                (state)=(texelFetch(iChannel0, ivec2(0, 0), 0));  
} 
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
                                                (state.x)=(max((state.x)-(5.00e-2F), 0.F)); 
}else if ( (state.z)==(1.0F) ) {
                                                (state.y)=(max((state.y)-(0.10F), 0.F)); 
}else if ( (state.z)==(2.0F) ) {
                                                (state.w)=(max((state.w)-(5.00e-2F), 0.10F)); 
}  
} 
        if ( right ) {
                                    if ( (state.z)==(0.F) ) {
                                                (state.x)=(min((state.x)+(5.00e-2F), 2.0F)); 
}else if ( (state.z)==(1.0F) ) {
                                                (state.y)=(min((state.y)+(0.10F), 4.0F)); 
}else if ( (state.z)==(2.0F) ) {
                                                (state.w)=(min((state.w)+(5.00e-2F), 1.50F)); 
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
                    (state.x)=((0.F)+((val)*(2.0F))); 
}else if ( ((my)>=(0.180F))&&((my)<=(0.230F)) ) {
                                                            (state.z)=(1.0F);
                    (state.y)=((0.F)+((val)*(4.0F))); 
}else if ( ((my)>=(0.260F))&&((my)<=(0.310F)) ) {
                                                            (state.z)=(2.0F);
                    (state.w)=((0.10F)+((val)*(1.40F))); 
}   
}    
} 
                (fragColor)=(state);  
} 
    if ( !((ipx)==(ivec2(0, 0))) ) {
                                (fragColor)=(vec4(0.F));  
}  
}
 