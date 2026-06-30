erzeuge ein markdown dokument mit all den informationen von Inigo Quilez ::
articles :: useful little functions

Intro When writing shader or during any procedural creation process (texturing,
modeling, shading, animation...) you often find yourself modifying signals in
different ways so they behave the way you need. It is common to use smoothstep()
to threshold some values, or pow() to shape a signal, or clamp() to clip it,
mod() to make it repeat, a mix() to blend between two signals, exp() for
attenuation, etc etc. All these functions are often conveniently available by
default in most languages. However there are some operations that are also
relatively used that don't come by default in any language. The following is a
list of some of the functions (of families of functions rather) that I find
myself using over and over again. You have the Graphtoy links in each so you can
explore them interactively:

Identities

Almost Identity graphtoy.com example

Imagine you want to clip a signal x to some epsilon value e. Using max(x,e)
creates a discontinuity. So instead, let m be a blending width, and smoothly
blend the epsilon e into the signal approaches as it approaches zero:

float almostIdentity( float x, float m, float e ) { if( x>m ) return x; float a
= 2.0e - m; float b = 2.0m - 3.0e; float t = x/m; return (at+b)tt + e; }

Smooth Identity 2/ Smooth abs() graphtoy.com example

Another piece-wise function to get a smooth identiy or smooth absolute value, is
this one, which happens to be the one we use for the traditional smooth minimum
of SDFs (the quadratic polynomial):

float smoothAbs( float x, float n ) { float x2 = xx; float n2 = nn; return x2<n2
? (x2+n2)/(2.0*n) : abs(x); }

Smooth Identity / Smooth abs 2() Youtube explanation graphtoy.com example

A different way to achieve a near identity is through the square root of a
biased square. I saw this technique first in a shader by user "omeometo" in
Shadertoy. This approach can be a bit slower than the cubic above, depending on
the hardware, but I find myself using it a lot these days, specially for "smooth
mirroring" shapes, since it behaves almost like the absolute value of x. While
it has zero derivative, it has a non-zero second derivative, so keep an eye in
case it causes problems in your application.

float smoothAbs( float x, float n ) { return sqrt(xx+nn); }

Smoothstep Integral Youtube explanation graphtoy.com example

If you use smoothstep for a velocity signal (say, you want to smoothly
accelerate a stationary object into constant velocity motion), you need to
integrate smoothstep() over time in order to get the actual position of value of
the animation. The function below is exactly that, the position of an object
that accelerates with smoothstep. Note its derivative is never larger than 1, so
no decelerations happen.

float integralSmoothstep( float x, float T ) { if( x>T ) return x - T/2.0;
return xxx*(1.0-x*0.5/T)/T/T; }

Impulses

Exponential Impulse graphtoy.com example

Impulses are great for triggering behaviours or making envelopes for music or
animation. Basically, for anything that grows fast and then decays slowly. The
following is an exponential impulse function. Use k to control the stretching of
the function. Its maximum, which is 1, happens at exactly x = 1/k.

float expImpulse( float x, float k ) { float h = kx; return hexp(1.0-h); }

Polynomial Impulse graphtoy.com example

Another impulse function that doesn't use exponentials can be designed by using
polynomials. Use k to control falloff of the function. For example, a quadratic
can be used, which peaks at x = sqrt(1/k).

float quaImpulse( float k, float x ) { return 2.0sqrt(k)x/(1.0+kxx); }

You can easily generalize it to other powers to get different falloff shapes,
where n is the degree of the polynomial:

float polyImpulse( float k, float n, float x ) { return (n/(n-1.0))*
pow((n-1.0)k,1.0/n) x/(1.0+k*pow(x,n)); }

These generalized impulses peak at x = [k(n-1)]-1/n.

Sustained Impulse

Similar to the previous, but it allows for control on the width of attack
(through the parameter "k") and the release (parameter "f") independently. Also,
the impulse releases at a value of 1 instead of 0.

float sustainedImpulse( float x, float f, float k ) { float s = max(x-f,0.0);
return min( xx/(ff), 1.0+(2.0/f)sexp(-k*s)); }

Sinc Impulse graphtoy.com example

A phase shifted sinc curve can be useful if it starts at zero and ends at zero,
for some bouncing behaviors (suggested by Hubert-Jan). Give k different integer
values to tweak the amount of bounces. The functions max value is 1.0, but it
can take negative values, which can make it unusable in some applications.

float sinc( float x, float k ) { float a = PI*(k*x-1.0); return sin(a)/a; }

Falloff graphtoy.com example

A quadratic falloff, like those in physically based point lights, but reaching
zero at a given distance "m" rather than just asymptotically reaching it at
infinity. Great for range controlled shadows, etc.

float trunc_fallof( float x, float m ) { if( x>m ) return 0.0; x /= m; return
(x-2.0)*x+1.0; }

Unitary remappings

These functions below remap the [0,1] interval into the [0,1] interval. They can
be used to adjust image contrasts, shape terrain slopes, modulate movements,
sculpt forms, etc. One such a common function is the smoothstep(), which I don't
include here for it is ubiquitous.

Almost Unit Identity graphtoy.com example

This is a near-identiy function that maps the unit interval into itself. It is
the cousin of smoothstep(), in that it maps 0 to 0, 1 to 1, and has a 0
derivative at the origin, just like smoothstep. However, instead of having a 0
derivative at 1, it has a derivative of 1 at that point. It's equivalent to the
Almost Identiy above with n=0 and m=1. Since it's a cubic just like smoothstep()
it is very fast to evaluate:

float almostUnitIdentity( float x ) { return xx(2.0-x); }

Gain graphtoy.com example

Remapping the unit interval into the unit interval by expanding the sides and
compressing the center, and keeping 1/2 mapped to 1/2, that can be done with the
gain() function. This was a common function in RSL tutorials (the Renderman
Shading Language). k=1 is the identity curve, k<1 produces the classic gain()
shape, and k>1 produces "s" shaped curves. The curves are symmetric (and
inverse) for k=a and k=1/a.

float gain( float x, float k ) { float a = 0.5pow(2.0((x<0.5)?x:1.0-x), k);
return (x<0.5)?a:1.0-a; }

Parabola graphtoy.com example

A nice choice to remap the 0..1 interval into 0..1, such that the corners are
mapped to 0 and the center to 1. You can then rise the parabola to a power k to
control its shape.

float parabola( float x, float k ) { return pow( 4.0x(1.0-x), k ); }

Power Curve

This is a generalization of the Parabola() above. It also maps the 0..1 interval
into 0..1 by keeping the corners mapped to 0. But in this generalization you can
control the shape one either side of the curve, which comes handy when creating
leaves, eyes, and many other interesting shapes.

float pcurve( float x, float a, float b ) { float k =
pow(a+b,a+b)/(pow(a,a)pow(b,b)); return kpow(x,a)*pow(1.0-x,b); }

Note that k is chosen such that the curve reaches exactly 1 at its maximum for
illustration purposes, but in many applications the curve needs to be scaled
anyways so the slow computation of k can be simply avoided.

Tonemap

This function maps the 0 to 0 and 1 to 1, while rising the middle tones upwards,
similar to a power function with exponent smaller than 1. However, when the
numerator is not k+1 but something else (usually larger), this function is often
used as a color tonemapping transfer function (similar to the Reinhard
tonemapper), hence the name. Usually k>0, but you can make it between 0 and -1
to bend the curves inwards like a possitive power curve.

float tone( float x, float k ) { return (k+1.0)x/(1.0+kx); }

Bumps

Cubic Pulse graphtoy.com example

Chances are you found yourself doing smoothstep(c-w,c,x)-smoothstep(c,c+w,x)
very often as a way to select a region centered at c that goes from c-w to c+w.
I know I do, and that's why I made this cubicPulse() below. You can also use it
as a replacement for a gaussian with local support.

float cubicPulse( float x, float c, float w ) { x = abs(x - c); if( x>w )
return 0.0; x /= w; return 1.0 - xx(3.0-2.0*x); }

Rational Bump graphtoy.com example

This function can also be a replacement for a gaussian sometimes if you can do
with infinite support, ie, this function never reaches exactly zero no matter
how far you go in the real axis:

float rationalBump( float x, float k ) { return 1.0/(1.0+kxx); }

Double Bump graphtoy.com example

This is a double bump with controllable attenuation. The way it's listed below,
the derivative at 0 is always 1, which makes it perfect to create smooth RELU
functions if you attach a y=x segment when x is positive. If on the other hand
yo don't care about the inclination at the origin and instead you want to
normalize it so it always peaks with a value of 1.0, then you can multiply the
function above by k/pow((k-1),(1-1/k)).

float doubleBump( float x, float k ) { return x/(1.0+pow(x,k)) ); }

Sigmoids / Steps

Exponential Step graphtoy.com example

A natural attenuation is an exponential of a linearly decaying quantity. A
gaussian, is an exponential of a quadratically decaying quantity: purple curve,
exp(-x2). You can generalize and keep increasing powers, and get a sharper and
sharper s-shaped curves. For really high values of n you can approximate a
perfect step().

float expStep( float x, float k ) { return exp2(-pow(x,k)); } inigo quilez -
learning computer graphics since 1994

Inigo Quilez :: articles :: distance functions

Intro Here you will find a list of Signed Distance Field implementations for
some basic 3D shapes, the formulas to smoothly combine them together needed for
building more complex shapes, and some other useful functions, which I used to
create all the beautiful images you can find in the Raymarching Distance Fields
article to see what kind of things you can do with SDFs. Please note that this
list is not exhaustive, you should read the many Articles I have under the "SDF
& Raymarching" that describe other SDFs shapes and techniques not listed here
such as fractal constructions, SDF noise, more rich shape and material blending,
lighting, acceleration structures for SDFs and many others. If you are
interested in the derivations of these formulas below, you can have a peek in
the SDF section in my video tutorials. Lastly, in the code snippets below
dot2(v) is just short for dot(v,v).

Primitives (https://www.shadertoy.com/playlist/43cXRl)

The first block of primitives are all exact and true SDFs. That is, for each
point in space they do measure, precisely, the distance to the closest surface
in that shape (with a traditional, Euclidean, L2-norm). Note you'll find many
resources in Shadertoy and elsewhere with incorrect SDF implementations that
might or might not work to some degree depending on the use case. The ones that
follow here, however, are all correct:

Sphere

float sdSphere( vec3 p, float r ) { return length(p) - r; }

Box https://www.youtube.com/watch?v=62-pRVZuS5c

float sdBox( vec3 p, vec3 b ) { vec3 q = abs(p) - b; return length(max(q,0.0)) +
min(max(q.x,max(q.y,q.z)),0.0); }

Round Box

float sdRoundBox( vec3 p, vec3 b, float r ) { vec3 q = abs(p) - b + r; return
length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0) - r; }

Box Frame https://www.shadertoy.com/view/3ljcRh

float sdBoxFrame( vec3 p, vec3 b, float e ) { p = abs(p )-b; vec3 q =
abs(p+e)-e; return min(min(
length(max(vec3(p.x,q.y,q.z),0.0))+min(max(p.x,max(q.y,q.z)),0.0),
length(max(vec3(q.x,p.y,q.z),0.0))+min(max(q.x,max(p.y,q.z)),0.0)),
length(max(vec3(q.x,q.y,p.z),0.0))+min(max(q.x,max(q.y,p.z)),0.0)); }

Torus

float sdTorus( vec3 p, vec2 t ) { vec2 q = vec2(length(p.xz)-t.x,p.y); return
length(q)-t.y; }

Capped Torus https://www.shadertoy.com/view/tl23RK

float sdCappedTorus( vec3 p, vec2 sc, float ra, float rb) { p.x = abs(p.x);
float k = (sc.yp.x>sc.xp.y) ? dot(p.xy,sc) : length(p.xy); return sqrt( dot(p,p)
+ rara - 2.0ra*k ) - rb; }

Link https://www.shadertoy.com/view/wlXSD7

float sdLink( vec3 p, float le, float r1, float r2 ) { vec3 q = vec3( p.x,
max(abs(p.y)-le,0.0), p.z ); return length(vec2(length(q.xy)-r1,q.z)) - r2; }

Infinite Cylinder

float sdCylinder( vec3 p, vec3 c ) { return length(p.xz-c.xy)-c.z; }

Cone

float sdCone( vec3 p, vec2 c, float h ) { // c is the sin/cos of the angle, h is
height // Alternatively pass q instead of (c,h), // which is the point at the
base in 2D vec2 q = h*vec2(c.x/c.y,-1.0);

vec2 w = vec2( length(p.xz), p.y ); vec2 a = w - qclamp(
dot(w,q)/dot(q,q), 0.0, 1.0 ); vec2 b = w - qvec2( clamp( w.x/q.x, 0.0, 1.0
), 1.0 ); float k = sign( q.y ); float d = min(dot( a, a ),dot(b, b)); float s =
max( k*(w.xq.y-w.yq.x),k*(w.y-q.y) ); return sqrt(d)*sign(s); }

Infinite Cone

float sdCone( vec3 p, vec2 c ) { // c is the sin/cos of the angle vec2 q = vec2(
length(p.xz), -p.y ); float d = length(q-cmax(dot(q,c), 0.0)); return d *
((q.xc.y-q.y*c.x<0.0)?-1.0:1.0); }

Plane

float sdPlane( vec3 p, vec3 n, float h ) { // n must be normalized return
dot(p,n) + h; }

Hexagonal Prism

float sdHexPrism( vec3 p, vec2 h ) { const vec3 k =
vec3(-0.8660254, 0.5, 0.57735); p = abs(p); p.xy -= 2.0min(dot(k.xy,
p.xy), 0.0)k.xy; vec2 d = vec2( length(p.xy-vec2(clamp(p.x,-k.zh.x,k.zh.x),
h.x))*sign(p.y-h.x), p.z-h.y ); return min(max(d.x,d.y),0.0) +
length(max(d,0.0)); }

Capsule / Line

float sdCapsule( vec3 p, vec3 a, vec3 b, float r ) { vec3 pa = p - a, ba = b -
a; float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 ); return length( pa - ba*h
) - r; }

Capsule / Line

float sdVerticalCapsule( vec3 p, float h, float r ) { p.y -= clamp( p.y, 0.0, h
); return length( p ) - r; }

Vertical Capped Cylinder https://www.shadertoy.com/view/wdXGDr

float sdCappedCylinder( vec3 p, float r, float h ) { vec2 d =
abs(vec2(length(p.xz),p.y)) - vec2(r,h); return min(max(d.x,d.y),0.0) +
length(max(d,0.0)); }

Arbitrary Capped Cylinder https://www.shadertoy.com/view/wdXGDr

float sdCappedCylinder( vec3 p, vec3 a, vec3 b, float r ) { vec3 ba = b - a;
vec3 pa = p - a; float baba = dot(ba,ba); float paba = dot(pa,ba); float x =
length(pababa-bapaba) - rbaba; float y = abs(paba-baba0.5)-baba0.5; float x2 =
xx; float y2 = yybaba; float d =
(max(x,y)<0.0)?-min(x2,y2):(((x>0.0)?x2:0.0)+((y>0.0)?y2:0.0)); return
sign(d)*sqrt(abs(d))/baba; }

Rounded Cylinder

float sdRoundedCylinder( vec3 p, float ra, float rb, float h ) { vec2 d = vec2(
length(p.xz)-ra+rb, abs(p.y) - h + rb ); return min(max(d.x,d.y),0.0) +
length(max(d,0.0)) - rb; }

Capped Cone

float sdCappedCone( vec3 p, float h, float r1, float r2 ) { vec2 q = vec2(
length(p.xz), p.y ); vec2 k1 = vec2(r2,h); vec2 k2 = vec2(r2-r1,2.0h); vec2 ca =
vec2(q.x-min(q.x,(q.y<0.0)?r1:r2), abs(q.y)-h); vec2 cb = q - k1 + k2clamp(
dot(k1-q,k2)/dot2(k2), 0.0, 1.0 ); float s = (cb.x<0.0 && ca.y<0.0) ? -1.0
: 1.0; return s*sqrt( min(dot2(ca),dot2(cb)) ); }

Capped Cone https://www.shadertoy.com/view/tsSXzK

float sdCappedCone( vec3 p, vec3 a, vec3 b, float ra, float rb ) { float rba =
rb-ra; float baba = dot(b-a,b-a); float papa = dot(p-a,p-a); float paba =
dot(p-a,b-a)/baba; float x = sqrt( papa - pabapabababa ); float cax =
max(0.0,x-((paba<0.5)?ra:rb)); float cay = abs(paba-0.5)-0.5; float k = rbarba +
baba; float f = clamp( (rba(x-ra)+pabababa)/k, 0.0, 1.0 ); float cbx = x-ra -
frba; float cby = paba - f; float s = (cbx<0.0 && cay<0.0) ? -1.0 : 1.0; return
ssqrt( min(caxcax + caycaybaba, cbxcbx + cbycby*baba) ); }

Solid Angle https://www.shadertoy.com/view/wtjSDW

float sdSolidAngle( vec3 p, vec2 c, float ra ) { // c is the sin/cos of the
angle vec2 q = vec2( length(p.xz), p.y ); float l = length(q) - ra; float m =
length(q - cclamp(dot(q,c),0.0,ra) ); return max(l,msign(c.yq.x-c.xq.y)); }

Cut Sphere https://www.shadertoy.com/view/stKSzc

float sdCutSphere( vec3 p, float r, float h ) { float w = sqrt(rr-hh);

vec2 q = vec2( length(p.xz), p.y ); float s = max( (h-r)q.xq.x+ww(h+r-2.0q.y),
hq.x-w*q.y ); return (s<0.0) ? length(q)-r : (q.x<w) ? h - q.y :
length(q-vec2(w,h)); }

Cut Hollow Sphere https://www.shadertoy.com/view/7tVXRt

float sdCutHollowSphere( vec3 p, float r, float h, float t ) { float w =
sqrt(rr-hh); vec2 q = vec2( length(p.xz), p.y ); return ((hq.x<wq.y) ?
length(q-vec2(w,h)) : abs(length(q)-r) ) - t; }

Death Star https://www.shadertoy.com/view/7lVXRt

float sdDeathStar( vec3 p2, float ra, float rb, float d ) { float a = (rara -
rbrb + dd)/(2.0d); float b = sqrt(max(rara-aa,0.0));

vec2 p = vec2( p2.x, length(p2.yz) ); if( p.xb-p.ya > d*max(b-p.y,0.0) ) return
length(p-vec2(a,b)); else return max( (length(p )-ra),
-(length(p-vec2(d,0.0))-rb)); }

Round cone

float sdRoundCone( vec3 p, float r1, float r2, float h ) { float b = (r1-r2)/h;
float a = sqrt(1.0-b*b);

vec2 q = vec2( length(p.xz), p.y ); float k = dot(q,vec2(-b,a)); if( k<0.0 )
return length(q) - r1; if( k>a*h ) return length(q-vec2(0.0,h)) - r2; return
dot(q, vec2(a,b) ) - r1; }

Round Cone https://www.shadertoy.com/view/tdXGWr

float sdRoundCone( vec3 p, vec3 a, vec3 b, float r1, float r2 ) { vec3 ba = b -
a; float l2 = dot(ba,ba); float rr = r1 - r2; float a2 = l2 - rr*rr; float il2
= 1.0/l2;

vec3 pa = p - a; float y = dot(pa,ba); float z = y - l2; float x2 = dot2( pal2 -
bay ); float y2 = yyl2; float z2 = zzl2;

// single square root! float k = sign(rr)rrrrx2; if( sign(z)a2z2>k ) return
sqrt(x2 + z2) il2 - r2; if( sign(y)a2y2<k ) return sqrt(x2 + y2) il2 - r1;
return (sqrt(x2a2il2)+yrr)*il2 - r1; }

Vesica Segment https://www.shadertoy.com/view/Ds2czG

float sdVesicaSegment( in vec3 p, in vec3 a, in vec3 b, in float w ) { vec3 c =
(a+b)0.5; float l = length(b-a); vec3 v = (b-a)/l; float y = dot(p-c,v); vec2 q
= vec2(length(p-c-yv),abs(y));

float r = 0.5*l;
float d = 0.5*(r*r-w*w)/w;
vec3  h = (r*q.x<d*(q.y-r)) ? vec3(0.0,r,0.0) : vec3(-d,0.0,d+w);

return length(q-h.xy) - h.z;

}

Rhombus https://www.shadertoy.com/view/tlVGDc

float sdRhombus( vec3 p, float la, float lb, float h, float ra ) { p = abs(p);
float f = clamp( (lap.x-lbp.z+lblb)/(lala+lb*lb), 0.0, 1.0 ); vec2 w = p.xz -
vec2(la,lb)*vec2(f,1.0-f); vec2 q = vec2( length(w)*sign(w.x)-ra, p.y-h); return
min(max(q.x,q.y),0.0) + length(max(q,0.0)); }

Octahedron https://www.shadertoy.com/view/wsSGDG

float sdOctahedron( vec3 p, float s ) { p = abs(p); float m = p.x+p.y+p.z-s;
vec3 q; if( 3.0p.x < m ) q = p.xyz; else if( 3.0p.y < m ) q = p.yzx; else
if( 3.0p.z < m ) q = p.zxy; else return m0.57735027;

float k = clamp(0.5*(q.z-q.y+s),0.0,s); return length(vec3(q.x,q.y-s+k,q.z-k));
}

Octahedron - bound (not exact)

float sdOctahedron( vec3 p, float s) { p = abs(p); return
(p.x+p.y+p.z-s)*0.57735027; }

Pyramid https://www.shadertoy.com/view/Ws3SDl

float sdPyramid( vec3 p, float h ) { float m2 = h*h + 0.25;

p.xz = abs(p.xz); p.xz = (p.z>p.x) ? p.zx : p.xz; p.xz -= 0.5;

vec3 q = vec3( p.z, hp.y - 0.5p.x, hp.x + 0.5p.y); float s = max(-q.x,0.0);
float t = clamp( (q.y-0.5p.z)/(m2+0.25), 0.0, 1.0 ); float a = m2(q.x+s)(q.x+s)
+ q.yq.y; float b = m2*(q.x+0.5t)(q.x+0.5t) + (q.y-m2t)(q.y-m2t);

float d2 = min(q.y,-q.xm2-q.y0.5) > 0.0 ? 0.0 : min(a,b); return sqrt(
(d2+q.z*q.z)/m2 ) * sign(max(q.z,-p.y)); }

Triangle https://www.shadertoy.com/view/4sXXRN

float udTriangle( vec3 p, vec3 a, vec3 b, vec3 c ) { vec3 ba = b - a; vec3 pa =
p - a; vec3 cb = c - b; vec3 pb = p - b; vec3 ac = a - c; vec3 pc = p - c; vec3
nor = cross( ba, ac );

return sqrt( (sign(dot(cross(ba,nor),pa)) + sign(dot(cross(cb,nor),pb)) +
sign(dot(cross(ac,nor),pc))<2.0) ? min( min(
dot2(baclamp(dot(ba,pa)/dot2(ba),0.0,1.0)-pa),
dot2(cbclamp(dot(cb,pb)/dot2(cb),0.0,1.0)-pb) ),
dot2(ac*clamp(dot(ac,pc)/dot2(ac),0.0,1.0)-pc) ) :
dot(nor,pa)*dot(nor,pa)/dot2(nor) ); }

Quad https://www.shadertoy.com/view/Md2BWW

float udQuad( vec3 p, vec3 a, vec3 b, vec3 c, vec3 d ) { vec3 ba = b - a; vec3
pa = p - a; vec3 cb = c - b; vec3 pb = p - b; vec3 dc = d - c; vec3 pc = p - c;
vec3 ad = a - d; vec3 pd = p - d; vec3 nor = cross( ba, ad );

return sqrt( (sign(dot(cross(ba,nor),pa)) + sign(dot(cross(cb,nor),pb)) +
sign(dot(cross(dc,nor),pc)) + sign(dot(cross(ad,nor),pd))<3.0) ? min( min( min(
dot2(baclamp(dot(ba,pa)/dot2(ba),0.0,1.0)-pa),
dot2(cbclamp(dot(cb,pb)/dot2(cb),0.0,1.0)-pb) ),
dot2(dcclamp(dot(dc,pc)/dot2(dc),0.0,1.0)-pc) ),
dot2(adclamp(dot(ad,pd)/dot2(ad),0.0,1.0)-pd) ) :
dot(nor,pa)*dot(nor,pa)/dot2(nor) ); }

The following block of functions are NOT correct and true SDFs. Instead they do
approximate SDFs in some sense or another, so sometimes they can be used with
certain rendering/collision algorithms. In particular, they are all lower bounds
and all produce zero exactly at the surface of the shape, so they can be used
with basic raymarching.

Ellipsoid - lower bound https://www.shadertoy.com/view/tdS3DG

float sdEllipsoid( vec3 p, vec3 r ) { float k0 = length(p/r); float k1 =
length(p/(rr)); return k0(k0-1.0)/k1; }

Triangular Prism - lower bound

float sdTriPrism( vec3 p, vec2 h ) { vec3 q = abs(p); return
max(q.z-h.y,max(q.x0.866025+p.y0.5,-p.y)-h.x*0.5); }

Creating more 3D SDFs, from 2D SDFs The list above is definitely not exhaustive,
many other shapes are easy to describe through a simple SDF. One simple way to
do so, is to take any 2D SDF and either revolve it or extrude it. This is really
easy to do, and has the advantage that if the the 2D SDF we start with is an
exact SDF, the resulting 3D volume is exact as well. This is interesting for a
couple of reasons: first, creating a shape through 3D boolean operations of
basic forms does not produce an exact SDF (we'll talk about this later in this
article), while doing it as a revolution or extrussion of a 2D shape does
produce the correct SDF. Secondly, doing 3D boolean operations generates
suboptimal code since it does not reuse common expressions across primitives. So
doing revolution or extrusion of a 2D shapes produces the correct SDF and is
also faster to compute.

You can find code to do extrussion and revolution below, and also here:
https://www.shadertoy.com/view/4lyfzw

float opRevolution( in vec3 p, in sdf2d primitive, in float o ) { vec2 q = vec2(
length(p.xz) - o, p.y ); return primitive(q) }

float opExtrusion( in vec3 p, in sdf2d primitive, in float h ) { float d =
primitive(p.xy) vec2 w = vec2( d, abs(p.z) - h ); return min(max(w.x,w.y),0.0) +
length(max(w,0.0)); }

Creating more 3D SDFs, from 3D SDFs It is also possible to create new types
of 3D primitives from other 3D primitives. Here are some examples:

Elongation - bound/exact

Elongating is a useful way to construct new shapes. It basically splits a
primitive in two (four or eight), moves the pieces apart and and connects them.
It is a perfect distance preserving operation, it does not introduce any
artifacts in the SDF. Some of the basic primitives above use this technique. For
example, the Capsule is an elongated Sphere along an axis really. You can find
code here: https://www.shadertoy.com/view/Ml3fWj

float opElongate( in sdf3d primitive, in vec3 p, in vec3 h ) { vec3 q = p -
clamp( p, -h, h ); return primitive( q ); }

float opElongate( in sdf3d primitive, in vec3 p, in vec3 h ) { vec3 q =
abs(p)-h; return primitive( max(q,0.0) ) + min(max(q.x,max(q.y,q.z)),0.0); }

The reason I provide implementations is the following. For 1D elongations, the
first function works perfectly and gives exact exterior and interior distances.
However, the first implementation produces a small core of zero distances inside
the volume for 2D and 3D elongations. Depending on your application that might
be a problem. One way to create exact interior distances all the way to the very
elongated core of the volume, is the following, which is in languages like GLSL
that don't have function pointers or lambdas need to be implemented a bit
differently (check the code linked about in Shadertoy to see one example).

Rounding/Inflating - exact

Rounding a shape is as simple as subtracting some distance (jumping to a
different isosurface). The rounded box above is an example, but you can apply it
to cones, hexagons or any other shape like the cone in the image below. If you
happen to be interested in preserving the overall volume of the shape, most of
the time it's pretty easy to shrink the source primitive by the same amount we
are rounding it by. You can find code here:
https://www.shadertoy.com/view/Mt3BDj

float opRound( in sdf3d primitive, in float rad ) { return primitive(p) - rad }

Onion - exact

For carving interiors or giving thickness to primitives, without performing
expensive boolean operations (see below) and without distorting the distance
field into a bound, one can use "onion-ing". You can use it multiple times to
create concentric layers in your SDF. You can find code here:
https://www.shadertoy.com/view/MlcBDj

float opOnion( in float sdf, in float thickness ) { return abs(sdf)-thickness; }

Change of Metric - bound

Most of these functions can be modified to use other norms than the Euclidean.
By replacing length(p), which computes (x2+y2+z2)1/2 by (xn+yn+zn)1/n one can
get variations of the basic primitives that have rounded edges rather than sharp
ones. However I do not recommend using this technique other than for toy demos,
since these are not true SDFs anymore but lower bounds, meaning that they
require more raymarching steps than usual. Since they only give a bound to the
real SDF, this kind of primitive alteration also doesn't play well with shadows
and occlusion algorithms that rely on true SDFs for measuring distance to
occludes. You can find the code here: https://www.shadertoy.com/view/ltcfDj

float length2( vec3 p ) { p=p*p; return sqrt( p.x+p.y+p.z); }

float length6( vec3 p ) { p=ppp; p=p*p; return pow(p.x+p.y+p.z,1.0/6.0); }

float length8( vec3 p ) { p=pp; p=pp; p=p*p; return pow(p.x+p.y+p.z,1.0/8.0); }

Primitive combinations Sometimes you cannot simply elongate, round or onion a
primitive, and you need to combine, carve or intersect basic primitives. Given
the SDFs a and b of two primitives, you can use the following operators to
combine together.

Union, Subtraction, Intersection - exact/bound, bound, bound

These are the most basic combinations of pairs of primitives you can do. They
correspond to the basic boolean operations. Please note that the Xor and the
Union of two SDFs produces a true SDF, but not the Subtraction or Intersection.
To make it more subtle, this is only true in the exterior of the SDF (where
distances are positive) and not in the interior. You can learn more in this
article about Xor, this and how to work around the incorrect interior distances
of the Union in the article about "Interior Distances". Also note that
opSubtraction() is not commutative and depending on the order of the operand it
will produce different results.

float opUnion( float a, float b ) { return min(a,b); } float opSubtraction(
float a, float b ) { return max(-a,b); } float opIntersection( float a, float b
) { return max(a,b); } float opXor( float a, float b ) { return
max(min(a,b),-max(a,b)); }

Smooth Union, Subtraction and Intersection - bound, bound, bound

Blending primitives is a really powerful tool - it allows to construct complex
and organic shapes without the geometrical seams that normal boolean operations
produce. There are many flavors of such operations, but the basic ones try to
replace the min() and max() functions used in the opUnion, opSubstraction and
opIntersection above with smooth versions. They all accept an extra parameter
called k that defines the size of the smooth transition between the two
primitives. It is given in actual distance units. You can find more details in
the smooth minimum article article in this same site. You can code here:
https://www.shadertoy.com/view/lt3BW2

float opSmoothUnion( float a, float b, float k ) { k = 4.0; float h =
max(k-abs(a-b),0.0); return min(a, b) - hh*0.25/k; }

float opSmoothSubtraction( float a, float b, float k ) { return
-opSmoothUnion(a,-b,k);

// k *= 4.0;
// float h = max(k-abs(-a-b),0.0);
// return max(-a, b) + h*h*0.25/k;

}

float opSmoothIntersection( float a, float b, float k ) { return
-opSmoothUnion(-a,-b,k);

// k *= 4.0;
// float h = max(k-abs(a-b),0.0);
// return max(a, b) + h*h*0.25/k;

}

Positioning Placing primitives in different locations and orientations in space
is a fundamental operation in designing SDFs. While rotations, uniform scaling
and translations are exact operations, non-uniform scaling distorts the
Euclidean spaces and can only be bound. Therefore I do not include it here.

Rotation/Translation - exact

Since rotations and translation don't compress nor dilate space, all we need to
do is simply to transform the point being sampled with the inverse of the
transformation used to place an object in the scene. This code below assumes
that transform encodes only a rotation and a translation (as a 3x4 matrix for
example, or as a quaternion and a vector), and that it does not contain any
scaling factors in it.

vec3 opTx( in vec3 p, in transform t, in sdf3d primitive ) { return primitive(
invert(t)*p ); }

Scale - exact

Scaling an obect is slightly more tricky since that compresses/dilates spaces,
so we have to take that into account on the resulting distance estimation.
Still, it's not difficult to perform, although it only works with uniform
scaling. Non uniform scaling is not possible (while still getting a correct
SDF):

float opScale( in vec3 p, in float s, in sdf3d primitive ) { return
primitive(p/s)*s; }

Symmetry and repetition Something beautiful with procedural SDFs is that
creating multiple copies of the same object can be done easily at no memory or
performance cost. By making the SDF function itself symmetric or periodic we get
automatic instancing in constant time, with just a few lines of code.

Symmetry - bound and exact

Symmetry is useful, since many things around us are symmetric, from humans,
animals, vehicles, instruments, furniture, ... Oftentimes, one can take
shortcuts and only model half or a quarter of the desired shape, and get it
duplicated automatically by using the absolute value of the domain coordinates
before evaluation. For example, in the image below, there's a single object
evaluation instead of two. You have to be aware however that the resulting SDF
might not be an exact SDF but a bound, if the object you are mirroring crosses
the mirroring plane.

float opSymX( in vec3 p, in sdf3d primitive ) { p.x = abs(p.x); return
primitive(p); }

float opSymXZ( in vec3 p, in sdf3d primitive ) { p.xz = abs(p.xz); return
primitive(p); }

Infinite and limited Repetition

Domain repetition is a very useful operator, since it allows you to create
infinitely many primitives with a single object evaluation:

float opRepetition( in vec3 p, in vec3 s, in sdf3d primitive ) { vec3 q = p -
s*round(p/s); return primitive( q ); }

In this code s is the spacing between the instances. This function above will
only work for symmetric shapes (with respect to the repeating tile boundaries),
generally raymarchers will nor render objects properly if this function is use
naively as is. To learn how to make it work for arbitrary SDFs without
artifacts, and to learn more about all sort of repetition patterns, like
circular, rectangular, please see the article on Domain Repetition.

Infinite domain repetition is similar to the above, but it contains the number
of instances, which is useful for human made things where we usually don't have
infinite of anything :) Again, if you want to use limited repetition with non
symmetric shapes, please read the article on Domain Repetition to learn how to
fix them.

vec3 opLimitedRepetition( in vec3 p, in float s, in vec3 l, in sdf3d primitive )
{ vec3 q = p - s*clamp(round(p/s),-l,l); return primitive( q ); }

The article linked above also explains how to use instance identifiers to
perform shape and color variations such that each shape in the repetition grid
looks different, among other things.

Deformations and distortions

Deformations and distortions allow to enhance the shape of primitives or even
fuse different primitives together. The operations usually distort the distance
field and make it non-Euclidean anymore, so one must be careful when raymarching
them, you will probably need to decrease your step size, if you are using a
raymarcher to sample this. In principle one can compute the factor by which the
step size needs to be reduced (inversely proportional to the compression of the
space, which is given by the Jacobian of the deformation function). But even
with dual numbers or automatic differentiation, it's usually just easier to find
the constant by hand for a given primitive.

I'd say that while it is tempting to use a distortion or displacement to achieve
a given shape, and I often use them myself of course, it is sometimes better to
get as close to the desired shape with actual exact Euclidean primitive
operations (elongation, rounding, onion-ing, union) or tight bounded functions
(intersection, subtraction) and then only apply as small of a distortion or
displacement as possible. That way the field stays as close as possible to an
actual distance field, and the raymarcher will be faster.

Displacement

The displacement example below is using sin(20*p.x)sin(20p.y)sin(20p.z) as
displacement pattern, but you can of course use anything you might imagine.

float opDisplace( in sdf3d primitive, in vec3 p ) { float d1 = primitive(p);
float d2 = displacement(p); return d1+d2; }

Twist

float opTwist( in sdf3d primitive, in vec3 p ) { const float k = 10.0; // or
some other amount float c = cos(kp.y); float s = sin(kp.y); mat2 m =
mat2(c,-s,s,c); vec3 q = vec3(m*p.xz,p.y); return primitive(q); }

Bend

float opCheapBend( in sdf3d primitive, in vec3 p ) { const float k = 10.0; // or
some other amount float c = cos(kp.x); float s = sin(kp.x); mat2 m =
mat2(c,-s,s,c); vec3 q = vec3(m*p.xy,p.z); return primitive(q); }

A reference implementation of most of these primitives and operators can be
found here: https://www.shadertoy.com/view/Xds3zN:

inigo quilez - learning computer graphics since 1994

Inigo Quilez :: articles :: 2D distance functions

Intro The 3D SDF functions article is pretty popular, so I decided to write a
similar one for 2D primitives, since most of the 3D primitives are grown as
extrusions or revolutions of these 2D shapes. So getting these right is
important. In particular, the functions bellow use the minimum number of square
roots and divisions possible, and also produce better/faster results than
constructing them from other primitives (when possible).

Note that all the primitives listed here come with a link to a realtime online
demo in Shadertoy. In fact, this public playlist contains all the Shadertoy
examples: https://www.shadertoy.com/playlist/MXdSRf, so don't miss that out.

Lastly, as with all in this website, all formulas and code are derived by myself
(unless otherwise stated), so there might be errors or better ways to do things.
Let me know if you think that's the case.

Primitives All primitives are centered at the origin; you will have to transform
the point to get arbitrarily rotated, translated and scaled objects (see below).
The dot2(v) function returns the dot product of a vector with itself (or the
square of its length).

Circle - exact https://www.shadertoy.com/view/3ltSW2

float sdCircle( vec2 p, float r ) { return length(p) - r; }

Rounded Box - exact https://www.shadertoy.com/view/4llXD7
https://www.youtube.com/watch?v=s5NGeUV2EyU

float sdRoundedBox( in vec2 p, in vec2 b, in vec4 r ) { r.xy = (p.x>0.0)?r.xy :
r.zw; r.x = (p.y>0.0)?r.x : r.y; vec2 q = abs(p)-b+r.x; return
min(max(q.x,q.y),0.0) + length(max(q,0.0)) - r.x; }

Chamfer Box - exact https://www.shadertoy.com/view/3fc3zs

float sdChamferBox( in vec2 p, in vec2 b, in float chamfer ) { p = abs(p)-b;

p = (p.y>p.x) ? p.yx : p.xy;
p.y += chamfer;

const float k = 1.0-sqrt(2.0);
if( p.y<0.0 && p.y+p.x*k<0.0 )
    return p.x;

if( p.x<p.y )
    return (p.x+p.y)*sqrt(0.5);    

return length(p);

}

Box - exact https://www.youtube.com/watch?v=62-pRVZuS5c

float sdBox( in vec2 p, in vec2 b ) { vec2 d = abs(p)-b; return
length(max(d,0.0)) + min(max(d.x,d.y),0.0); }

Oriented Box - exact https://www.shadertoy.com/view/stcfzn

float sdOrientedBox( in vec2 p, in vec2 a, in vec2 b, float th ) { float l =
length(b-a); vec2 d = (b-a)/l; vec2 q = (p-(a+b)*0.5); q =
mat2(d.x,-d.y,d.y,d.x)*q; q = abs(q)-vec2(l,th)*0.5; return length(max(q,0.0)) +
min(max(q.x,q.y),0.0);
}

Segment - exact https://www.shadertoy.com/view/3tdSDj
https://www.youtube.com/watch?v=PMltMdi1Wzg

float sdSegment( in vec2 p, in vec2 a, in vec2 b ) { vec2 pa = p-a, ba = b-a;
float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 ); return length( pa - ba*h );
}

Rhombus - exact https://www.shadertoy.com/view/XdXcRB

float sdRhombus( in vec2 p, in vec2 b ) { b.y = -b.y; p = abs(p); float h =
clamp( (dot(b,p)+b.yb.y)/dot(b,b), 0.0, 1.0 ); p -= bvec2(h,h-1.0); return
length(p)*sign(p.x); }

Isosceles Trapezoid - exact https://www.shadertoy.com/view/MlycD3

float sdTrapezoid( in vec2 p, in float r1, float r2, float he ) { vec2 k1 =
vec2(r2,he); vec2 k2 = vec2(r2-r1,2.0he); p.x = abs(p.x); vec2 ca =
vec2(p.x-min(p.x,(p.y<0.0)?r1:r2), abs(p.y)-he); vec2 cb = p - k1 + k2clamp(
dot(k1-p,k2)/dot2(k2), 0.0, 1.0 ); float s = (cb.x<0.0 && ca.y<0.0) ? -1.0
: 1.0; return s*sqrt( min(dot2(ca),dot2(cb)) ); }

Parallelogram - exact https://www.shadertoy.com/view/7dlGRf

float sdParallelogram( in vec2 p, float wi, float he, float sk ) { vec2 e =
vec2(sk,he); p = (p.y<0.0)?-p:p; vec2 w = p - e; w.x -= clamp(w.x,-wi,wi); vec2
d = vec2(dot(w,w), -w.y); float s = p.xe.y - p.ye.x; p = (s<0.0)?-p:p; vec2 v =
p - vec2(wi,0); v -= eclamp(dot(v,e)/dot(e,e),-1.0,1.0); d = min( d,
vec2(dot(v,v), wihe-abs(s))); return sqrt(d.x)*sign(-d.y); }

Equilateral Triangle - exact https://www.shadertoy.com/view/Xl2yDW

float sdEquilateralTriangle( in vec2 p, in float r ) { const float k =
sqrt(3.0); p.x = abs(p.x) - r; p.y = p.y + r/k; if( p.x+kp.y>0.0 ) p =
vec2(p.x-kp.y,-kp.x-p.y)/2.0; p.x -= clamp( p.x, -2.0r, 0.0 ); return
-length(p)*sign(p.y); }

Isosceles Triangle - exact https://www.shadertoy.com/view/MldcD7

float sdTriangleIsosceles( in vec2 p, in vec2 q ) { p.x = abs(p.x); vec2 a = p -
qclamp( dot(p,q)/dot(q,q), 0.0, 1.0 ); vec2 b = p - qvec2( clamp(
p.x/q.x, 0.0, 1.0 ), 1.0 ); float s = -sign( q.y ); vec2 d = min( vec2(
dot(a,a), s*(p.xq.y-p.yq.x) ), vec2( dot(b,b), s*(p.y-q.y) )); return
-sqrt(d.x)*sign(d.y); }

Triangle - exact https://www.shadertoy.com/view/XsXSz4

float sdTriangle( in vec2 p, in vec2 p0, in vec2 p1, in vec2 p2 ) { vec2 e0 =
p1-p0, e1 = p2-p1, e2 = p0-p2; vec2 v0 = p -p0, v1 = p -p1, v2 = p -p2; vec2 pq0
= v0 - e0clamp( dot(v0,e0)/dot(e0,e0), 0.0, 1.0 ); vec2 pq1 = v1 - e1clamp(
dot(v1,e1)/dot(e1,e1), 0.0, 1.0 ); vec2 pq2 = v2 - e2clamp(
dot(v2,e2)/dot(e2,e2), 0.0, 1.0 ); float s = sign( e0.xe2.y - e0.ye2.x ); vec2 d
= min(min(vec2(dot(pq0,pq0), s(v0.xe0.y-v0.ye0.x)), vec2(dot(pq1,pq1),
s*(v1.xe1.y-v1.ye1.x))), vec2(dot(pq2,pq2), s*(v2.xe2.y-v2.ye2.x))); return
-sqrt(d.x)*sign(d.y); }

Uneven Capsule - exact https://www.shadertoy.com/view/4lcBWn

float sdUnevenCapsule( vec2 p, float r1, float r2, float h ) { p.x = abs(p.x);
float b = (r1-r2)/h; float a = sqrt(1.0-bb); float k = dot(p,vec2(-b,a)); if( k
< 0.0 ) return length(p) - r1; if( k > ah ) return length(p-vec2(0.0,h)) - r2;
return dot(p, vec2(a,b) ) - r1; }

Regular Pentagon - exact https://www.shadertoy.com/view/llVyWW

float sdPentagon( in vec2 p, in float r ) { const vec3 k =
vec3(0.809016994,0.587785252,0.726542528); p.x = abs(p.x); p
-= 2.0min(dot(vec2(-k.x,k.y),p),0.0)vec2(-k.x,k.y); p -= 2.0min(dot(vec2(
k.x,k.y),p),0.0)vec2( k.x,k.y); p -= vec2(clamp(p.x,-rk.z,rk.z),r);
return length(p)*sign(p.y); }

Regular Hexagon - exact

float sdHexagon( in vec2 p, in float r ) { const vec3 k =
vec3(-0.866025404,0.5,0.577350269); p = abs(p); p
-= 2.0min(dot(k.xy,p),0.0)k.xy; p -= vec2(clamp(p.x, -k.zr, k.zr), r); return
length(p)*sign(p.y); }

Regular Octogon - exact https://www.shadertoy.com/view/llGfDG

float sdOctogon( in vec2 p, in float r ) { const vec3 k =
vec3(-0.9238795325, 0.3826834323, 0.4142135623 ); p = abs(p); p
-= 2.0min(dot(vec2( k.x,k.y),p),0.0)vec2( k.x,k.y); p
-= 2.0min(dot(vec2(-k.x,k.y),p),0.0)vec2(-k.x,k.y); p -= vec2(clamp(p.x, -k.zr,
k.zr), r); return length(p)*sign(p.y); }

Hexagram - exact https://www.shadertoy.com/view/tt23RR

float sdHexagram( in vec2 p, in float r ) { const vec4 k =
vec4(-0.5,0.8660254038,0.5773502692,1.7320508076); p = abs(p); p
-= 2.0min(dot(k.xy,p),0.0)k.xy; p -= 2.0min(dot(k.yx,p),0.0)k.yx; p -=
vec2(clamp(p.x,rk.z,rk.w),r); return length(p)*sign(p.y); }

Pentagram - exact https://www.shadertoy.com/view/t3X3z4

float sdPentagram(in vec2 p, in float r ) { const float k1x = 0.809016994; //
cos(π/ 5) = ¼(√5+1) const float k2x = 0.309016994; // sin(π/10) = ¼(√5-1) const
float k1y = 0.587785252; // sin(π/ 5) = ¼√(10-2√5) const float k2y
= 0.951056516; // cos(π/10) = ¼√(10+2√5) const float k1z = 0.726542528; //
tan(π/ 5) = √(5-2√5) const vec2 v1 = vec2( k1x,-k1y); const vec2 v2 =
vec2(-k1x,-k1y); const vec2 v3 = vec2( k2x,-k2y);

p.x = abs(p.x);
p -= 2.0*max(dot(v1,p),0.0)*v1;
p -= 2.0*max(dot(v2,p),0.0)*v2;
p.x = abs(p.x);
p.y -= r;
return length(p-v3*clamp(dot(p,v3),0.0,k1z*r))
       * sign(p.y*v3.x-p.x*v3.y);

}

Regular Star - exact https://www.shadertoy.com/view/3tSGDy

float sdStar( in vec2 p, in float r, in int n, in float m) { // next 4 lines can
be precomputed for a given shape float an = 3.141593/float(n); float en
= 3.141593/m; // m is between 2 and n vec2 acs = vec2(cos(an),sin(an)); vec2 ecs
= vec2(cos(en),sin(en)); // ecs=vec2(0,1) for regular polygon

float bn = mod(atan(p.x,p.y),2.0*an) - an;
p = length(p)*vec2(cos(bn),abs(sin(bn)));
p -= r*acs;
p += ecs*clamp( -dot(p,ecs), 0.0, r*acs.y/ecs.y);
return length(p)*sign(p.x);

}

Pie - exact https://www.shadertoy.com/view/3l23RK

float sdPie( in vec2 p, in vec2 c, in float r ) { p.x = abs(p.x); float l =
length(p) - r; float m = length(p-cclamp(dot(p,c),0.0,r)); // c=sin/cos of
aperture return max(l,msign(c.yp.x-c.xp.y)); }

Cut Disk - exact https://www.shadertoy.com/view/ftVXRc

float sdCutDisk( in vec2 p, in float r, in float h ) { float w = sqrt(rr-hh); //
constant for any given shape p.x = abs(p.x); float s = max(
(h-r)p.xp.x+ww(h+r-2.0p.y), hp.x-w*p.y ); return (s<0.0) ? length(p)-r : (p.x<w)
? h - p.y : length(p-vec2(w,h)); }

Arc - exact https://www.shadertoy.com/view/wl23RK

float sdArc( in vec2 p, in vec2 sc, in float ra, float rb ) { // sc is the
sin/cos of the arc's aperture p.x = abs(p.x); return ((sc.yp.x>sc.xp.y) ?
length(p-sc*ra) : abs(length(p)-ra)) - rb; }

Ring - exact https://www.shadertoy.com/view/DsccDH

float sdRing( in vec2 p, in vec2 n, in float r, float th ) { p.x = abs(p.x); p =
mat2x2(n.x,n.y,-n.y,n.x)p; return max( abs(length(p)-r)-th0.5,
length(vec2(p.x,max(0.0,abs(r-p.y)-th*0.5)))*sign(p.x) ); }

Horseshoe - exact https://www.shadertoy.com/view/WlSGW1

float sdHorseshoe( in vec2 p, in vec2 c, in float r, in vec2 w ) { p.x =
abs(p.x); float l = length(p); p = mat2(-c.x, c.y, c.y, c.x)p; p = vec2((p.y>0.0
|| p.x>0.0)?p.x:lsign(-c.x), (p.x>0.0)?p.y:l ); p = vec2(p.x,abs(p.y-r))-w;
return length(max(p,0.0)) + min(0.0,max(p.x,p.y)); }

Vesica - exact https://www.shadertoy.com/view/XtVfRW

float sdVesica(vec2 p, float w, float h) { vec3 d = 0.5*(ww-hh)/h; p = abs(p);
vec3 c = (wp.y<d(p.x-w)) ? vec3(0.0,w,0.0) : vec3(-d,0.0,d+h); return
length(p-c.yx) - c.z; }

Oriented Vesica - exact https://www.shadertoy.com/view/cs2yzG

float sdOrientedVesica( vec2 p, vec2 a, vec2 b, float w ) { float r
\= 0.5length(b-a); float d = 0.5(rr-ww)/w; vec2 v = (b-a)/r; vec2 c = (b+a)0.5;
vec2 q = 0.5abs(mat2(v.y,v.x,-v.x,v.y)(p-c)); vec3 h = (rq.x<d*(q.y-r)) ?
vec3(0.0,r,0.0) : vec3(-d,0.0,d+w); return length( q-h.xy) - h.z; }

Moon - exact https://www.shadertoy.com/view/WtdBRS

float sdMoon(vec2 p, float d, float ra, float rb ) { p.y = abs(p.y); float a =
(rara - rbrb + dd)/(2.0d); float b = sqrt(max(rara-aa,0.0)); if( d*(p.xb-p.ya) >
ddmax(b-p.y,0.0) ) return length(p-vec2(a,b)); return max( (length(p )-ra),
-(length(p-vec2(d,0))-rb)); }

Circle Cross - exact https://www.shadertoy.com/view/NslXDM

float sdRoundedCross( in vec2 p, in float h ) { float k = 0.5*(h+1.0/h); p =
abs(p); return ( p.x<1.0 && p.y<p.x*(k-h)+h ) ? k-sqrt(dot2(p-vec2(1,k))) :
sqrt(min(dot2(p-vec2(0,h)), dot2(p-vec2(1,0)))); }

Simple Egg - exact https://www.shadertoy.com/view/XtVfRW

float sdEgg( in vec2 p, in float he, in float ra, in float rb, in float bu ) {
// all this can be precomputed for any given shape float r = 0.5*(he +
ra+rb)/bu; float da = r - ra; float db = r - rb; float y = (dbdb - dada -
hehe)/(2.0he); float x = sqrt(dada - yy);

// only this needs to be run per pixel
p.x = abs(p.x);
float k = p.y*x - p.x*y;
if( k>0.0 && k<he*(p.x+x) )
    return length(p+vec2(x,y))-r;
return min( length(p)-ra,
            length(vec2(p.x,p.y-he))-rb );

}

Heart - exact https://www.shadertoy.com/view/3tyBzV

float sdHeart( in vec2 p ) { p.x = abs(p.x);

if( p.y+p.x>1.0 )
    return sqrt(dot2(p-vec2(0.25,0.75))) - sqrt(2.0)/4.0;
return sqrt(min(dot2(p-vec2(0.00,1.00)),
                dot2(p-0.5*max(p.x+p.y,0.0)))) * sign(p.x-p.y);

}

Cross - exact exterior, bound interior https://www.shadertoy.com/view/XtGfzw

float sdCross( in vec2 p, in vec2 b, float r ) { p = abs(p); p = (p.y>p.x) ?
p.yx : p.xy; vec2 q = p - b; float k = max(q.y,q.x); vec2 w = (k>0.0) ? q :
vec2(b.y-p.x,-k); return sign(k)*length(max(w,0.0)) + r; }

Rounded X - exact https://www.shadertoy.com/view/3dKSDc

float sdRoundedX( in vec2 p, in float w, in float r ) { p = abs(p); return
length(p-min(p.x+p.y,w)*0.5) - r; }

Polygon - exact https://www.shadertoy.com/view/wdBXRW

float sdPolygon( in vec2[N] v, in vec2 p ) { float d = dot(p-v[0],p-v[0]); float
s = 1.0; for( int i=0, j=N-1; i<N; j=i, i++ ) { vec2 e = v[j] - v[i]; vec2 w = p
- v[i]; vec2 b = w - eclamp( dot(w,e)/dot(e,e), 0.0, 1.0 ); d = min( d, dot(b,b)
); bvec3 c = bvec3(p.y>=v[i].y,p.y<v[j].y,e.xw.y>e.yw.x); if( all(c) ||
all(not(c)) ) s=-1.0;
} return s*sqrt(d); }

Ellipse - exact https://www.shadertoy.com/view/4sS3zz

float sdEllipse( in vec2 p, in vec2 ab ) { p = abs(p); if( p.x > p.y )
{p=p.yx;ab=ab.yx;} float l = ab.yab.y - ab.xab.x; float m = ab.xp.x/l; float m2
= mm; float n = ab.yp.y/l; float n2 = nn; float c = (m2+n2-1.0)/3.0; float c3 =
ccc; float q = c3 + m2n22.0; float d = c3 + m2n2; float g = m + mn2; float co;
if( d<0.0 ) { float h = acos(q/c3)/3.0; float s = cos(h); float t =
sin(h)sqrt(3.0); float rx = sqrt( -c(s + t + 2.0) + m2 ); float ry = sqrt( -c*(s
- t + 2.0) + m2 ); co = (ry+sign(l)rx+abs(g)/(rxry)- m)/2.0; } else { float h
= 2.0mnsqrt( d ); float s = sign(q+h)pow(abs(q+h), 1.0/3.0); float u =
sign(q-h)pow(abs(q-h), 1.0/3.0); float rx = -s - u - c4.0 + 2.0m2; float ry = (s
- u)sqrt(3.0); float rm = sqrt( rxrx + ryry ); co =
(ry/sqrt(rm-rx)+2.0g/rm-m)/2.0; } vec2 r = ab * vec2(co, sqrt(1.0-coco)); return
length(r-p) * sign(p.y-r.y); }

Parabola - exact https://www.shadertoy.com/view/ws3GD7

float sdParabola( in vec2 pos, in float k ) { pos.x = abs(pos.x); float ik
= 1.0/k; float p = ik*(pos.y - 0.5ik)/3.0; float q = 0.25ikikpos.x; float h = qq
- ppp; float x; if( h>0.0 ) { float r = pow(q+sqrt(h),1.0/3.0); x = r + p/r; }
else { float r = sqrt(p); x = 2.0rcos(acos(q/(pr))/3.0); } return
length(pos-vec2(x,kxx)) * sign(pos.x-x); }

Parabola Segment - exact https://www.shadertoy.com/view/3lSczz

float sdParabola( in vec2 pos, in float wi, in float he ) { pos.x = abs(pos.x);
float ik = wiwi/he; float p = ik(he-pos.y-0.5ik)/3.0; float q = pos.xikik/4.0;
float h = qq - ppp; float x; if( h>0.0 ) { float r = pow(q+sqrt(h),1.0/3.0); x =
r + p/r; } else { float r = sqrt(p); x = 2.0rcos(acos(q/(pr))/3.0); } x =
min(x,wi); return length(pos-vec2(x,he-xx/ik)) *
sign(ik*(pos.y-he)+pos.x*pos.x); }

Quadratic Bezier - exact https://www.shadertoy.com/view/MlKcDD

float sdBezier( in vec2 pos, in vec2 A, in vec2 B, in vec2 C ) {
vec2 a = B - A; vec2 b = A - 2.0B + C; vec2 c = a * 2.0; vec2 d = A - pos; float
kk = 1.0/dot(b,b); float kx = kk * dot(a,b); float ky = kk *
(2.0dot(a,a)+dot(d,b)) / 3.0; float kz = kk * dot(d,a);
float res = 0.0; float p = ky - kxkx; float p3 = ppp; float q =
kx(2.0kxkx-3.0ky) + kz; float h = qq + 4.0p3; if( h >= 0.0) { h = sqrt(h); vec2
x = (vec2(h,-h)-q)/2.0; vec2 uv = sign(x)pow(abs(x), vec2(1.0/3.0)); float t =
clamp( uv.x+uv.y-kx, 0.0, 1.0 ); res = dot2(d + (c + bt)t); } else { float z =
sqrt(-p); float v = acos( q/(pz2.0) ) / 3.0; float m = cos(v); float n =
sin(v)*1.732050808; vec3 t = clamp(vec3(m+m,-n-m,n-m)z-kx,0.0,1.0); res = min(
dot2(d+(c+bt.x)t.x), dot2(d+(c+bt.y)t.y) ); // the third root cannot be the
closest // res = min(res,dot2(d+(c+bt.z)*t.z)); } return sqrt( res ); }

Bobbly Cross - exact https://www.shadertoy.com/view/NssXWM

float sdBlobbyCross( in vec2 pos, float he ) { pos = abs(pos); pos =
vec2(abs(pos.x-pos.y),1.0-pos.x-pos.y)/sqrt(2.0);

float p = (he-pos.y-0.25/he)/(6.0*he);
float q = pos.x/(he*he*16.0);
float h = q*q - p*p*p;

float x;
if( h>0.0 ) { float r = sqrt(h); x = pow(q+r,1.0/3.0)-pow(abs(q-r),1.0/3.0)*sign(r-q); }
else        { float r = sqrt(p); x = 2.0*r*cos(acos(q/(p*r))/3.0); }
x = min(x,sqrt(2.0)/2.0);

vec2 z = vec2(x,he*(1.0-2.0*x*x)) - pos;
return length(z) * sign(z.y);

}

Tunnel - exact https://www.shadertoy.com/view/flSSDy

float sdTunnel( in vec2 p, in vec2 wh ) { p.x = abs(p.x); p.y = -p.y; vec2 q = p
- wh;

float d1 = dot2(vec2(max(q.x,0.0),q.y));
q.x = (p.y>0.0) ? q.x : length(p)-wh.x;
float d2 = dot2(vec2(q.x,max(q.y,0.0)));
float d = sqrt( min(d1,d2) );

return (max(q.x,q.y)<0.0) ? -d : d;

}

Stairs - exact https://www.shadertoy.com/view/7tKSWt

float sdStairs( in vec2 p, in vec2 wh, in float n ) { vec2 ba = wh*n; float d =
min(dot2(p-vec2(clamp(p.x,0.0,ba.x),0.0)),
dot2(p-vec2(ba.x,clamp(p.y,0.0,ba.y))) ); float s = sign(max(-p.y,p.x-ba.x) );

float dia = length(wh);
p = mat2(wh.x,-wh.y, wh.y,wh.x)*p/dia;
float id = clamp(round(p.x/dia),0.0,n-1.0);
p.x = p.x - id*dia;
p = mat2(wh.x, wh.y,-wh.y,wh.x)*p/dia;

float hh = wh.y/2.0;
p.y -= hh;
if( p.y>hh*sign(p.x) ) s=1.0;
p = (id<0.5 || p.x>0.0) ? p : -p;
d = min( d, dot2(p-vec2(0.0,clamp(p.y,-hh,hh))) );
d = min( d, dot2(p-vec2(clamp(p.x,0.0,wh.x),hh)) );

return sqrt(d)*s;

}

Quadratic Circle - exact https://www.shadertoy.com/view/Nd3cW8

float sdQuadraticCircle( in vec2 p ) { p = abs(p); if( p.y>p.x ) p=p.yx;

float a = p.x-p.y;
float b = p.x+p.y;
float c = (2.0*b-1.0)/3.0;
float h = a*a + c*c*c;
float t;
if( h>=0.0 )
{   
    h = sqrt(h);
    t = sign(h-a)*pow(abs(h-a),1.0/3.0) - pow(h+a,1.0/3.0);
}
else
{   
    float z = sqrt(-c);
    float v = acos(a/(c*z))/3.0;
    t = -z*(cos(v)+sin(v)*1.732050808);
}
t *= 0.5;
vec2 w = vec2(-t,t) + 0.75 - t*t - p;
return length(w) * sign( a*a*0.5+b-1.5 );

}

Hyperbola - exact https://www.shadertoy.com/view/DtjXDG

float sdHyberbola( in vec2 p, in float k, in float he ) // k in (0,inf) { p =
abs(p); p = vec2(p.x-p.y,p.x+p.y)/sqrt(2.0);

float x2 = p.x*p.x/16.0;
float y2 = p.y*p.y/16.0;
float r = k*(4.0*k - p.x*p.y)/12.0;
float q = (x2 - y2)*k*k;
float h = q*q + r*r*r;
float u;
if( h<0.0 )
{
    float m = sqrt(-r);
    u = m*cos( acos(q/(r*m))/3.0 );
}
else
{
    float m = pow(sqrt(h)-q,1.0/3.0);
    u = (m - r/m)/2.0;
}
float w = sqrt( u + x2 );
float b = k*p.y - x2*p.x*2.0;
float t = p.x/4.0 - w + sqrt( 2.0*x2 - u + b/w/4.0 );
t = max(t,sqrt(he*he*0.5+k)-he/sqrt(2.0));
float d = length( p-vec2(t,k/t) );
return p.x*p.y < k ? d : -d;

}

Cool S - exact https://www.shadertoy.com/view/clVXWc

float sdfCoolS( in vec2 p ) { float six = (p.y<0.0) ? -p.x : p.x; p.x =
abs(p.x); p.y = abs(p.y) - 0.2; float rex = p.x - min(round(p.x/0.4),0.4); float
aby = abs(p.y-0.2)-0.6;

float d = dot2(vec2(six,-p.y)-clamp(0.5*(six-p.y),0.0,0.2));
d = min(d,dot2(vec2(p.x,-aby)-clamp(0.5*(p.x-aby),0.0,0.4)));
d = min(d,dot2(vec2(rex,p.y  -clamp(p.y          ,0.0,0.4))));

float s = 2.0*p.x + aby + abs(aby+0.4) - 0.4;
return sqrt(d) * sign(s);

}

Circle Wave - exact https://www.shadertoy.com/view/stGyzt

float sdCircleWave( in vec2 p, in float tb, in float ra ) { tb
\= 3.14159275.0/6.0max(tb,0.0001); vec2 co = ravec2(sin(tb),cos(tb)); p.x =
abs(mod(p.x,co.x4.0)-co.x2.0); vec2 p1 = p; vec2 p2 =
vec2(abs(p.x-2.0co.x),-p.y+2.0co.y); float d1 = ((co.yp1.x>co.xp1.y) ?
length(p1-co) : abs(length(p1)-ra)); float d2 = ((co.yp2.x>co.x*p2.y) ?
length(p2-co) : abs(length(p2)-ra)); return min(d1, d2); }

Making shapes rounded

All the shapes above can be converted into rounded shapes by subtracting a
constant from their distance function. That, effectively moves the isosurface
(isoperimeter I guess) from the level zero to one of the outer rings, which
naturally are rounded, as it can be seen in the yellow areas in all the images
above. So, basically, for any shape defined by d(x,y) = sdf(x,y), one can make
it sounded by computing d(x,y) = sdf(x,y) - r. You can learn more about this in
this Youtube video.

float opRound( in vec2 p, in float r ) { return sdShape(p) - r; }

These are a few examples: rounded line, rounded triangle, rounded box and a
rounded pentagon:

Making shapes annular

Similarly, shapes can be made annular (like a ring or the layers of an onion),
but taking their absolute value and then subtracting a constant from their
field. So, for any shape defined by d(x,y) = sdf(x,y) compute d(x,y) =
|sdf(x,y)| - r:

float opOnion( in vec2 p, in float r ) { return abs(sdShape(p)) - r; }

These are a few examples: annular rounded line, an annular triangle, an annular
box and a annular pentagon:

Rigid deformations, domain repetition, range distortion, boolean operations,
smooth connections, etc

All of these work just as well as in 3D case, so I won't repeat the examples
already exist in this article.

inigo quilez - learning computer graphics since 1994

Inigo Quilez :: articles :: 2D distance and gradient functions - 2019

Intro We know that proper 2D and 3D SDFs have a gradient of length 1.0
everywhere, since SDFs measure distances, and the rate of change of distance
with respect to distance will always be the identity. However, knowing the exact
direction of the gradient is useful for computer graphics (for lighting,
aligning objects to surfaces, etc etc). Of course, one can use central
differences to compute gradients, but that has a high computational cost and can
get tricky to tweak. Automatic differentiation works well with SDFs of course,
but comes with a cost.

In this page you'll find a list of primitives and operators and their analytic
gradients that I developed around 2020, where lots of terms are reused between
the SDF computation and its gradient computation, making them very cheap. The
list is incomplete, so feel free to carry on all primitives and operations that
I haven't defined with automatic differentiation/duals. Also, this list just
contains 2D SDFs, but extending them to 3D through revolution or extrusion is
pretty easy.

Lastly, note that all the primitives listed here come with a link to a realtime
online demo in Shadertoy. In fact, this public playlist contains all the
Shadertoy examples: https://www.shadertoy.com/playlist/M3dSRf, so don't miss
that out.

Primitives All primitives are centered at the origin; you will have to transform
the point to get arbitrarily rotated, translated and scaled objects, and
transform the gradient back accordingly (remember not to use non uniform scales
since such transforms do not preserve distances). All SDFs below return the
actual distance in the .x component, and its partial derivatives with respect to
x and y in the .y and .z components. In other words,

.x = f(p) .y = ∂f(p)/∂x .z = ∂f(p)/∂y .yz = ∇f(p) with ‖∇f(p)‖ = 1

Circle https://www.shadertoy.com/view/WltSDj

vec3 sdgCircle( in vec2 p, in float r ) { float d = length(p); return vec3( d-r,
p/d ); }

Pie https://www.shadertoy.com/view/3tGXRc

// sc = sin/cos of aperture vec3 sdgPie( in vec2 p, in vec2 sc, in float r ) {
float s = sign(p.x); p.x = abs(p.x); float l = length(p); float n = l - r; vec2
q = p - scclamp(dot(p,sc),0.0,r); float m = length(q) * sign(sc.yp.x-sc.xp.y);
vec3 res = (n>m) ? vec3(n,p/l) : vec3(m,q/m); return vec3(res.x,sres.y,res.z); }

Arc https://www.shadertoy.com/view/WtGXRc

// sc = sin/cos of aperture vec3 sdgArc( in vec2 p, in vec2 sc, in float ra, in
float rb ) { vec2 q = p; float s = sign(p.x); p.x = abs(p.x); if( sc.yp.x >
sc.xp.y ) { vec2 w = p - rasc; float d = length(w); return vec3( d-rb,
vec2(sw.x,w.y)/d ); } else { float l = length(q); float w = l - ra; return vec3(
abs(w)-rb, sign(w)*q/l ); } }

Segment https://www.shadertoy.com/view/WtdSDj

vec3 sdgSegment( in vec2 p, in vec2 a, in vec2 b, in float r ) { vec2 ba = b-a,
pa = p-a; float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 ); vec2 q = pa-h*ba;
float d = length(q); return vec3(d-r,q/d); }

Vesica https://www.shadertoy.com/view/3lGXRc

vec3 sdgVesica( vec2 p, float r, float d ) { vec2 s = sign(p); p = abs(p); float
b = sqrt(rr-dd); if( (p.y-b)d>p.xb ) { vec2 q = vec2(p.x,p.y-b); float l =
length(q)sign(d); return vec3( l, sq/l ); } else { vec2 q = vec2(p.x+d,p.y);
float l = length(q); return vec3( l-r, s*q/l ); } }

Box https://www.shadertoy.com/view/wlcXD2

vec3 sdgBox( in vec2 p, in vec2 b ) { vec2 w = abs(p)-b; vec2 s =
vec2(p.x<0.0?-1:1,p.y<0.0?-1:1); float g = max(w.x,w.y); vec2 q = max(w,0.0);
float l = length(q); return vec3( (g>0.0)?l :g,
s*((g>0.0)?q/l:((w.x>w.y)?vec2(1,0):vec2(0,1)))); }

Cross https://www.shadertoy.com/view/WtdXWj

vec3 sdgCross( in vec2 p, in vec2 b ) { vec2 s = sign(p); p = abs(p); vec2 q =
((p.y>p.x)?p.yx:p.xy) - b; float h = max( q.x, q.y ); vec2 o = max(
(h<0.0)?vec2(b.y-b.x,0.0)-q:q, 0.0 ); float l = length(o); vec3 r = (h<0.0 &&
-q.x<l)?vec3(-q.x,1.0,0.0):vec3(l,o/l); return vec3( sign(h)r.x,
s((p.y>p.x)?r.zy:r.yz) ); }

Pentagon https://www.shadertoy.com/view/3lySRc

vec3 sdgPentagon( in vec2 p, in float r ) { const vec3 m =
vec3(0.80901699,0.58778525,0.72654253); const vec2 n =
vec2(m.xm.x-m.ym.y,2.0m.xm.y ); float s = sign(p.x); p.x = abs(p.x); float w1 =
p.xm.x + p.ym.y; float w2 = p.xn.x - p.yn.y; p -= 2.0max(w1,0.0)vec2(m.x,m.y); p
-= 2.0min(w2,0.0)vec2(m.x,-m.y); p -= vec2(clamp(p.x,-rm.z,rm.z),-r); float d =
length(p)*sign(-p.y); vec2 g = (w2<0.0)?mat2x2(-m.x,m.y,-m.y,-m.x)*p:
(w1>0.0)?mat2x2(-n.x,-n.y,-n.y,n.x)*p: p; g.x *= s; return vec3(d, g/d ); }

Hexagon https://www.shadertoy.com/view/WtySRc

vec3 sdgHexagon( in vec2 p, in float r ) { const vec3 k =
vec3(-0.866025404,0.5,0.577350269); vec2 s = sign(p); p = abs(p); float w =
dot(k.xy,p);
p -= 2.0min(w,0.0)k.xy; p -= vec2(clamp(p.x, -k.zr, k.zr), r); float d =
length(p)*sign(p.y); vec2 g = (w<0.0) ? mat2(-k.y,-k.x,-k.x,k.y)p : p; return
vec3( d, sg/d ); }

Isosceles Triangle https://www.shadertoy.com/view/3dyfDd

vec3 sdgTriangleIsosceles( in vec2 p, in vec2 q ) { float w = sign(p.x); p.x =
abs(p.x); vec2 a = p - qclamp( dot(p,q)/dot(q,q), 0.0, 1.0 ); vec2 b = p -
qvec2( clamp( p.x/q.x, 0.0, 1.0 ), 1.0 ); float k = sign(q.y); float l1 =
dot(a,a); float l2 = dot(b,b); float d = sqrt((l1<l2)?l1:l2); vec2 g = (l1<l2)?
a: b; float s = max( k*(p.xq.y-p.yq.x),k*(p.y-q.y) ); return
vec3(d,vec2(w*g.x,g.y)/d)*sign(s); }

Triangle https://www.shadertoy.com/view/tlVyWh

float cro( in vec2 a, in vec2 b ) { return a.xb.y - a.yb.x; } vec3 sdgTriangle(
in vec2 p, in vec2 v[3] ) { float gs = cro(v[0]-v[2],v[1]-v[0]); vec4 res;

{
vec2  e = v[1]-v[0], w = p-v[0];
vec2  q = w-e*clamp(dot(w,e)/dot(e,e),0.0,1.0);
float d = dot(q,q), s = gs*cro(w,e);
res = vec4(d,q,s);
} {
vec2  e = v[2]-v[1], w = p-v[1];
vec2  q = w-e*clamp(dot(w,e)/dot(e,e),0.0,1.0);
float d = dot(q,q), s = gs*cro(w,e);
res = vec4( (d<res.x) ? vec3(d,q) : res.xyz,
            (s>res.w) ?      s    : res.w );
} {
vec2  e = v[0]-v[2], w = p-v[2];
vec2  q = w-e*clamp(dot(w,e)/dot(e,e),0.0,1.0);
float d = dot(q,q), s = gs*cro(w,e);
res = vec4( (d<res.x) ? vec3(d,q) : res.xyz,
            (s>res.w) ?      s    : res.w );
}

float d = sqrt(res.x)*sign(res.w);
return vec3(d,res.yz/d);

}

Quad https://www.shadertoy.com/view/WtVcD1

float cro( in vec2 a, in vec2 b ) { return a.xb.y - a.yb.x; } vec3 sdgQuad( in
vec2 p, in vec2 v[4] ) { float gs = cro(v[0]-v[3],v[1]-v[0]); vec4 res;

{
vec2  e = v[1]-v[0], w = p-v[0];
vec2  q = w-e*clamp(dot(w,e)/dot(e,e),0.0,1.0);
float d = dot(q,q), s = gs*cro(w,e);
res = vec4(d,q,s);
} {
vec2  e = v[2]-v[1], w = p-v[1];
vec2  q = w-e*clamp(dot(w,e)/dot(e,e),0.0,1.0);
float d = dot(q,q), s = gs*cro(w,e);
res = vec4( (d<res.x) ? vec3(d,q) : res.xyz,
            (s>res.w) ?      s    : res.w );
} {
vec2  e = v[3]-v[2], w = p-v[2];
vec2  q = w-e*clamp(dot(w,e)/dot(e,e),0.0,1.0);
float d = dot(q,q), s = gs*cro(w,e);
res = vec4( (d<res.x) ? vec3(d,q) : res.xyz,
            (s>res.w) ?      s    : res.w );
} {
vec2  e = v[0]-v[3], w = p-v[3];
vec2  q = w-e*clamp(dot(w,e)/dot(e,e),0.0,1.0);
float d = dot(q,q), s = gs*cro(w,e);
res = vec4( (d<res.x) ? vec3(d,q) : res.xyz,
            (s>res.w) ?      s    : res.w );
}    

float d = sqrt(res.x)*sign(res.w);
return vec3(d,res.yz/d);

}

Ellipse https://www.shadertoy.com/view/3lcfR8

vec3 sdgEllipse( vec2 p, in vec2 ab ) { vec2 sp = sign(p); p = abs( p );

bool s = dot(p/ab,p/ab)>1.0;
float w = atan(p.y*ab.x, p.x*ab.y);
if(!s) w=(ab.x*(p.x-ab.x)<ab.y*(p.y-ab.y))? 1.570796327 : 0.0;

for( int i=0; i<4; i++ )
{
    vec2 cs = vec2(cos(w),sin(w));
    vec2 u = ab*vec2( cs.x,cs.y);
    vec2 v = ab*vec2(-cs.y,cs.x);
    w = w + dot(p-u,v)/(dot(p-u,u)+dot(v,v));
}
vec2  q = ab*vec2(cos(w),sin(w));

float d = length(p-q);
return vec3( d, sp*(p-q)/d ) * (s?1.0:-1.0);

}

Moon https://www.shadertoy.com/view/ddX3WH

vec3 sdMoon(vec2 p, float d, float ra, float rb ) { float s = sign(p.y); p.y =
abs(p.y);

float a = (ra*ra - rb*rb + d*d)/(2.0*d);
float b = sqrt(max(ra*ra-a*a,0.0));
if( d*(p.x*b-p.y*a) > d*d*max(b-p.y,0.0) )
{
    vec2 w = p-vec2(a,b); float d = length(w); w.y *= s;
    return vec3(d,w/d);
}

vec2 w1 = p;
vec2 w2 = p-vec2(d,0);
float l1 = length(w1); float d1 = l1-ra; w1.y *= s;
float l2 = length(w2); float d2 = rb-l2; w2.y *= s;

return (d1>d2) ? vec3(d1,w1/l1) : vec3(d2,-w2/l2);

}

Parabola https://www.shadertoy.com/view/mdX3WH

vec3 sdgParabola( in vec2 pos, in float k ) { float s = sign(pos.x); pos.x =
abs(pos.x);

float ik = 1.0/k;
float p = ik*(pos.y - 0.5*ik)/3.0;
float q = 0.25*ik*ik*pos.x;
float h = q*q - p*p*p;
float r = sqrt(abs(h));

float x = (h>0.0) ? 
    pow(q+r,1.0/3.0) - pow(abs(q-r),1.0/3.0)*sign(r-q) :
    2.0*cos(atan(r,q)/3.0)*sqrt(p);

float z = sign(pos.x-x);
vec2 w = pos-vec2(x,k*x*x); float l = length(w); w.x*=s;
return z*vec3(l, w/l );

}

Trapezoid https://www.shadertoy.com/view/ddt3Rs

vec3 sdgTrapezoid( vec2 p, float ra, float rb, float he, out vec2 oc ) { float
sx = (p.x<0.0)?-1.0:1.0; float sy = (p.y<0.0)?-1.0:1.0;

p.x = abs(p.x);

vec4 re;
{
    float h = min(p.x,(p.y<0.0)?ra:rb);
    vec2  c = vec2(h,sy*he);
    vec2  q = p - c;
    float d = dot(q,q);
    float s = abs(p.y) - he;
    re = vec4(d,q,s);
    oc = c;
}
{
    vec2  k = vec2(rb-ra,2.0*he);
    vec2  w = p - vec2(ra, -he);
    float h = clamp(dot(w,k)/dot(k,k),0.0,1.0);
    vec2  c = vec2(ra,-he) + h*k;
    vec2  q = p - c;
    float d = dot(q,q);
    float s = w.x*k.y - w.y*k.x;
    if( d<re.x ) { oc = c; re.xyz = vec3(d,q); }
    if( s>re.w ) { re.w = s; }
}

float d = sqrt(re.x)*sign(re.w);
re.y *= sx;
oc.x *= sx;

return vec3(d,re.yz/d);

}

Heart https://www.shadertoy.com/view/DldXRf

vec3 sdgHeart( in vec2 p ) { float sx = (p.x<0.0)?-1.0:1.0; p.x = abs(p.x);

if( p.y+p.x>1.0 )
{
    const float r = sqrt(2.0)/4.0;
    vec2 q0 = p - vec2(0.25,0.75);
    float l = length(q0);
    vec3 d = vec3(l-r, q0/l);
    d.y *= sx;
    return d;
}
else
{
    vec2 q1 = p - vec2(0.0,1.0);      
    vec2 q2 = p - 0.5*max(p.x+p.y,0.0);
    vec3 d1 = vec3(dot(q1,q1),q1);
    vec3 d2 = vec3(dot(q2,q2),q2);
    vec3 d = (d1.x<d2.x) ? d1: d2;
    d.x = sqrt(d.x);
    d.yz /= d.x;
    d *= (p.x>p.y)?1.0:-1.0;
    d.y *= sx;
    return d;
}

}

Operators/modifiers

As said, once the gradients of the basic primitives are hardcoded, automatic
differentiation can take over for most of the remaining composition work.
However, if you really want to make it all crystal clear and remove all opaque
aspects of your code, you might want to implement analytic gradient computation
for the composition operators too. Here go a few:

Rounding

As we know, inflating any shape into a rounded self is pretty easy with the
single subtraction of a quantity r. If such r is a constant, the derivatives of
the SDF won't change, so we don't really need to do anything in order to
continue getting correct gradients.

vec3 sdgRound( in vec2 p, in float r ) { vec3 dis_gra = sdgShape(p); return
vec3( dis_gra.x - r, dis_gra.yz ); }

These are a few examples:

Onion - annular shapes

Similarly, shapes can be made annular (like a ring or the layers of an onion) by
taking their absolute value and then subtracting a constant from their field.
This operation is also very easy to carry over for the gradients:

vec3 sdgOnion( in vec2 p, in float r ) { vec3 dis_gra = sdgShape(p); return
vec3( abs(dis_gra.x) - r, sign(dis_gra.x)*dis_gra.yz ); }

These are a few examples:

Minimum and Smooth-minimum https://www.shadertoy.com/view/tdGBDt

One way to combine shapes is to take the minimum of two SDFs. Naturally, the
same conditional branching that picks the smallest of the SDFs should be used to
select and propagate the gradient. However, when a smooth-minimum is used to
smoothly connect shapes, the derivative selection is more involved. Luckily,
using polynomial smooth-minimums have simple derivatives and accept an
analytical propagation. Of course, bear in mind that the smooth-minimum does not
respect the constraints of having unit length everywhere, so its output is only
a true SDF (or almost) far away from the surface of the resulting shape. You can
learn more about this and to derive the formula below in the Smooth-minimum
article, but I leave the code here too for the analytic derivatives/gradient of
a quadratic smooth-minimum for convenience:

vec3 min( in vec3 a, in vec3 b ) { return (a.x<b.x)?a:b; }

vec3 smin( in vec3 a, in vec3 b, in float k ) { k = 4.0; float h =
max(k-abs(a.x-b.x),0.0); float m = 0.25hh/k; float n = 0.50 h/k; return vec3(
min(a.x, b.x) - m, mix(a.yz, b.yz, (a.x<b.x)?n:1.0-n) ); }

Here's on the left the minimum, and on the right the smooth-minimum:

inigo quilez - learning computer graphics since 1994

Inigo Quilez :: articles :: 3D distance and gradient functions - 2025

Intro Knowing the gradient of an SDF is very useful. The gradient for an SDF at
any point always has length 1 (since the SDF measures distances), so it's also a
normal to the isosurface passing through that point. This gradient also points
in the direction of the closest point, and as a normal, is useful for lighting,
aligning objects to surfaces, etc etc. Of course, one can use central
differences to compute gradients, but that has a high computational cost
(requires 4 or 6 taps), and can get tricky to tweak its epsilon values to always
give good results. Automatic differentiation is another way to compute
gradients, and works well with SDFs of course, but we can do even better.

In this page you'll find a list of primitives and their analytic gradients,
where lots of terms are reused between the SDF computation and its gradient
computation, making them very cheap to evaluate. The list is incomplete, but
when you don't find a particular primitive in it you can very easily reach out
to the 2D SDF Gradients list, and extend the primitive you want to 3D through
thru extrusion or revolution and carry its gradient/normal with it.

Lastly, note that all the code listed here comes with a link to a realtime
online demo in Shadertoy. In fact, this public playlist contains all the
Shadertoy examples: https://www.shadertoy.com/playlist/cXccR2, so don't miss
that out.

Primitives All primitives are centered at the origin; you will have to transform
the point to get arbitrarily rotated, translated and scaled objects, and
transform the gradient back accordingly (remember not to use non uniform scales
since such transforms do not preserve distances). All SDFs below return the
actual distance in the .x component, and its partial derivatives with respect to
x and y in the .y, .z and .w components. In other words,

.x = f(p) .y = ∂f(p)/∂x .z = ∂f(p)/∂y .w = ∂f(p)/∂z .yzw = ∇f(p) with ‖∇f(p)‖
= 1

Box https://www.shadertoy.com/view/WfdyD4

vec4 sdgBox( in vec3 p, in vec3 b, in float r ) { vec3 w = abs(p)-(b-r); float g
= max(w.x,max(w.y,w.z)); vec3 q = max(w,0.0); float l = length(q); vec4 f =
(g>0.0)?vec4(l, q/l) : vec4(g, w.x==g?1.0:0.0, w.y==g?1.0:0.0, w.z==g?1.0:0.0);
return vec4(f.x-r, f.yzw*sign(p)); }

Torus https://www.shadertoy.com/view/wtcfzM

vec4 sdgTorus( in vec3 p, in float ra, in float rb ) { float h = length(p.xz);
return vec4( length(vec2(h-ra,p.y))-rb, normalize(p*vec3(h-ra,h,h-ra)) ); }

Segment https://www.shadertoy.com/view/WttfR7

vec4 sdgSegment( in vec3 p, in vec3 a, in vec3 b, in float r ) { vec3 ba = b-a;
vec3 pa = p-a; float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 ); vec3 q =
pa-h*ba; float d = length(q); return vec4(d-r,q/d);
}

Ellipsoid https://www.shadertoy.com/view/flXyRS Note: exact gradient but
innexact distance

vec4 sdgEllipsoid( in vec3 p, in vec3 r ) { p /= r; float k0 = sqrt(dot(p,p)); p
/= r; float k1 = inversesqrt(dot(p,p)); return vec4( k0*(k0-1.0)k1, pk1 ); }

Sphere https://www.shadertoy.com/view/wcccW7

vec4 sdgSphere( in vec3, in float r ) { float l = length(p); return vec4(l-r,
p/l); }

Link https://www.shadertoy.com/view/wc3yD7

vec4 sdgLink( in vec3 p, in float le, in float r1, in float r2 ) { vec3 q =
vec3( p.x, p.y-clamp(p.y,-le,le), p.z ); float w = length(q.xy); float l =
length(vec2(w-r1,q.z)); return vec4(l-r2, (q-vec3(r1*q.xy/w,0.0))/l); }

Rounded Cone https://www.shadertoy.com/view/tfccD7

vec4 sdgRoundCone( vec3 p, vec3 a, vec3 b, float r1, float r2 ) { vec3 ba = b -
a; float l2 = dot(ba,ba); float rr = r1 - r2; float a2 = l2 - rr*rr; float il2
= 1.0/l2;

vec3  pa = p - a;
vec3  pb = p - b;
float y  = dot(pa,ba);
float z  = y-l2; //dot(pb,ba)
float x2 = l2*dot(pa,pa)-y*y;
float y2 = y*y;
float z2 = z*z;
float k  = sign(rr)*rr*rr*x2;
if( sign(z)*a2*z2>k ) { float w=sqrt(il2*(x2+z2));
                        return vec4(w-r2,pb/w); }
if( sign(y)*a2*y2<k ) { float w=sqrt(il2*(x2+y2));
                        return vec4(w-r1,pa/w); }
                      { float w=sqrt(x2*a2);
                        return vec4((w+y*rr)*il2-r1,
                        il2*(rr*ba+a2*(pa*l2-y*ba)/w)); }

}

Vertical Capped Cone https://www.shadertoy.com/view/3ctcW7

vec4 sdgCappedCone( in vec3 p, in float he, in float r1, in float r2 ) { vec2 k
= vec2(r2-r1,2.0he); float m = dot(k,k); float l = length(p.xz); vec2 q =
vec2(r2-l, he-p.y); vec2 a = vec2(l-min(l,p.y<0.0?r1:r2), abs(p.y)-he); vec2 b =
kclamp(dot(q,k)/m,0.0,1.0) - q; float s = (b.x<0.0 && a.y<0.0) ? -1.0 : 1.0;
float la = dot(a,a); float lb = dot(b,b); return (la<lb)?vec4(ssqrt(la), 0.0,
sign(p.y), 0.0 ) : vec4(ssqrt(lb),vec3(k.y*p.xz/l,-k.x).xzy/sqrt(m)); }

Vertical Cylinder https://www.shadertoy.com/view/3fcyz2

vec4 sdgCylinder( in vec3 p, in float he, in float r ) { float l = length(p.xz);
vec2 e = vec2(l-r,abs(p.y)-he/2.0); vec2 h = max(e,0.0); float f = length(h);
float g = max(e.x,e.y); vec3 du = vec3(p.x/l,0.0,p.z/l); vec3 dv =
vec3(0.0,p.y<0.0?-1.0:1.0,0.0); return (g<=0.0) ? vec4( g, (e.x>e.y)?du:dv ):
vec4( f, (h.xdu+h.ydv)/f ); }

Cylinder https://www.shadertoy.com/view/3fcyz2

vec4 sdgCylinder( in vec3 p, in vec3 a, in vec3 b, in float r ) { vec3 ba =
(b-a)*0.5; vec3 ce = (b+a)*0.5; float l = length(ba); vec3 d = ba/l;

vec3  q = p-ce;
float v = dot(q,d);
vec3  u = q-v*d;
float k = length(u); // k = sqrt(dot(q,q)-v*v);
vec2  e = vec2(k-r,abs(v)-l);
vec2  h = max(e,0.0);
float f = length(h);
float g = max(e.x,e.y);
vec3 du = u/k;
vec3 dv = v<0.0?-d:d;
return (g<=0.0) ? vec4(g, (e.x>e.y)?du:dv):
                  vec4(f, (h.x*du+h.y*dv)/f);

}

inigo quilez - learning computer graphics since 1994

Inigo Quilez :: articles :: 2D axis aligned bounding boxes

Intro This is a list of 2D Axis Aligned Bounding Boxes (AABBs) for many of the
SDF primitives in my 2D SDF functions. Computing bounding boxes is important for
culling and pruning of primitives, which allows for faster rendering and physics
of content made out of SDF shapes. For example, raymarching can be accelerated a
big deal by only considering the primitives for which the bounding boxes overlap
a tiled set of frustums. Computing the 3D boxes of an extrusion or revolution is
trivial and should normally be based on the 2D bounding boxes below.

The parameters of all functions below match exactly those of the
corresponding 2D SDFs. The return value of all these functions is a vec4, where
the .xy components contain the minimum (of bottom left) corner of the box, and
the .zw components contain the maximum (or top right) corner of the box.

Note that all the primitives listed here come with a link to a realtime online
demo in Shadertoy. In fact, this public playlist contains all the Shadertoy
examples: https://www.shadertoy.com/playlist/mcGBDK, so don't miss that out.

Triangle https://www.shadertoy.com/view/tX2BDw

vec4 aabbTriangle( in vec2 p0, in vec2 p1, in vec2 p2) { return vec4(
min(p0,min(p1,p2)), max(p0,max(p1,p2)) ); }

Oriented Box https://www.shadertoy.com/view/t32BWw

vec4 aabbOrientedBox( in vec2 a, in vec2 b, in float r ) { vec2 v =
r*abs(normalize(vec2(b.y-a.y,b.x-a.x))); return vec4(min(a,b)-v,max(a,b)+v); }

Segment https://www.shadertoy.com/view/332BDm

vec4 aabbSegment( in vec2 a, in vec2 b, in float r ) { return
vec4(min(a,b)-r,max(a,b)+r); }

Pie https://www.shadertoy.com/view/t3ffDS

vec4 boxPie( in vec2 c, in vec2 d, in float a, in float r ) { float si = sin(a);
float co = cos(a); vec2 m = (d.xy)*co; vec2 n = abs(d.yx)si; return c.xyxy +
rvec4( (d.x>-co) ? min(m.x-n.x,0.0) : -1.0, (d.y>-co) ? min(m.y-n.y,0.0) : -1.0,
(d.x< co) ? max(m.x+n.x,0.0) : 1.0, (d.y< co) ? max(m.y+n.y,0.0) : 1.0 ); }

Quadratic Bezier https://www.shadertoy.com/view/lsyfWc
https://iquilezles.org/articles/bezierbbox/

vec4 aabbBezier( in vec2 p0, in vec2 p1, in vec2 p2 ) { vec2 a = p0-2.0p1+p2;
vec2 b = p1-p0; vec2 t = clamp(-b/a,0.0,1.0); vec2 q = p0+t(2.0b+ta); return
vec4(min(min(p0,p2),q), max(max(p0,p2),q)); }

Cubic Bezier https://www.shadertoy.com/view/XdVBWd
https://iquilezles.org/articles/bezierbbox/

vec4 aabbBezier( in vec2 p0, in vec2 p1, in vec2 p2, in vec2 p3 ) { vec2 c =
-p0+ p1; vec2 b = p0-2.0p1+ p2; vec2 a = -p0+3.0p1-3.0p2+p3; vec2 g =
sqrt(max(bb-ac,0.0)); vec2 t1 = clamp((-b-g)/a,0.0,1.0); vec2 t2 =
clamp((-b+g)/a,0.0,1.0); vec2 q1 = p0+t1(3.0c+t1(3.0b+t1a)); vec2 q2 =
p0+t2*(3.0c+t2(3.0b+t2a)); return vec4(min(min(p0,p3),min(q1,q2)),
max(max(p0,p3),max(q1,q2))); }

Parabola https://www.shadertoy.com/view/t3jBWw

vec4 aabbParabola( in float w, in float h, in float r ) { return
vec4(-w-r,min(h,0.0)-r, w+r,max(h,0.0)+r); }

Cut Disk https://www.shadertoy.com/view/t3jfDw

vec4 aabbCutDisk( in float r, in float h ) { float m = h>0.0 ? sqrt(rr-hh) : r;
return vec4(-m,h,m,r); }

Egg https://www.shadertoy.com/view/tXjfDw

vec4 aabbEgg( in float he, in float ra, in float rb, in float bu ) { float wi =
max(ra, rb); float r = 0.5*(he+ra+rb)/bu; float da = r - ra; float db = r - rb;
float h = dbdb - dada; if( abs(h)<hehe ) { float y = (h-hehe)/(2.0he); wi =
max(wi, r - sqrt(dada-y*y)); } return vec4(-wi, -ra, wi, he + rb); }

Star https://www.shadertoy.com/view/tXjfWw

vec4 boxStar( in float r, in int n, in float w) { float an = 6.283185/float(n);
vec2 kk = vec2( cos( round(float(n)/2.0)*an ), sin( round(float(n)/4.0)an ) );
return rvec4(-kk.y,kk.x,kk.y,1.0); }

Vesica Segment https://www.shadertoy.com/view/wccczH

vec4 boxVesicaSegment( in vec2 a, in vec2 b, in float w ) { vec2 c = (b+a)0.5;
vec2 v = (b-a)0.5; float v2 = dot(v,v); float d = 0.5(v2-ww)/w; float h =
-v2/(d+w); vec2 p = abs(v.yx)*d/sqrt(v2); vec2 q = max(p-d-w,h); return vec4(
min(min(a,b),c+q), max(max(a,b),c-q) ); }

inigo quilez - learning computer graphics since 1994

Inigo Quilez :: articles :: 3D axis aligned bounding boxes

Intro This is a list of 3D Axis Aligned Bounding Boxes (AABBs) for many of the
SDF primitives in my 3D SDF functions. Computing bounding boxes is important for
culling and pruning of primitives, which allows for faster rendering and physics
of content made out of SDF shapes. For example, raymarching can be accelerated a
big deal by only considering the primitives for which the bounding boxes overlap
a tiled set of frustums.

The parameters of all functions below match exactly those of the
corresponding 3D SDFs. The return value of all these functions is a bound3
datatype, which simply contains two vec3 members with the minimum and maximum
corners of the bounds in the X, Y and Z directions, ie, bounding box.

Note that all the primitives listed here come with a link to a realtime online
demo in Shadertoy. In fact, this public playlist contains all the Shadertoy
examples: https://www.shadertoy.com/playlist/DXyczw, so don't miss that out.

Segment https://www.shadertoy.com/view/tX2BDw

bound3 aabbSegment( in vec3 pa, in vec3 pb, in float ra ) { vec3 a = pb - pa;
return bound3( min(pa, pb) - ra, max(pa, pb) + ra ); }

Cone https://www.shadertoy.com/view/WdjSRK

bound3 aabbCone( in vec3 pa, in vec3 pb, in float ra, in float rb ) { vec3 a =
pb - pa; vec3 e = sqrt(1.0-aa/dot(a,a)); vec3 ea = era; vec3 eb = e*rb; return
bound3( min(pa-ea, pb-eb), max(pa+ea, pb+eb) ); }

Cylinder https://www.shadertoy.com/view/MtcXRf
https://iquilezles.org/articles/diskbbox

bound3 aabbCylinder( in vec3 pa, in vec3 pb, in float ra ) { vec3 a = pb - pa;
vec3 e = rasqrt(1.0-aa/dot(a,a)); return bound3( min(pa, pb)-e, max(pa, pb)+e );
}

Disk https://www.shadertoy.com/view/ll3Xzf
https://iquilezles.org/articles/diskbbox

bound3 aabbDisk( in vec3 ce, in vec3 no, in float ra ) { vec3 e =
rasqrt(1.0-nono); return bound3(ce-e, ce+e); }

Ellipse https://www.shadertoy.com/view/t3jBWw
https://iquilezles.org/articles/ellipses

bound3 aabbEllipse( in vec3 ce, in vec3 au, in vec3 av ) { vec3 e = sqrt( auau +
avav ); return bound3( ce-e, ce+e ); }

Quadratic Bezier https://www.shadertoy.com/view/tsBfRD
https://iquilezles.org/articles/bezierbbox/

bound3 aabbBezier( in vec3 p0, in vec3 p1, in vec3 p2 ) { vec3 a = p0-2.0p1+p2;
vec3 b = p1-p0; vec3 t = clamp(-b/a,0.0,1.0); vec3 q = p0+t(2.0b+ta); return
bound3(min(min(p0,p2),q), max(max(p0,p2),q)); }

Cubic Bezier https://www.shadertoy.com/view/MdKBWt
https://iquilezles.org/articles/bezierbbox/

bound3 aabbBezier( in vec3 p0, in vec3 p1, in vec3 p2, in vec3 p3 ) { vec3 c =
-p0+ p1; vec3 b = p0-2.0p1+ p2; vec3 a = -p0+3.0p1-3.0p2+p3; vec3 g =
sqrt(max(bb-ac,0.0)); vec3 t1 = clamp((-b-g)/a,0.0,1.0); vec3 t2 =
clamp((-b+g)/a,0.0,1.0); vec3 q1 = p0+t1(3.0c+t1(3.0b+t1a)); vec3 q2 =
p0+t2*(3.0c+t2(3.0b+t2a)); return bound3(min(min(p0,p3),min(q1,q2)),
max(max(p0,p3),max(q1,q2))); }

inigo quilez - learning computer graphics since 1994

Inigo Quilez :: articles :: intersectors

Intro This is a collection of ray-surface intersectors for some common primitive
types. You can use them to implement a pathtracer, UI clicking, sound
propagation, collision detection or anything that requires analytic raycasting.
I have derived these myself, and while I have used them professionally, I
haven't put them to the most strict of the tests. If you notice precision errors
in any of them you can choose to improve these implementations (at some
performance cost) or improve how your renderer generally handles precision
problems. When possible, I've tried to get the number of operations to a minimum
(for example the capped cone and capsule intersectors only check against one of
the caps, even though these primitives have two).

Note that all the primitives listed here come with a link to a realtime online
demo in Shadertoy. In fact, this public playlist contains all the shadertoy
examples: https://www.shadertoy.com/playlist/l3dXRf, so don't miss that out.

Primitives In all the examples below, ro is the ray origin, and rd is the ray
direction. The functions always return the distance to the closest intersection,
and sometimes some additional information (second intersection, normal or
barycentric coordinates). Most primitives are centered at the origin (you'll
need to transform the ray origin and direction accordingly), but some of the
functions above accept an arbitrary primitive location and orientation when such
implementation is faster than performing a ray transformation. Most of these
functions only work for a ray with origin outside of the primitive. Fixing that
shouldn't be difficult.

Sphere Shadertoy example

// sphere of size ra centered at point ce vec2 sphIntersect( in vec3 ro, in vec3
rd, in vec3 ce, float ra ) { vec3 oc = ro - ce; float b = dot( oc, rd ); float c
= dot( oc, oc ) - rara; float h = bb - c; if( h<0.0 ) return vec2(-1.0); // no
intersection h = sqrt( h ); return vec2( -b-h, -b+h ); }

Alternative method computes h (the squared distance from the closest ray point
to the sphere, qc below) with a projection rather than by using Pythagoras'
theorem. This is less precision hungry because we don't generate large numbers
(in comparison to the size of the sphere) since we don't square triangle edges
(b*b):

// sphere of size ra centered at point ce vec2 sphIntersect( in vec3 ro, in vec3
rd, in vec3 ce, float ra ) { vec3 oc = ro - ce; float b = dot( oc, rd ); vec3 qc
= oc - brd; float h = rara - dot( qc, qc ); if( h<0.0 ) return vec2(-1.0); // no
intersection h = sqrt( h ); return vec2( -b-h, -b+h ); }

The functions return the entry and exit point of the ray with the sphere in X
and Y, and marks the exit (Y) with a negative value if there isn't any
intersection. So,

vec2 t = sphIntersect( ro, rd, center, radius );

 if( t.y<0.0 ) { } // ray does NOT intersect the sphere

else if( t.x<0.0 ) { } // ro inside the sphere, t.y is intersection distance
else { } // ro outside the sphere, t.x is intersection distance

Box Shadertoy example

// axis aligned, center at the origin, dimensions "boxSize" vec2
boxIntersection( in vec3 ro, in vec3 rd, vec3 boxSize, out vec3 oNormal ) { vec3
m = 1.0/rd; // can precompute if traversing a set of aligned boxes vec3 n =
m*ro; // can precompute if traversing a set of aligned boxes vec3 k =
abs(m)*boxSize; vec3 t1 = -n - k; vec3 t2 = -n + k; float tN = max( max( t1.x,
t1.y ), t1.z ); float tF = min( min( t2.x, t2.y ), t2.z ); if( tN>tF || tF<0.0)
return vec2(-1.0); // no intersection oNormal = (tN>0.0) ? step(vec3(tN),t1)) :
// ro ouside the box step(t2,vec3(tF))); // ro inside the box oNormal *=
-sign(rd); return vec2( tN, tF ); }

Rounded Box Shadertoy example

// axis aligned, center at the origin, dimensions "size", extrusion by "rad"
float roundedboxIntersect( in vec3 ro, in vec3 rd, in vec3 size, in float rad )
{ // bounding box vec3 m = 1.0/rd; vec3 n = mro; vec3 k = abs(m)(size+rad); vec3
t1 = -n - k; vec3 t2 = -n + k; float tN = max( max( t1.x, t1.y ), t1.z ); float
tF = min( min( t2.x, t2.y ), t2.z ); if( tN>tF || tF<0.0) return -1.0; float t =
tN;

// convert to first octant
vec3 pos = ro+t*rd;
vec3 s = sign(pos);
ro  *= s;
rd  *= s;
pos *= s;
    
// faces
pos -= size;
pos = max( pos.xyz, pos.yzx );
if( min(min(pos.x,pos.y),pos.z) < 0.0 ) return t;

// some precomputation
vec3 oc = ro - size;
vec3 dd = rd*rd;
vec3 oo = oc*oc;
vec3 od = oc*rd;
float ra2 = rad*rad;

t = 1e20;        

// corner
{
float b = od.x + od.y + od.z;
float c = oo.x + oo.y + oo.z - ra2;
float h = b*b - c;
if( h>0.0 ) t = -b-sqrt(h);
}
// edge X
{
float a = dd.y + dd.z;
float b = od.y + od.z;
float c = oo.y + oo.z - ra2;
float h = b*b - a*c;
if( h>0.0 )
{
    h = (-b-sqrt(h))/a;
    if( h>0.0 && h<t && abs(ro.x+rd.x*h)<size.x ) t = h;
}
}
// edge Y
{
float a = dd.z + dd.x;
float b = od.z + od.x;
float c = oo.z + oo.x - ra2;
float h = b*b - a*c;
if( h>0.0 )
{
    h = (-b-sqrt(h))/a;
    if( h>0.0 && h<t && abs(ro.y+rd.y*h)<size.y ) t = h;
}
}
// edge Z
{
float a = dd.x + dd.y;
float b = od.x + od.y;
float c = oo.x + oo.y - ra2;
float h = b*b - a*c;
if( h>0.0 )
{
    h = (-b-sqrt(h))/a;
    if( h>0.0 && h<t && abs(ro.z+rd.z*h)<size.z ) t = h;
}
}

if( t>1e19 ) t=-1.0;

return t;

}

// normal of a rounded box vec3 roundedboxNormal( in vec3 pos, in vec3 siz, in
float rad ) { return sign(pos)*normalize(max(abs(pos)-siz,0.0)); }

Plane // plane degined by p (p.xyz must be normalized) float plaIntersect( in
vec3 ro, in vec3 rd, in vec4 p ) { return -(dot(ro,p.xyz)+p.w)/dot(rd,p.xyz); }

Disk Shadertoy example

// disk center c, normal n, radius r float diskIntersect( in vec3 ro, in vec3
rd, vec3 c, vec3 n, float r ) { vec3 o = ro - c; float t = -dot(n,o)/dot(rd,n);
vec3 q = o + rdt; return (dot(q,q)<rr) ? t : -1.0; }

Hexagonal Prism Shadertoy example

// returns t and normal vec4 iHexPrism( in vec3 ro, in vec3 rd, in float ra, in
float he ) { const float ks3 = 0.866025;

// normals
const vec3 n1 = vec3( 1.0,0.0,0.0);
const vec3 n2 = vec3( 0.5,0.0,ks3);
const vec3 n3 = vec3(-0.5,0.0,ks3);
const vec3 n4 = vec3( 0.0,1.0,0.0);

// slabs intersections
vec3 t1 = vec3((vec2(ra,-ra)-dot(ro,n1))/dot(rd,n1), 1.0);
vec3 t2 = vec3((vec2(ra,-ra)-dot(ro,n2))/dot(rd,n2), 1.0);
vec3 t3 = vec3((vec2(ra,-ra)-dot(ro,n3))/dot(rd,n3), 1.0);
vec3 t4 = vec3((vec2(he,-he)-dot(ro,n4))/dot(rd,n4), 1.0);

// inetsection selection
if( t1.y<t1.x ) t1=vec3(t1.yx,-1.0);
if( t2.y<t2.x ) t2=vec3(t2.yx,-1.0);
if( t3.y<t3.x ) t3=vec3(t3.yx,-1.0);
if( t4.y<t4.x ) t4=vec3(t4.yx,-1.0);

vec4            tN=vec4(t1.x,t1.z*n1);
if( t2.x>tN.x ) tN=vec4(t2.x,t2.z*n2);
if( t3.x>tN.x ) tN=vec4(t3.x,t3.z*n3);
if( t4.x>tN.x ) tN=vec4(t4.x,t4.z*n4);

float tF = min(min(t1.y,t2.y),min(t3.y,t4.y));

// no intersection
if( tN.x>tF || tF<0.0) return vec4(-1.0);

return tN;  // return tF too for exit point

}

Wedge Shadertoy example

// returns t and normal vec4 iWedge( in vec3 ro, in vec3 rd, in vec3 s ) { //
intersect plane box vec3 m = 1.0/rd; vec3 z = vec3(rd.x>=0.0?1.0:-1.0,
rd.y>=0.0?1.0:-1.0, rd.z>=0.0?1.0:-1.0); vec3 k = s*z; vec3 t1 = (-ro - k)*m;
vec3 t2 = (-ro + k)*m; float tn = max(max(t1.x, t1.y), t1.z); float tf =
min(min(t2.x, t2.y), t2.z); if( tn>tf ) return vec4(-1.0);

// boolean with plane
float k1 = s.y*ro.x - s.x*ro.y;
float k2 = s.x*rd.y - s.y*rd.x;
float tp = k1/k2;

if( k1>tn*k2 )
    return vec4(tn,-step(tn,t1)*z); // box
if( tp>tn && tp<tf )
    return vec4(tp,normalize(vec3(-s.y,s.x,0.0))); // plane
return vec4(-1.0);

}

Capsule Shadertoy example

// capsule defined by extremes pa and pb, and radious ra // Note that only ONE
of the two spherical caps is checked for // intersections, which is a nice
optimization float capIntersect( vec3 ro, vec3 rd, vec3 pa, vec3 pb, float ra )
{ vec3 ba = pb - pa; vec3 oa = ro - pa; float baba = dot(ba,ba); float bard =
dot(ba,rd); float baoa = dot(ba,oa); float rdoa = dot(rd,oa); float oaoa =
dot(oa,oa); float a = baba - bardbard; float b = babardoa - baoabard; float c =
babaoaoa - baoabaoa - rarababa; float h = bb - ac; if( h >= 0.0 ) { float t =
(-b-sqrt(h))/a; float y = baoa + tbard; // body if( y>0.0 && y<baba ) return t;
// caps vec3 oc = (y <= 0.0) ? oa : ro - pb; b = dot(rd,oc); c = dot(oc,oc) -
rara; h = bb - c; if( h>0.0 ) return -b - sqrt(h); } return -1.0; }

Capped cylinder Shadertoy example

// cylinder defined by extremes a and b, and radious ra vec4 cylIntersect( in
vec3 ro, in vec3 rd, in vec3 a, in vec3 b, float ra ) { vec3 ba = b - a; vec3 oc
= ro - a; float baba = dot(ba,ba); float bard = dot(ba,rd); float baoc =
dot(ba,oc); float k2 = baba - bardbard; float k1 = babadot(oc,rd) - baocbard;
float k0 = babadot(oc,oc) - baocbaoc - rarababa; float h = k1k1 - k2k0; if(
h<0.0 ) return vec4(-1.0);//no intersection h = sqrt(h); float t = (-k1-h)/k2;
// body float y = baoc + tbard; if( y>0.0 && y<baba ) return vec4( t, (oc+trd -
bay/baba)/ra ); // caps t = ( ((y<0.0) ? 0.0 : baba) - baoc)/bard; if(
abs(k1+k2t)<h ) { return vec4( t, basign(y)/sqrt(baba) ); } return
vec4(-1.0);//no intersection }

// normal at point p of cylinder (a,b,ra), see above vec3 cylNormal( in vec3 p,
in vec3 a, in vec3 b, float ra ) { vec3 pa = p - a; vec3 ba = b - a; float baba
= dot(ba,ba); float paba = dot(pa,ba); float h = dot(pa,ba)/baba; return (pa -
ba*h)/ra; }

Cylinder

// infinite cylinder with base point "cb", normalized axis "ca" and radius "cr"
vec2 cylIntersect( in vec3 ro, in vec3 rd, in vec3 cb, in vec3 ca, float cr ) {
vec3 oc = ro - cb; float card = dot(ca,rd); float caoc = dot(ca,oc); float a
= 1.0 - cardcard; float b = dot( oc, rd) - caoccard; float c = dot( oc, oc) -
caoccaoc - crcr; float h = bb - ac; if( h<0.0 ) return vec2(-1.0); //no
intersection h = sqrt(h); return vec2(-b-h,-b+h)/a; }

Capped cone Shadertoy example

// cone defined by extremes "pa" and "pb", and radius "ra" and "rb" // Only one
square root and one division in the worst case. dot2(v) is dot(v,v) vec4
coneIntersect( vec3 ro, vec3 rd, vec3 pa, vec3 pb, float ra, float rb ) { vec3
ba = pb - pa; vec3 oa = ro - pa; vec3 ob = ro - pb; float m0 = dot(ba,ba); float
m1 = dot(oa,ba); float m2 = dot(rd,ba); float m3 = dot(rd,oa); float m5 =
dot(oa,oa); float m9 = dot(ob,ba);

// caps
if( m1<0.0 )
{
    if( dot2(oa*m2-rd*m1)<(ra*ra*m2*m2) ) // delayed division
        return vec4(-m1/m2,-ba*inversesqrt(m0));
}
else if( m9>0.0 )
{
	float t = -m9/m2;                     // NOTE delayed division
    if( dot2(ob+rd*t)<(rb*rb) )
        return vec4(t,ba*inversesqrt(m0));
}

// body
float rr = ra - rb;
float hy = m0 + rr*rr;
float k2 = m0*m0    - m2*m2*hy;
float k1 = m0*m0*m3 - m1*m2*hy + m0*ra*(rr*m2*1.0        );
float k0 = m0*m0*m5 - m1*m1*hy + m0*ra*(rr*m1*2.0 - m0*ra);
float h = k1*k1 - k2*k0;
if( h<0.0 ) return vec4(-1.0); //no intersection
float t = (-k1-sqrt(h))/k2;
float y = m1 + t*m2;
if( y<0.0 || y>m0 ) return vec4(-1.0); //no intersection
return vec4(t, normalize(m0*(m0*(oa+t*rd)+rr*ba*ra)-ba*hy*y));

}

Rounded cone Shadertoy example

// cone defined by extremes "pa" and "pb", and radius "ra" and "rb" vec4
iRoundedCone( vec3 ro, vec3 rd, vec3 pa, vec3 pb, float ra, float rb ) { vec3 ba
= pb - pa; vec3 oa = ro - pa; vec3 ob = ro - pb; float rr = ra - rb; float m0 =
dot(ba,ba); float m1 = dot(ba,oa); float m2 = dot(ba,rd); float m3 = dot(rd,oa);
float m5 = dot(oa,oa); float m6 = dot(ob,rd); float m7 = dot(ob,ob);

// body
float d2 = m0-rr*rr;
float k2 = d2    - m2*m2;
float k1 = d2*m3 - m1*m2 + m2*rr*ra;
float k0 = d2*m5 - m1*m1 + m1*rr*ra*2.0 - m0*ra*ra;
float h = k1*k1 - k0*k2;
if( h<0.0) return vec4(-1.0);
float t = (-sqrt(h)-k1)/k2;

//if( t<0.0 ) return vec4(-1.0); float y = m1 - rarr + tm2; if( y>0.0 && y<d2 )
return vec4(t, normalize(d2*(oa+trd)-bay));

// caps
float h1 = m3*m3 - m5 + ra*ra;
float h2 = m6*m6 - m7 + rb*rb;
if( max(h1,h2)<0.0 ) return vec4(-1.0);
vec4 r = vec4(1e20);
if( h1>0.0 )
{        
	t = -m3 - sqrt( h1 );
    r = vec4( t, (oa+t*rd)/ra );
}
if( h2>0.0 )
{
	t = -m6 - sqrt( h2 );
    if( t<r.x )
    r = vec4( t, (ob+t*rd)/rb );
}
return r;

}

Ellipsoid Shadertoy example

// ellipsoid centered at the origin with radii ra vec2 eliIntersect( in vec3 ro,
in vec3 rd, in vec3 ra ) { vec3 ocn = ro/ra; vec3 rdn = rd/ra; float a = dot(
rdn, rdn ); float b = dot( ocn, rdn ); float c = dot( ocn, ocn ); float h = bb -
a(c-1.0); if( h<0.0 ) return vec2(-1.0); //no intersection h = sqrt(h); return
vec2(-b-h,-b+h)/a; }

vec3 eliNormal( in vec3 pos, in vec3 ra ) { return normalize( pos/(ra*ra) ); }

Triangle Shadertoy example

// triangle degined by vertices v0, v1 and v2 vec3 triIntersect( in vec3 ro, in
vec3 rd, in vec3 v0, in vec3 v1, in vec3 v2 ) { vec3 v1v0 = v1 - v0; vec3 v2v0 =
v2 - v0; vec3 rov0 = ro - v0; vec3 n = cross( v1v0, v2v0 ); vec3 q = cross(
rov0, rd ); float d = 1.0/dot( rd, n ); float u = ddot( -q, v2v0 ); float v =
ddot( q, v1v0 ); float t = d*dot( -n, rov0 ); if( u<0.0 || v<0.0 || (u+v)>1.0 )
t = -1.0; return vec3( t, u, v ); }

Ellipse Shadertoy example

// ellipse defined by its center c and its two radii u and v vec3 ellIntersect(
in vec3 ro, in vec3 rd, vec3 c, vec3 u, vec3 v ) { vec3 q = ro - c; vec3 n =
cross(u,v); float t = -dot(n,q)/dot(rd,n); float r = dot(u,q + rdt); float s =
dot(v,q + rdt); if( rr+ss>1.0 ) return vec3(-1.0); //no intersection return
vec3(t,s,r); }

vec3 ellNormal( in vec2 u, vec2 v ) { return normalize( cross(u,v) ); }

Torus Shadertoy example

float torIntersect( in vec3 ro, in vec3 rd, in vec2 tor ) { float po = 1.0;
float Ra2 = tor.xtor.x; float ra2 = tor.ytor.y; float m = dot(ro,ro); float n =
dot(ro,rd); float k = (m + Ra2 - ra2)/2.0; float k3 = n; float k2 = nn -
Ra2dot(rd.xy,rd.xy) + k; float k1 = nk - Ra2dot(rd.xy,ro.xy); float k0 = kk -
Ra2dot(ro.xy,ro.xy);

if( abs(k3*(k3*k3-k2)+k1) < 0.01 )
{
    po = -1.0;
    float tmp=k1; k1=k3; k3=tmp;
    k0 = 1.0/k0;
    k1 = k1*k0;
    k2 = k2*k0;
    k3 = k3*k0;
}

float c2 = k2*2.0 - 3.0*k3*k3;
float c1 = k3*(k3*k3-k2)+k1;
float c0 = k3*(k3*(c2+2.0*k2)-8.0*k1)+4.0*k0;
c2 /= 3.0;
c1 *= 2.0;
c0 /= 3.0;
float Q = c2*c2 + c0;
float R = c2*c2*c2 - 3.0*c2*c0 + c1*c1;
float h = R*R - Q*Q*Q;

if( h>=0.0 )  
{
    h = sqrt(h);
    float v = sign(R+h)*pow(abs(R+h),1.0/3.0); // cube root
    float u = sign(R-h)*pow(abs(R-h),1.0/3.0); // cube root
    vec2 s = vec2( (v+u)+4.0*c2, (v-u)*sqrt(3.0));
    float y = sqrt(0.5*(length(s)+s.x));
    float x = 0.5*s.y/y;
    float r = 2.0*c1/(x*x+y*y);
    float t1 =  x - r - k3; t1 = (po<0.0)?2.0/t1:t1;
    float t2 = -x - r - k3; t2 = (po<0.0)?2.0/t2:t2;
    float t = 1e20;
    if( t1>0.0 ) t=t1;
    if( t2>0.0 ) t=min(t,t2);
    return t;
}

float sQ = sqrt(Q);
float w = sQ*cos( acos(-R/(sQ*Q)) / 3.0 );
float d2 = -(w+c2); if( d2<0.0 ) return -1.0;
float d1 = sqrt(d2);
float h1 = sqrt(w - 2.0*c2 + c1/d1);
float h2 = sqrt(w - 2.0*c2 - c1/d1);
float t1 = -d1 - h1 - k3; t1 = (po<0.0)?2.0/t1:t1;
float t2 = -d1 + h1 - k3; t2 = (po<0.0)?2.0/t2:t2;
float t3 =  d1 - h2 - k3; t3 = (po<0.0)?2.0/t3:t3;
float t4 =  d1 + h2 - k3; t4 = (po<0.0)?2.0/t4:t4;
float t = 1e20;
if( t1>0.0 ) t=t1;
if( t2>0.0 ) t=min(t,t2);
if( t3>0.0 ) t=min(t,t3);
if( t4>0.0 ) t=min(t,t4);
return t;

}

vec3 torNormal( in vec3 pos, vec2 tor ) { return normalize(
pos*(dot(pos,pos)-tor.ytor.y - tor.xtor.x*vec3(1.0,1.0,-1.0))); }

Sphere 4 Shadertoy example

float sph4Intersect( in vec3 ro, in vec3 rd, in float ra ) { float r2 = rara;
vec3 d2 = rdrd; vec3 d3 = d2rd; vec3 o2 = roro; vec3 o3 = o2ro; float ka
= 1.0/dot(d2,d2); float k3 = ka dot(ro,d3); float k2 = ka* dot(o2,d2); float k1
= ka* dot(o3,rd); float k0 = ka*(dot(o2,o2) - r2r2); float c2 = k2 - k3k3; float
c1 = k1 + 2.0k3k3k3 - 3.0k3k2; float c0 = k0 - 3.0k3k3k3k3 + 6.0k3k3k2
\- 4.0k3k1; float p = c2c2 + c0/3.0; float q = c2c2c2 - c2c0 + c1c1; float h =
qq - ppp; if( h<0.0 ) return -1.0; //no intersection float sh = sqrt(h); float s
= sign(q+sh)pow(abs(q+sh),1.0/3.0); // cuberoot float t =
sign(q-sh)pow(abs(q-sh),1.0/3.0); // cuberoot vec2 w = vec2( s+t,s-t ); vec2 v =
vec2( w.x+c24.0, w.ysqrt(3.0) )*0.5; float r = length(v); return
-abs(v.y)/sqrt(r+v.x) - c1/r - k3; }

vec3 sph4Normal( in vec3 pos ) { return normalize( pospospos ); }

Goursat Shadertoy example

float gouIntersect( in vec3 ro, in vec3 rd, in float ka, float kb ) { float po
= 1.0; vec3 rd2 = rdrd; vec3 rd3 = rd2rd; vec3 ro2 = roro; vec3 ro3 = ro2ro;
float k4 = dot(rd2,rd2); float k3 = dot(ro ,rd3); float k2 = dot(ro2,rd2) -
kb/6.0; float k1 = dot(ro3,rd ) - kbdot(rd,ro)/2.0; float k0 = dot(ro2,ro2) + ka
- kbdot(ro,ro); k3 /= k4; k2 /= k4; k1 /= k4; k0 /= k4; float c2 = k2 - k3*(k3);
float c1 = k1 + k3*(2.0k3k3-3.0k2); float c0 = k0 + k3(k3*(c2+k2)3.0-4.0k1);

if( abs(c1) < 0.1*abs(c2) )
{
    po = -1.0;
    float tmp=k1; k1=k3; k3=tmp;
    k0 = 1.0/k0;
    k1 = k1*k0;
    k2 = k2*k0;
    k3 = k3*k0;
    c2 = k2 - k3*(k3);
    c1 = k1 + k3*(2.0*k3*k3-3.0*k2);
    c0 = k0 + k3*(k3*(c2+k2)*3.0-4.0*k1);
}

c0 /= 3.0;
float Q = c2*c2 + c0;
float R = c2*c2*c2 - 3.0*c0*c2 + c1*c1;
float h = R*R - Q*Q*Q;

if( h>0.0 ) // 2 intersections
{
    h = sqrt(h);
    float s = sign(R+h)*pow(abs(R+h),1.0/3.0); // cube root
    float u = sign(R-h)*pow(abs(R-h),1.0/3.0); // cube root
    float x = s+u+4.0*c2;
    float y = s-u;
    float ks = x*x + y*y*3.0;
    float k = sqrt(ks);
    float t = -0.5*po*abs(y)*sqrt(6.0/(k+x)) - 2.0*c1*(k+x)/(ks+x*k) - k3;
    return (po<0.0)?1.0/t:t;
}

// 4 intersections
float sQ = sqrt(Q);
float w = sQ*cos(acos(-R/(sQ*Q))/3.0);
float d2 = -w - c2; 
if( d2<0.0 ) return -1.0; //no intersection
float d1 = sqrt(d2);
float h1 = sqrt(w - 2.0*c2 + c1/d1);
float h2 = sqrt(w - 2.0*c2 - c1/d1);
float t1 = -d1 - h1 - k3; t1 = (po<0.0)?1.0/t1:t1;
float t2 = -d1 + h1 - k3; t2 = (po<0.0)?1.0/t2:t2;
float t3 =  d1 - h2 - k3; t3 = (po<0.0)?1.0/t3:t3;
float t4 =  d1 + h2 - k3; t4 = (po<0.0)?1.0/t4:t4;
float t = 1e20;
if( t1>0.0 ) t=t1;
if( t2>0.0 ) t=min(t,t2);
if( t3>0.0 ) t=min(t,t3);
if( t4>0.0 ) t=min(t,t4);
return t;

}

vec3 gouNormal( in vec3 pos, float ka, float kb ) { return
normalize( 4.0pospospos - 2.0pos*kb ); }

inigo quilez - learning computer graphics since 1994

Inigo Quilez :: articles :: smoothstep functions

Intro Smoothstep is one of the most frequently used functions during procedural
texturing and modeling, but shows up in many other areas of Computer Graphics as
well, as a way to interpolate data with local support (I don't consider
functions like tanh() or other sigmoids as Smoothsteps if they extend to
infinity). Most shading languages provide a standard implementation of
smoothstep that is based on the cubic x2(3-2x) which is quick to evaluate, but
other alternatives exist that I sometimes find myself using in some situations.

Some of those alternatives are more convenient, depending on the context. For
example the "Quartic Polynomial" below only uses even powers of x, which is
convenient when x is a distance for example and x2 can indeed be computed with a
dot product without square roots. Other variants like the "Quintic Polynomial"
and "Rational" below have higher continuity than the default smoothstep so they
are great for surfaces from which we need to stich them together (say in a noise
function). Others like the "Piecewise Polynomial" one allow controlling the
sharpness of the step and others like the "Cubic Polynomial" allow for easy
inversion. Worth noting that quartic polynomial version is not symmetric and
that the quintic polynomial doesn't have a closed form inverse. Also, I'm not
providing integrals in this article, but I have another article about the
integral of the standard smoothstep.

In this article I collected a few smoothstep alternatives that I've derived
myself or found around. I am presenting the code for each in an unoptimized form
so that the formula is clearer so you can more easily do math with it. Usually
you can reuse terms and chope a few cycles, but that's easy enough and not worth
obfuscating the code.

All the smoothsteps below work in the 0 to 1 domain only. For the general form
of smoothstep(a,b,x) in the interval [a, b], you can simply remap x to
(x-a)/(b-a) and clamp to the 0 to 1 range before calling the canonical
smoothstep of your choice.

You can see all these smoothsteps, their inverses and derivatives together with
the code that implements them in this realtime shader:
https://www.shadertoy.com/view/st2BRd

Name	Continuity	Code	Inverse Cubic Polynomial	C1 float smoothstep( float x ) {
return xx(3.0-2.0x); } float inv_smoothstep( float x ) {
return 0.5-sin(asin(1.0-2.0x)/3.0); } Quartic Polynomial	C1 float smoothstep(
float x ) { return xx(2.0-xx); } float inv_smoothstep( float x ) { return
sqrt(1.0-sqrt(1.0-x)); } Quintic Polynomial	C2 float smoothstep( float x ) {
return xxx(x*(x6.0-15.0)+10.0); } None Quadratic Rational	C1 float smoothstep(
float x ) { return xx/(2.0xx-2.0x+1.0); } float inv_smoothstep( float x ) {
return (x-sqrt(x(1.0-x)))/(2.0x-1.0); } Cubic Rational	C2 float smoothstep(
float x ) { return xxx/(3.0xx-3.0x+1.0); } float inv_smoothstep( float x ) {
float a = pow( x,1.0/3.0); float b = pow(1.0-x,1.0/3.0); return a/(a+b); }
Rational	C(n-1) float smoothstep( float x, float n ) { return
pow(x,n)/(pow(x,n)+pow(1.0-x,n)); } float inv_smoothstep( float x, float n ) {
return smoothstep( x, 1.0/n ); } Piecewise Quadratic	C1 float smoothstep( float
x ) { return (x<0.5) ? 2.0xx: 2.0x(2.0-x)-1.0; } float inv_smoothstep( float x )
{ return (x<0.5) ? sqrt(0.5x): 1.0-sqrt(0.5-0.5x); } Piecewise Polynomial	C(n-1)
float smoothstep( float x, float n ) { return (x<0.5) ? 0.5pow(2.0 x, n):
1.0-0.5pow(2.0(1.0-x), n); } float inv_smoothstep( float x, float n ) { return
(x<0.5) ? 0.5pow(2.0 x, 1.0/P): 1.0-0.5pow(2.0(1.0-x),1.0/P); } Trigonometric C1
float smoothstep( float x ) { return 0.5-0.5cos(PIx); } float inv_smoothstep(
float x ) { return acos(1.0-2.0*x)/PI; }

Below goes a visual comparison of the different smoothsteps above in blue color
and their inverse in green. On the right side you can see the first and second
derivatives of each smoothstep in yellow and red respectively. As you can see,
those I marked as C2 in the table above have second derivative that evaluates to
zero on the edges of the 0 to 1 interval, since that ensures that adjacent
smoothsteps will connects smoothly to each other. Top row is polynomial
smoothsteps, middle row is rationals, bottom row is piecewise polynomials and
trigonometric at the bottom right.

As a side note, if you ever need the inverse of the cubic rational smoothstep,
instead of solving through the general inverse you can try solving the cubic
equation directly as shown in this article, which gives an alternative
formulation for it:

float inv_smoothstep( float x ) { float w=2.0sqrt(x(1.0-x)); float
t=(x*(3.0-2.0x)-1.0)/(w(1.0-x)); return x-w*sinh(asinh(t)/3.0); }

inigo quilez - learning computer graphics since 1994

Inigo Quilez :: articles :: fBM - 2019

Intro

fBM stands for Fractional Brownian Motion. But before we talk about nature,
fractals and procedural terrains, let's get a bit theoretical for a moment.

A Brownian Motion (BM), without the "fractional" part, is a motion where the
position of a given object over time changes in random increments (imagine a
sequence of "position+=white_noise();"). Formally, BM is the integral of white
noise. These movements define paths that are random yet (statistically) self
similar, ie, a zoomed-in version of the path resembles the whole path. A
Fractional Brownian Motion is a similar process in which the increments are not
completely independent from each other, but there's some sort of memory to the
process. If the memory is positively correlated, changes in a given direction
will tend to produce future changes in the same direction, and the path will
then be smoother than a vanilla BM. If the memory is negatively correlated, a
positive change will be most likely followed by a negative change, and the path
will be much more random. The parameter that controls the behavior of the memory
or the integration, and therefore the self-similarity, its fractal dimension and
its power spectrum is called the Hurst Exponent, and it's usually abbreviated as
H. Mathematically speaking, H allows us to integrate white noise only partially
(say, perform 1/3 of an integral, hence the "Fractional" part in the name) to
design fBMs for any memory characteristics and visual look that we desire. In
fact, H takes values between 0 and 1, describing rough and smooth fBMs
respectively, where the normal BM happens for H=1/2.

fBM() was used to generate the terrain, the clouds, the tree distribution, their
color variations, and the canopy details. "Rainforest", 2016:
https://www.shadertoy.com/view/4ttSWf

Now, that's all very theoretical, and not how us computer graphics folks
generate fBM, but I wanted to describe it because it is important to keep its
qualities in mind even when doing graphics. Let's see how:

As we know, self-similar structures that are also random are very useful for
modeling all sort of natural phenomena procedurally, from clouds to mountains to
bark textures. It is intuitively evident that shapes in nature can be decomposed
in few big shapes that describe the overall form, and a larger number of medium
size shapes that distort the basic contour or surface of the initial shape, and
even many more even smaller shapes that add extra detail to the contour and
shape of the previous too. This incremental way of adding detail to an object,
which allows for an easy way to band limit our shapes for the purposes of LOD
(Level Of Detail) and filtering/antialiasing, is an easy one to code and produce
visually stunning results. Because of that, it is widely used in films and
games. However I believe fBM is not necessarily a well understood mechanism. So
this article describes how it functions and their different spectral and visual
characteristics for various values of their main parameter H, backed with some
experiments and measurements.

Basic Idea

The way fBMs are constructed normally (there are multiple methods) is to invoke
deterministic and smooth randomness through some noise function of our choice
(value, gradient, cellular, voronoise, trigonometric, simplex, ..., you name it,
the choice doesn't matter much), and then construct self-similarity explicitly
with it. The fBM does this by starting with a basic noise signal and
continuously adding smaller and smaller detailed noise invocations to it.
Something like this:

float fbm( in vecN x, in float H ) {
float t = 0.0; for( int i=0; i<numOctaves; i++ ) { float f = pow( 2.0, float(i)
); float a = pow( f, -H ); t += anoise(fx); } return t; }

This is the purest form of fBM. Each noise() signal (or "wave"), of which we
have "numOctaves", gets additively combined with the running total, but it gets
compressed horizontally by two effectively reducing its wavelength by two as
well, and its amplitude gets reduced exponentially. This accumulation of waves
with coordinated reduction of wavelength and amplitude is what produces
self-similarity like that seen in nature. After all, in a given space, there's
room for just a few big shape changes but there's naturally room for a lot and a
lot of tiny ones. Sounds pretty reasonable. In fact, these kind of Power-Law
behaviors are found everywhere in nature.

The first thing you might have noticed is that the code above doesn't completely
look like most fBM implementations you might have seen in Shadertoy and other
code snippets around. This following code is equivalent to the one bove, but is
far more popular because it avoids the expensive pow() functions:

float fbm( in vecN x, in float H ) {
float G = exp2(-H); float f = 1.0; float a = 1.0; float t = 0.0; for( int i=0;
i<numOctaves; i++ ) { t += anoise(fx); f *= 2.0; a *= G; } return t; }

So let's talk about "numOctaves" first. Since each noise is half the wavelength
of the previous one (or twice the frequency), the term for what otherwise should
have been "numFrequencies" is replaced by "numOctaves" as a reference to the
musical term where a separation of one octave between two notes corresponds to
doubling the frequency of the base note. Now, fBMs can be constructed by
incrementing the frequency of each noise by something different than two. In
that case the term "octave" wouldn't be technically correct anymore, but I've
seen people use it regardless. There are cases where you might even want to
create waves/noise of frequencies that increase at constant linear rate rather
than geometrically, like in an FFT (which can be used indeed to generate
periodic fBMs(), which can be useful for ocean textures). But, as we'll later
see in this article, for most base noise() functions we can actually increment
frequencies in multiples of two, which means we only need a very few iterations,
and still get good looking fBMS. In fact, synthetizing the fBM one octave at a
time allows us to be very efficient - for example with just 24
octaves/iterations we can create and fBM that covers the whole planet Earth and
provides details of just 2 meters. Doing the same with linearly increasing
frequencies would take a few orders of magnitude more iterations.

One last note on the sequence of frequencies is that moving from a fi=2i
approach to a fi = 2⋅fi-1, gives us some flexibility regarding the frequency
doubling (or wavelength halving) - we can easily unroll the loop and detune each
octave slightly by replacing 2.0 by 2.01, 1.99 and similar values, such that the
zeros and peaks of the different noise waves we are accumulating don't
superspose exactly, which can sometimes creates unrealistic patterns. In the
case of 2D fBM one can also rotate the domain a bit besides stretching it by an
octave.

Now, in this new code implementation of fBM(), not only we've replaced the
frequency generation from a power based formulation to an iterative process, but
we've replaced the exponential amplitude decay as well, the Power Law, with a
geometric series driven by a "gain" factor G. One needs to convert from H to G
by doing G=2-H which you can derive easily from the first version of the code.
However more often than not graphics programmers ignore or don't even know about
the Hurst exponent H, and only work with values of G directly. Since we know H
goes from 0 to 1, G goes from 1 to 0.5. And indeed, G=0.5 is what most people
have hardcoded into their fBM implementations. This hardcoding isn't as flexible
as leaving G variable, but there's a good reason to do so, and we are about to
see why.

Self Similarity

As we mentioned, the H parameter determines the selfimilarity of the curve. This
is statistical self-similarity of course. So, in the case of a one-dimensional
fBM(), if we horizontally zoom-in in it by a factor of U, how much would we need
to zoom in vertically in V to get a curve that "looks" the same? Well, since
a=f-H, then a⋅V = (f⋅U)-H = f-H⋅U-H = a⋅U-H, meaning V=U-H. So, if we are
zooming in a fBM with a horizontal factor of two, then we'll need to scale
vertically with a factor of 2-H. But 2-H is G! Not coincidentally, when using G
to scale our noise amplitudes, we are, by construction, building the
self-similarity of fBM with a scale factor of G = 2-H.

Left, Brownian Motion (H=1/2) and anisotropic zooming. Right fBM (H=1) and
isotropic zooming. Code: https://www.shadertoy.com/view/WsV3zz

Now, what about our procedural mountains? A naive Brownian Motion has a value of
H=1/2, which produces a G=0.707107... This generates a curve which looks like
itself when zoomed in anisotropically in X and Y (if it's a one-dimensional
curve). Indeed, for every horizontal zoom factor U we'd need to scale the curve
vertically by V=sqrt(U), which is not very natural. However, stock market curves
do approach H=1/2 often, since in theory, each increment or decrement in the
value of a stock is independent of its previous changes (remember BM is a
process with no memory). In practice of course there are some dependencies and
these curves are closer to H=0.6.

But natural processes have more "memory" into them, and the self-similarity is
much more isotropic than that. For example a mountain that is higher is also
wider at its base by the same amount, ie, mountains they don't usually stretch
or get thinner. So this suggests G should be 1/2 for mountains - equal zoom in
the horizontal and vertical directions. That corresponds to H=1, which suggests
mountain profiles should be smoother than a stock market curve. And they are, as
we'll be measuring actual profiles in a few moments later in this article to
confirm this. But we do know from experience that G=0.5 produces beautiful
fractal terrains and clouds, so G=0.5 is indeed the most popular value of G
found in all fbm implementations.

But now we have a bit deeper understanding of H, G and fBMs in general. We know
that a value of G closer to 1 will make our fBM even wilder than a pure BM, and
indeed for G=1, which corresponds to H=0, we get the noisiests of all the fBMs.

Now, all these parametrized fBMs functions do have names, such as "Pink Noise"
for H=0, G=1 or "Brown Noise" for H=1/2, G=sqrt(2), which are inherited from the
field of Digital Signal Processing and well known to people who have sleeping
problems. Actually, let's dive a little bit into DSP and compute some spectral
characteristics so we gain more intuitions about fBMs.

A Signal Processing look

If you think of Fourier analysis, or additive sound synthesis, the fBM()
implementation above is similar to that of an Inverse Fourier Transform that is
discrete like DFT, although very sparse, and uses a different basic function
(basically, it's very different to an IFT, but bear with me). In fact, you can
also generate fBM() and CG terrains and ever ocean surfaces by performing IFFTs,
but it gets very costly quickly. The reason is that IFFT works by additively
combining sine waves instead of noise waves, and sine waves are not very
efficient at filling the power density spectrum, since each sine wave
contributes to a single frequency. However, noise functions have wide spectrums
that cover long ranges of frequencies with a single wave. Both Gradient Noise
and Value noise have such rich and thick spectral density plots. Have a look:

Sin Wave

Value Noise

Gradient Noise

Note how the spectrum of both Value Noise and Gradient Noise have most of the
energy concentrated in the low frequencies but are wide - perfect to fill the
whole spectrum rapidly with few shifted and rescaled copies. The other problem
with sine wave based fBM is that of course it generates repeating patterns which
is not desirable most of the times, although it can become handy for generating
tiling textures. The one advantage of sin() based fBM() is that it is super
performant, since trigonometric functions run much faster in hardware than
constructing noise with polynomials and hashes/luts, so sometimes it's still
worth using sin based fBM for performance reasons, even if it produces poor
landscapes.

Now let’s have a look at the spectral density plots for fBMs of different H
values. Pay special attention to the vertical axis labels, since the three
graphs are normalized and do not represent the same slopes even if at first
glance they all look almost the same. If we call the negative slope of these
spectral graphs "B", then, since these graphics are in log-log scale, the
spectrum follows a power law of the form f-B. For this test I am using 10
octaves of regular gradient noise to construct these fBMs below.

G=1.0 (H=0)

G=0.707 (H=1/2)

G=0.5 (H=1)

As we can see, the energy of an fBM with H=0 (G=1) decays at 3db per octave, or
basically, inversely to the frequency. This is a power law of f-1 (B=1), and is
called Pink Noise. It sounds like rain.

An fBM() with H=1/2 (G=0.707) generates a spectrum that decays faster, at 6 db
per octave, meaning it has less high frequencies. It does sound indeed deeper,
like listening to rain again but from the inside of your room with the window
closed. A 6db/Oct decay means the energy is proportional to f-2 (B=2), and this
is indeed the characterization of a Brownian Motion in DSP.

Lastly, our computer graphics favourite fBM, H=1 (G=0.5), generates a spectral
density plot with a 9 db/Octave decay, which means the energy is inversely
proportional to the cube of the frequency (f-3, B=3). This is an even lower
frequency signal, which corresponds with a process with positively correlated
memory as we mentioned in the intro. This kind of signal doesn't have a name as
far as I know, so I am tempted to call it "Yellow Noise" (just because this
color isn't used for some other type of signal). As we know, being isotropic, it
models many natural shapes that are self-similar.

Name	H	G=2-H	B=2H+1	dB/Oct	Sound Blue	-	-	+1	+3	Spraying Water. Link
White	-	-	0	0	Windy Leaves. Link Pink	0	1	-1	-3	Rain. Link
Brown	1/2	sqrt(2)	-2	-6	Indoors Rain. Link Yellow	1	1/2	-3	-9	Engine behind door

Measuring

No, I did some claims about nature being isotropic and therefore being best
simulated with yellow noise (H=1). So let's put them to some form of testing.

I'd like to caveat that the following is not a rigorous/scientific experiment,
but I want to share it here anyways. What I did was to take photos of mountain
chains running parallel to the image plane, to prevent perspective distortion.
Then I segmented the images in black and white, and converted the sky-mountain
interface into a 1D signal. I then interpreted it as a WAV sound file and
computed its frequency plot just as with the synthetic fBM() signals I analyzed
earlier. I made sure the images were high enough resolution that the FFT
algorithm would have something meaningful to work with.

Source: Greek Reporter

Source: Wikipedia

The results seem to indicate that, indeed, mountain profiles follow a
-9dB/octave frequency distribution, which corresponds with B=-3 or H=1 or G=0.5,
or in other words, Yellow Noise.

While not a rigorous study, this seems to validates our intuition, and also what
we already know from experience as computer graphics programmers, namely that
H=1 (G=0.5) produce realistic (isotropic) fractal terrain shapes. But now we
just have a better understanding of it, I hope!

inigo quilez - learning computer graphics since 1994

Inigo Quilez :: articles :: gradient noise derivatives - 2017

Intro Similar to the analytical derivatives of Value Noise, Gradient Noise
accepts analytic computation of derivatives. And just like with Value Noise
derivatives, this allows for much faster lighting computations or any other
operation that requires gradients/normals based on the noise since, because no
longer need to approximate it numerically (which often involve taking with
multiple samples of the noise). It's also faster than using autodiff o dual
numbers (thinks SLANG) because here we are reusing sub-expressions across both
value and derivatives.

The code

Assuming we have some standard Gradient Noise implementation like the code on
the left, the computation of the derivatives involves only a few more
computations, as shown in the right.

// returns 3D gradient noise float noise( in vec3 x ) { // grid vec3 i =
floor(x); vec3 f = fract(x);

// quintic interpolant
vec3 u = f*f*f*(f*(f*6.0-15.0)+10.0);

// gradients
vec3 ga = hash( i+vec3(0.0,0.0,0.0) );
vec3 gb = hash( i+vec3(1.0,0.0,0.0) );
vec3 gc = hash( i+vec3(0.0,1.0,0.0) );
vec3 gd = hash( i+vec3(1.0,1.0,0.0) );
vec3 ge = hash( i+vec3(0.0,0.0,1.0) );
vec3 gf = hash( i+vec3(1.0,0.0,1.0) );
vec3 gg = hash( i+vec3(0.0,1.0,1.0) );
vec3 gh = hash( i+vec3(1.0,1.0,1.0) );

// projections
float va = dot( ga, f-vec3(0.0,0.0,0.0) );
float vb = dot( gb, f-vec3(1.0,0.0,0.0) );
float vc = dot( gc, f-vec3(0.0,1.0,0.0) );
float vd = dot( gd, f-vec3(1.0,1.0,0.0) );
float ve = dot( ge, f-vec3(0.0,0.0,1.0) );
float vf = dot( gf, f-vec3(1.0,0.0,1.0) );
float vg = dot( gg, f-vec3(0.0,1.0,1.0) );
float vh = dot( gh, f-vec3(1.0,1.0,1.0) );

// interpolation
return va + 
       u.x*(vb-va) + 
       u.y*(vc-va) + 
       u.z*(ve-va) + 
       u.x*u.y*(va-vb-vc+vd) + 
       u.y*u.z*(va-vc-ve+vg) + 
       u.z*u.x*(va-vb-ve+vf) + 
       u.x*u.y*u.z*(-va+vb+vc-vd+ve-vf-vg+vh);

} // returns 3D gradient noise (in .x) and its derivatives (in .yzw) vec4
noised( in vec3 x ) { // grid vec3 i = floor(x); vec3 f = fract(x);

// quintic interpolant
vec3 u = f*f*f*(f*(f*6.0-15.0)+10.0);
vec3 du = 30.0*f*f*(f*(f-2.0)+1.0);

// gradients
vec3 ga = hash( i+vec3(0.0,0.0,0.0) );
vec3 gb = hash( i+vec3(1.0,0.0,0.0) );
vec3 gc = hash( i+vec3(0.0,1.0,0.0) );
vec3 gd = hash( i+vec3(1.0,1.0,0.0) );
vec3 ge = hash( i+vec3(0.0,0.0,1.0) );
vec3 gf = hash( i+vec3(1.0,0.0,1.0) );
vec3 gg = hash( i+vec3(0.0,1.0,1.0) );
vec3 gh = hash( i+vec3(1.0,1.0,1.0) );

// projections
float va = dot( ga, f-vec3(0.0,0.0,0.0) );
float vb = dot( gb, f-vec3(1.0,0.0,0.0) );
float vc = dot( gc, f-vec3(0.0,1.0,0.0) );
float vd = dot( gd, f-vec3(1.0,1.0,0.0) );
float ve = dot( ge, f-vec3(0.0,0.0,1.0) );
float vf = dot( gf, f-vec3(1.0,0.0,1.0) );
float vg = dot( gg, f-vec3(0.0,1.0,1.0) );
float vh = dot( gh, f-vec3(1.0,1.0,1.0) );

// interpolation
float v = va + 
          u.x*(vb-va) + 
          u.y*(vc-va) + 
          u.z*(ve-va) + 
          u.x*u.y*(va-vb-vc+vd) + 
          u.y*u.z*(va-vc-ve+vg) + 
          u.z*u.x*(va-vb-ve+vf) + 
          u.x*u.y*u.z*(-va+vb+vc-vd+ve-vf-vg+vh);
          
vec3 d = ga + 
         u.x*(gb-ga) + 
         u.y*(gc-ga) + 
         u.z*(ge-ga) + 
         u.x*u.y*(ga-gb-gc+gd) + 
         u.y*u.z*(ga-gc-ge+gg) + 
         u.z*u.x*(ga-gb-ge+gf) + 
         u.x*u.y*u.z*(-ga+gb+gc-gd+ge-gf-gg+gh) +   
         
         du * (vec3(vb-va,vc-va,ve-va) + 
               u.yzx*vec3(va-vb-vc+vd,va-vc-ve+vg,va-vb-ve+vf) + 
               u.zxy*vec3(va-vb-ve+vf,va-vb-vc+vd,va-vc-ve+vg) + 
               u.yzx*u.zxy*(-va+vb+vc-vd+ve-vf-vg+vh) );
               
return vec4( v, d );                   

}

You can find a reference implementation here:
https://www.shadertoy.com/view/4dffRH

Left: gradient noise, Right: derivatives of gradient noise

In the case of 2D, the code gets naturally smaller:

// returns 3D gradient noise (in .x) and its derivatives (in .yz) vec3 noised(
in vec2 x ) { vec2 i = floor( x ); vec2 f = fract( x );

vec2 u = f*f*f*(f*(f*6.0-15.0)+10.0);
vec2 du = 30.0*f*f*(f*(f-2.0)+1.0);

vec2 ga = hash( i + vec2(0.0,0.0) );
vec2 gb = hash( i + vec2(1.0,0.0) );
vec2 gc = hash( i + vec2(0.0,1.0) );
vec2 gd = hash( i + vec2(1.0,1.0) );

float va = dot( ga, f - vec2(0.0,0.0) );
float vb = dot( gb, f - vec2(1.0,0.0) );
float vc = dot( gc, f - vec2(0.0,1.0) );
float vd = dot( gd, f - vec2(1.0,1.0) );

return vec3( va + u.x*(vb-va) + u.y*(vc-va) + u.x*u.y*(va-vb-vc+vd),   // value
             ga + u.x*(gb-ga) + u.y*(gc-ga) + u.x*u.y*(ga-gb-gc+gd) +  // derivatives
             du * (u.yx*(va-vb-vc+vd) + vec2(vb,vc) - va));

}

An implementation can be found here: https://www.shadertoy.com/view/XdXBRH

inigo quilez - learning computer graphics since 1994

Inigo Quilez :: articles :: voronoise - 2014

Intro Two of the most common building blocks for procedural pattern generation
are Noise, which have many variations (Perlin's being the first and most
relevant), and Voronoi (also known as "celular") which also has different
variations. For Voronoi, the most common of those variations is the one that
splits the domain in a regular grid such that there's one feature point in each
of the cells. That means that Voronoi patterns are based on a grid after all
just like Noise, the difference being that while in Noise the feature
originators are in the vertices of the grid (random values or random gradients),
Voronoi has the feature generators jittered somewhere in the grid. That might be
a first indicator that, perhaps, the two patterns are not that unrelated, at
least from an implementation perspective?

Despite this similarity, the fact is that the way the grid is used in both
patterns is different. Noise interpolates/averages random values (as in value
noise) or gradients (as in gradient noise), while Voronoi computes the distance
to the closest feature point. Now, smooth-bilinear interpolation and minimum
evaluation are two very different operations, or... are they? Can they perhaps
be combined in a more general metric? If that was so, then both Noise and
Voronoi patterns could be seen as particular cases of a more general grid-based
pattern genereator?

This article is about a small effort to find such generalized pattern. Of
course, the code implementing such generalization will never be as fast as
implementations of the particular cases (rendering this articles with no obvious
immediate practical purpose), but at least it might open the window to a bigger
picture understanding and perhaps, one day, new findings!

Voronoise - a combination of Voronoi, and Noise

The code In order to generalize Voronoi and Noise, we must introduced two
parameters: one to control the amount of jittering of the feature points, and
one for controling the metric. Let's call the grid control parameter u, and the
metric controller v.

The grid parameter is pretty simple to design: u=0 will simply use a Noise-like
regular grid, and u=1 will be the Voronoi-like jittered grid. So, the value of u
can simply control the amount of jitter. Straightforward.

The v parameters will have to blend between a Noise-like bilinear interpolator
of values, and a Voronoi-like min operator. The main difficulty here is that the
min() operation is a non-continuous function. However, luckily enough for us,
there are smooth alternatives such as the Smooth Voronoi. If we apply a power
functions to the distance to each feature points in order to highlight the
closest one over the rest, then we get a nice side effect: using a power of 1
gives all features the same relevance and therefore we get an equal
interpolation of features, which is what we need for Noise-like patterns! So,
something like this might do it:

float ww = pow( 1.0-smoothstep(0.0,1.414,sqrt(d)), 64.0 - 63.0*v );

However, a bit of experimentation proves that a better perceptually linear
interpolation between the Noise-like and the Voronoi-like pattern can be
achieved by rising v to some power:

float ww = pow( 1.0-smoothstep(0.0,1.414,sqrt(d)), 1.0 + 63.0*pow(1.0-v,4.0) );

So, it seems that after all it's not so difficult to generalize Noise and
Vonoroi. Therefore, assuming one has a way to generate random values
deterministically as a function of the grid cell id (which you are already doing
both in your favorite Voronoi and Noise implementations), which we could call

vec3 hash3( in vec2 p )

then the code for our new generalized super pattern could be like this:

float noise( in vec2 x, float u, float v ) { vec2 p = floor(x); vec2 f =
fract(x);

float k = 1.0 + 63.0*pow(1.0-v,4.0);
float va = 0.0;
float wt = 0.0;
for( int j=-2; j<=2; j++ )
for( int i=-2; i<=2; i++ )
{
    vec2  g = vec2( float(i), float(j) );
    vec3  o = hash3( p + g )*vec3(u,u,1.0);
    vec2  r = g - f + o.xy;
    float d = dot(r,r);
    float w = pow( 1.0-smoothstep(0.0,1.414,sqrt(d)), k );
    va += w*o.z;
    wt += w;
}

return va/wt;

}

The implementation is very similar to the regular Voronoi pattern, the
difference being that we now have the weighted average of distance computations
happening (the accumulation happens in wa and the counting for later
normalization is in wt.

Results

The results of the generalization are rather interesting. Of course, we have
generalized Noise and Voronoi. Indeed, noise happens when u=0, v=1, ie, regular
grid and interpolation of feature distances. Voronoi happens when u=1, v=0, ie,
when the grid is jittered and the metric is the minimum distance.

However there's two side effects. The first happens when u=0, v=0, which gives a
minimum distance to a non jittered grid of features. This basically gives a
patten with a constant value per grid cell, or what normally is called "cell
noise".

The second side effect happens for u=1, v=1, which generates a pattern that has
an interpolated value of distances to features in a jittered grid. It's a
combination of Voronoi and Noise, or as I am naming it, Voronoise (top right in
the image). This pattern can be useful for regular procedural generation where
grid artifacts are visible, because the jittering certainly hides the
underlaying grid structure of Noise.

Botton Left: u=0, v=0: Cell Noise Bottom Right: u=0, v=1: Noise Top Left: u=1,
v=0: Voronoi Top Right: u=1, v=1: Voronoise

A realtime interactive implementation of the code above can be found here (click
in the title to navigate to the source code, or simply move the mouse along the
image to vary the u and v parameters.

inigo quilez - learning computer graphics since 1994

Inigo Quilez :: articles :: simple pathtracing - 2012

Intro Writing a global illumination renderer takes anything between one hour and
one weekend. Starting from scratch, I promise. But writing an efficient and
general production ready global illumination renderer takes form one year to one
decade.

When doing computer graphics as an aficionado rather than a professional, the
"efficient" and "general" aspect can be dropped from your implementations. Which
means you can indeed write a full global illumination renderer in one hour.
Also, given the power of the hardware these days, even if you don't do any
clever optimizations or algorithms, a global illumination system can render in a
matter of seconds or even realtime.

A path traced fractal, brute forced, rendered in around one minute

For those of us who wasted hours and hours waiting to get a simple 2D low
resolution basic fractal rendered back in the early 90s, todays brute-force raw
power of machines seems pretty astonishing. In my opinion, the main advantage of
fast hardware is not that the graphics get rendered quicker, but that clever
algorithms are not that necessary anymore, meaning that straight away approaches
(those which are actually the most intuitive of all) can be directly coded and a
result can be expected in a reasonable amount of time. 20 years ago expensive
techniques required the implementation of clever, complex and obscure
algorithms, making the entry level for the computer graphics hobbyist much
higher. But thanks to new hardware that's not true anymore - today, writing a
global illumination renderer takes one hour at most.

What you need first So, let's say you have been doing some ray-based rendering
lately and that you have the following functions available to you:

vec2 worldIntersect( in vec3 ro, in vec3 rd, in float maxlen ); float
worldShadow( in vec3 ro, in vec3 rd, in float maxlen ); which compute the
intersection of a ray with the geometry of a 3D scene. The worldIntersect
function returns the closest intersection of ray with origin ro and normalized
direction rd in the form of a distance and an object ID. In the other hand,
worldShadow returns the existence of any intersection (or well, it returns 1.0
if there isn't any intersection happening and 0.0 if there is any intersection).
The implementation of these functions depends on the context of your
application. If you are raytracing hand modeled objects, these functions
probably traverse a kd-tree/bih or a bvh (bounding volume hierarchy). If you are
rendering procedural models, you are perhaps implementing these two functions as
raymarching in a distance field. If you are modelling terrains or 3d fractals or
voxels you most probably have specialized intersection functions.

vec3 worldGetNormal( in vec3 po, in float objectID ); vec3 worldGetColor( in
vec3 po, in vec3 no, in float objectID ); vec3 worldGetBackground( in vec3 rd );
The first two functions return the normal and surface color at a given surface
point in the 3D scene, and the third returns a background/sky color so we can
return a color for primary rays that don't hit any geometry.

void worldMoveObjects( in float ctime ); mat4x3 worldMoveCamera( in float ctime
); These two functions move the object in the scene and position the camera for
a given animation time.

vec3 worldApplyLighting( in vec3 pos, in vec3 nor ); This function computes the
direct lighting a given point and a normal on the surface of the 3D scene. This
is where regular point, directional, spot, dome or area lighting is done, and it
includes the cast of shadow rays.

Once you have these functions, implementing a pathtracer for global illumination
is not any more difficult than it is to implement a regular raytracing renderer.

A classic direct lighting renderer If you are reading this article, it probably
means you have implemented stuff like this already a million times:

//---------------------------------- // run for every pixel in the image
//---------------------------------- vec3 calcPixelColor( in vec2 pixel, in vec2
resolution, in float frameTime ) { // screen coords vec2 p =
(2.0*pixel(-resolution) / resolution.y;

// move objects
worldMoveObjects( frameTime );

// get camera position, and right/up/front axis
vec3 (ro, uu, vv, ww) = worldMoveCamera( frameTime );

// create ray
vec3 rd = normalize( p.x*uu + p.y*vv + 2.5*ww );

// calc pixel color
vec3 col = rendererCalculateColor( ro, rd );

// apply gamma correction
col = pow( col, 0.45 );

return col;

} which is the main entry function that computes the color of every pixel of the
image, followed by the function that initiates the actual ray casting process:

vec3 rendererCalculateColor( vec3 ro, vec3 rd ) { // intersect scene vec2 tres =
worldIntersect( ro, rd, 1000.0 );

// if nothing found, return background color
if( tres.y < 0.0 )
   return worldGetBackground( rd );

// get position and normal at the intersection point
vec3 pos = ro + rd * tres.x;
vec3 nor = worldGetNormal( pos, tres.y );

// get color for the surface
vec3 scol = worldGetColor( pos, nor, tres.y );

// compute direct lighting
vec3 dcol = worldApplyLighting( pos, nor );

// surface * lighting
vec3 tcol = scol * dcol;

return tcol;

} This is indeed a regular direct lighting renderer, as used in most intros,
demos and games.

Note that when rendering with rays, it all starts by iterating the pixels of the
screen. So if you are writing a CPU tracer, you probably want to do this by
splitting the screen in tiles of say, 32x32 pixels, and by consuming the tiles
by a pool of threads that contain as many threads as cores you have. You can see
code that does that here. If you are in the GPU, like in a fragment shader or a
compute shader, then that work is done for you. Either case, we have a function
calcPixelColor() that needs to compute the color of a pixel given its
coordinates in screen and a scene description (the scene description is given by
the functions above).

The montecarlo path tracer As said, the point of this article is to keep things
simple and not be too smart. So we are writing our Montecarlo tracer in quite a
brute force manner.

We of course start from the pixels, and the easiest way to get our rays
randomized by blindly sampling the pixel area to get antialiasing, the lens of
the camera to get depth of field, and the animation across the frame to get
motion blur. For free. Since we will do this random sampling for every ray, then
light integration and these other effects happen simultaneously, which is pretty
nice. Imagine we were using 256 light paths/samples per pixel to get a good
noise-free illumination. Then we would be effectively computing 256x
antialiasing for free. Neat. So, the main rendering function that runs for every
pixel looks something like this: // compute the color of a pixel vec3
calcPixelColor( in vec2 pixel, in vec2 resolution, in float frameTime ) { float
shutterAperture = 0.6; float fov = 2.5; float focusDistance = 1.3; float
blurAmount = 0.0015; int numLevels = 5;

// 256 paths per pixel
vec3 col = vec3(0.0);
for( int i=0; i<256; i++ )
{
    // screen coords with antialiasing
    vec2 p = (2.0*(pixel + random2f())-resolution) / resolution.y;

    // motion blur
    float ctime = frameTime + shutterAperture*(1.0/24.0)*random1f();

    // move objects
    worldMoveObjects( ctime );

    // get camera position, and right/up/front axis
    vec3 (ro, uu, vv, ww) = worldMoveCamera( ctime );

    // create ray with depth of field
    vec3 er = normalize( vec3( p.xy, fov ) );
    vec3 rd = er.x*uu + er.y*vv + er.z*ww;

    vec3 go = blurAmount*vec3( -1.0 + 2.0*random2f(), 0.0 );
    vec3 gd = normalize( er*focusDistance - go );
    ro += go.x*uu + go.y*vv;
    rd += gd.x*uu + gd.y*vv;

    // accumulate path
    col += rendererCalculateColor( ro, normalize(rd), numLevels );
}
col = col / 256.0;

// apply gamma correction
col = pow( col, 0.45 );

return col;

}

A frame with depth of field, motion blur and 256x antialising, rendered with the
code above this image

Note that the worldMoveObjects() and worldMoveCamera() function will position
all the objects in the scene and the camera for a given time passed as argument.
Of course repositioning all the objects can be expensive in some contexts (not
in procedurally defined scenes, but in BVH/KDTree based scenes), you might want
to implement time jittering for motion blur differently, like passing the
shutter time as part of the ray information and then linearly interpolating
polygons positions based on that. But for simple procedural graphics, the
approach above is just simple and easy :)

Another note difference is that now rendererCalculateColor() receives an integer
with the amount of levels of recursive raytracing we will want for our tracer
(which is one plus the amount of light bounces - but more to come on this topic
soon).

The ball is now in rendererCalculateColor()'s roof. This function, given a ray
and the scene, has to compute a color. As with the classic direct lighting
renderer, we start by casting our ray against the scene geometry looking for an
intersection, computing the position and normal at the intersection point,
geting the local surface color of the object that was hit, and then computing
local lighting.

vec3 rendererCalculateColor( vec3 ro, vec3 rd, int numLevels ) { // intersect
scene vec2 tres = worldIntersect( ro, rd, 1000.0 );

// if nothing found, return background color
if( tres.y < 0.0 )
   return worldGetBackground( rd );

// get position and normal at the intersection point
vec3 pos = ro + rd * tres.x;
vec3 nor = worldGetNormal( pos, tres.y );

// get color for the surface
vec3 scol = worldGetColor( pos, nor, tres.y );

// compute direct lighting
vec3 dcol = worldApplyLighting( pos, nor );

...

There is a big difference this time though in applyLighting(). Usually that
function tries to be clever and approximate lighting with soft shadow tricks, or
concavity based ambient occlusion, or just things like blurred shadows maps,
actual ambient occlusion, and other techniques. Indeed, that's how realtime
demos and games work. However, this time we are not doing any of these (which
are too smart for us this time). Instead, our applyLighting() is going to do the
simplest (and correct) sampling of lights. Which we can do this in multiple
ways. For example, you can pick a random light source (the sky, the sun, one of
the lamps in your scene, etc), grab one point in it, and cast one single ray to
it. If the ray hits the light source instead of a blocking object, we return
some light from the function, otherwise we return black. We can also play
differently and actually sample all of the lights, grabbing one random point in
it, and casting one shadow ray to that point. It would also be possible to
sample the light multiples times and cast a few rays per light. You probably
want to do some importance sampling and sample lights differently depending on
their size and intensity. But in its simplest form, the function simple casts
one shadow ray against the light sources. This will return result in a very
noisy image of course, but remember that all of this is run 256 times per pixel
anyway (or more), so in practice we are casting many shadow rays per
pixel/lens/aperture.

Still this would be a direct lighting renderer. In order to capture indirect
lighting, and before we multiply any lighting information with the surface
color, we need to cast at least one ray to gather indirect lighting. Again, one
could cast a few gather rays, but the idea of a pathtracer is to keep it all
simple, and cast only one ray every time (to make one single "light path",
therefore the name path-tracing). If the surface we hit is completely diffuse we
should just cast our ray in any random direction in the hemisphere centered
around the surface normal of the point we are lighting. If the surface was
glossy/specular, we should compute the reflected direction of the incoming ray
along the surface normal, and cast a ray in a cone centered in that direction
(the width of the cone being the glossiness factor of our surface). If the
surface was both diffuse and glossy at the same time, the we should choose
between both possible outgoing ray directions randomly, with probabilities
proportional to the amount of diffuse vs glossiness we wanted for our surface.
Once we had our ray, we would start the process again that we already have in
place for the direct lighting (cast, calc normal, calc surface color, calc
direct lighting and multiply).

This can be done both recursively or iteratively. If it was recursive everything
would look like this:

vec3 rendererCalculateColor( vec3 ro, vec3 rd, int numLevels ) { // after some
recursion level, we just don't gather more light if( numLevels==0 ) return
vec3(0.0);

// intersect scene
vec2 tres = worldIntersect( ro, rd, 1000.0 );

// if nothing found, return background color
if( tres.y < 0.0 )
   return worldGetBackground( rd );

// get position and normal at the intersection point
vec3 pos = ro + rd * tres.x;
vec3 nor = worldGetNormal( pos, tres.y );

// get color for the surface
vec3 scol = worldGetColor( pos, nor, tres.y );

// compute direct lighting
vec3 dcol = worldApplyLighting( pos, nor );

// compute indirect lighting
rd = worldGetBRDFRay( pos, nor, rd, tres.y );
vec3 icol = rendererCalculateColor( pos, rd, numLevels-1 );

// surface * lighting
vec3 tcol = scol * (dcol + icol);

return tcol;

} As said the new function worldGetBRDFRay() returns a new ray direction for the
recursive tracer, and again, this can be a random vector in the hemisphere for
diffuse surfaces or a ray on a cone around the reflected ray direction based on
how glossy vs diffuse the surface is at that point.

The problem with this recursive implementation is that it's not suitable for
current generations of graphics hardware (which has no stacks in its shader
units). The solution is either to build your own stack if the hardware allows
writing to arrays, or go for with an iterative implementation, which is very
similar: vec3 rendererCalculateColor( vec3 ro, vec3 rd, int numLevels ) { vec3
tcol = vec3(0.0); vec3 fcol = vec3(1.0);

// create numLevels light paths iteratively
for( int i=0; i < numLevels; i++ )
{
    // intersect scene
    vec2 tres = worldIntersect( ro, rd, 1000.0 );

    // if nothing found, return background color or break
    if( tres.y < 0.0 )
    {
       if( i==0 )  fcol = worldGetBackground( rd );
       else        break;
    }
    // get position and normal at the intersection point
    vec3 pos = ro + rd * tres.x;
    vec3 nor = worldGetNormal( pos, tres.y );

    // get color for the surface
    vec3 scol = worldGetColor( pos, nor, tres.y );

    // compute direct lighting
    vec3 dcol = worldApplyLighting( pos, nor );

    // prepare ray for indirect lighting gathering
    ro = pos;
    rd = worldGetBRDFRay( pos, nor, rd, tres.y );

    // surface * lighting
    fcol *= scol;
    tcol += fcol * dcol;
}

return tcol;

}

In this case we are computing only direct illumination at the hit points and
letting the ray bounce literally by changing its origin (to be the surface hit
position) and its direction according to the local BRDF, then letting it being
casted again. The only trick to keep in mind is that the surface modulation
color decreases exponentially as the ray depth increases, for the light hitting
a given point in the scene gets attenuated by every surface color that the path
hits on its way to the camera. Hence the exponential color/intensity decay fcol
*= scol;

And that's basically all you need in order to have a basic global illumination
renderer able to produce photorealistic images, just a few lines of code and
some fast hardware. As I promised, this can take one hour to code. Now, adding
extra features as participating media (non uniform density fog), subsurface
scattering, efficient hair intersection, etc etc, can take years :) So, choose
your feature set carefully before you plan to conquer the world or something
like that.

Final notes I assume that anybody reaching the end of this article knows how to
do direct lighting and is able to generate a ray in a random direction with a
cosine distribution, a point in a disk or quad and a ray within a cone. The
Total Compendium by Philip Dutr is a good reference. As for reference too, I
leave here a couple of the functions used in all of the code and images above
too - the one doing the direct lighting computations and the one generating a
ray based on the surface BRDF:

vec3 worldApplyLighting( in vec3 pos, in vec3 nor ) { vec3 dcol = vec3(0.0);

 // sample sun
 {
 vec3  point = 1000.0*sunDirection + 50.0*diskPoint(nor);
 vec3  liray = normalize( point - pos );
 float ndl =  max(0.0, dot(liray, nor));
 dcol += ndl * sunColor * worldShadow( pos, liray );
 }

 // sample sky
 {
 vec3  point = 1000.0*cosineDirection(nor);
 vec3  liray = normalize( point - pos );
 dcol += skyColor * worldShadow( pos, liray );
 }

 return dcol;

} The sun is a disk, and the sky is a dome. Note how the sky light doesn't
compute the usual diffuse "N dot L" factor. Instead, the code replaces the
uniform sampling of the sky dome with a cosine distribution based sampling,
which sends more samples in the direction of the normal and less to the sides
proportionally to the cosine term, therefore achieving the same effect while
casting far less rays (you have probably heard the word "importance sampling"
before).

vec3 worldGetBRDFRay( in vec3 pos, in vec3 nor, in vec3 eye, in float materialID
) { if( random1f() < 0.8 ) { return cosineDirection( nor ); } else { return
coneDirection( reflect(eye,nor), 0.9 ); } } In this case the function is 80%
diffuse and 20% glossy (with a glossiness cone angle of 0.9 radians).

Full source code example Here are two images with full source code that
implement the techniques discussed in this article:

https://www.shadertoy.com/view/Xtt3Wn:

https://www.shadertoy.com/view/MsdGzl:

inigo quilez - learning computer graphics since 1994

Inigo Quilez :: articles :: sphere soft shadow - 2014

Motivation

Spheres are awesome in may ways. One of them is that they allow for analytical
solution to many problems, such as that of computing approximated but plausible
soft shadows. Having a closed form for the soft shadow computation rather than
having to resort to sampling (of a shadowmap or the scene though raycasting) is
convenient. It is fast and noise free, and it is stable. The interest of
spherical shapes might seem limited, but there's a lot spheres can approximate
shape wise. I have successfully used this in real film production. And of
course, they are also natural bounding volumes for more complex geometry, so
having fast analytical ways to compute properties is actually very valuable.

Analytical soft shadows in action https://www.shadertoy.com/view/XdjXWK

The idea

The idea is simple. For a given point being shaded, a sphere in space and a
directional light source (not an area light), see if the ray travelling from the
point in question ro in the light direction rd does hit the sphere or misses it,
and if it misses it, by how much. The closer the ray was to hit the sphere the
darker the shadow will be (the penumbra). There's an observation though: the
farther this point of closest approximation is form the receiving point ro, the
less intense the shadow will be. In other words, in this simplistic model the
darkness of the shadow depends on two parameters: the closest distance from the
ray to the sphere's surface (which is perpendicular to the ray direction rd) and
the distance from ro at which this closest distance event happens. If we call
these d and t, then the soft shadow will be proportional to their ratio d/t. See
diagram to the right of this text. This method will create sharp shadows near
the contact between occluder and ocludee and softer shadows as this distance
increases (hence the "plausible" attribute in the technique).

Configuration for our plausible soft shadow

Implementation

All we need to do is computing d and t. Clearly d is simply the distance from
the ray to the sphere's center minus the the radius of the sphere. Getting the
closest distance between a ray (line segment) and a point is as easy as
projecting the point into the line and seeing how far it landed, and t is the
distance from that point to the origin ro.

Interestingly, this can be rewritten in terms of the parameters needed to solve
the usual ray-sphere intersection, b, c and the discriminant h. The code below
is an implementation of this technique:

float sphSoftShadow( in vec3 ro, in vec3 rd, in vec4 sph, in float k ) { vec3 oc
= ro - sph.xyz; float b = dot( oc, rd ); float c = dot( oc, oc ) - sph.wsph.w;
float h = bb - c;

float d = -sph.w + sqrt( max(0.0,sph.w*sph.w-h));
float t = -b     - sqrt( max(0.0,h) );
return (t<0.0) ? 1.0 : smoothstep(0.0,1.0,k*d/t);

}

In this case the parameter k controls the sharpness of the shadow penumbra.
Higher values make it sharper. The smoothstep() function is there just to
smoothen then transition between light and shadow.

Resulting analytical soft shadow

Alternative formula for soft shadow

Alternative formula

The above code is super efficient if you compare it to stocastic raycasting.
However somtimes "super efficient" is not efficient enough. One way to make the
code above faster is removing the square roots. I created the alternative
approximation below which produces less physically correct shadows, but still
plausible as in the sharpness of the shadows depend on the distance between the
object producing the shadow and that receiving it.

float sphSoftShadow( in vec3 ro, in vec3 rd, in vec4 sph, in float k ) { vec3 oc
= ro - sph.xyz; float b = dot( oc, rd ); float c = dot( oc, oc ) - sph.wsph.w;
float h = bb - c;

return (b>0.0) ? step(-0.0001,c) : smoothstep(0.0,1.0,h*k/b);

}

Another example of analytical soft shadows https://www.shadertoy.com/view/lsSSWV

Here's a link to the two versions of the soft shadows above in action running
live: https://www.shadertoy.com/view/4d2XWV

inigo quilez - learning computer graphics since 1994

Inigo Quilez :: articles :: simple gpu raytracing - 2005

Prefade When you first try to code for a new platform that provides a pixel
buffer that you can write to, there are only two types of "Hello World"
applications you can write - either a Mandelbrot set or a small
raytracer/raymarcher. Pixel shaders 3.0 was such a platform for me in 2005,
cause it was the first shader model that allows for conditional branching. It
was still not as flexible as regular CPU programing or current GPU shaders, but
it was good enought to try raytracing a few sphere in it. 15 years later, in a
workd of abudant shader based raytracers and raymarchers (hello Shadertoy!) this
feels obvious, but in 2005 it was all new territory. So, this article might seem
terrible simply by today standards, but I leave it here for historical context.
The rest of the article will be writen in present tense.

The idea So, it seems GPUs are so fast these days that even a brute force
implementations of raytracing runs fast at high screen resolutions. This is
perfect for tiny demos (like the 4 kilobyte demos of the Demoscene) where you
don't really have room for acceleration structures or anything. After a few
months playing with this tech I've finally made a little demo called
Kinderpainter as an experiment on GLSL coding. I decided to implement a simple
Whitted raytracer: only local lighting plus one shadow and one perfectly
specular reflection. I used two spheres, two cylinders and two planes as scene,
and I allowed them to move so I could build two or three virtual different
scenes and so I could synchronize the movements to the music too. All the image
was synthetized in a quad covering the complete screen to which a pixel shader
was attached. The shader was responsible for creating the image, and the CPU was
just left with the code to create a desktop window, initialize OpenGL and the
shader, move the objects with some trigonometric functions, and render the quad.

As a simple raytracer, what the code does is, for each pixel, cast a ray on the
scene to find the closest intersected object. That's done in the calcinter()
function on the pixel shader below. The implementation just calles the
intersection functions for the six objects in the scene (two spheres, two
cylinders and two planes), while it keeps track of the closest intersection at
all time.

Then the code calls the shading function calcshade(). The first thing this one
does is to compute the normal of the object at the intersection point, by
calling calcnor(). Depending on the primitive type, that function executes the
necessary computations. Then calcshade() does some basic diffuse and specular
lighting calculations. It also calls calcshadow to decide if the point being
shaded is in shadow or not. This function is a simplified version of the regular
intersection function calcinter() (it's simple cause we don't really need to
know the closest interseted object, we just need to know if any object was
intersected at all).

Therefore main() function, the one executed for each pixel, just calls
calcinter() and calcshade() and returns the result to the hardware so the pixel
of the screen is colored. Normally a Whitted raytracer would recursivelly call
calcinter() and calcshade() from within the calcshade() function, up to a number
of levels, say 4. However, since current GPU shader models don't support
recursion yet, I made the trick of doing a nonrecursive version of calcshade()
and calling calcinter() and calcshade() for a second time from the main()
function, with the right reflection ray as argument.

The complete pixel shader is below, and you have a live version online in
Shadertoy. Of course, fpar00[] contains all the input to the shader. For
example, fpar00[0] contains the information of the first sphere (a position and
a radius), fpar00[1] for the second sphere, fpar00[2] and fpar00[3] are the two
cylinders, and fpar00[4] and fpar00[5] are the two planes. The colors of the
objects are stored from fpar00[6] to fpar00[11]. Also, fpar00[12] contains the
camera position, and fpar00[13], fpar00[14] and fpar00[15] contain the camera
matrix. The raydirection is partially computed in the vertex shader for the
corners of the full screen quad, and interpolated down to the pixel shader by
the rasterization hardware, and it arrives to the pixel shader trhu the varying
called raydir.

The Code Pixel shader:

uniform vec4 fpar00[16]; uniform sampler2D tex00; varying vec3 raydir;

bool intSphere( in vec4 sp, in vec3 ro, in vec3 rd, in float tm, out float t ) {
bool r = false; vec3 d = ro - sp.xyz; float b = dot(rd,d); float c = dot(d,d) -
sp.wsp.w; t = bb-c; if( t > 0.0 ) { t = -b-sqrt(t); r = (t > 0.0) && (t < tm); }

return r;

}

bool intCylinder( in vec4 sp, in vec3 ro, in vec3 rd, in float tm, out float t )
{ bool r = false; vec3 d = ro - sp.xyz; float a = dot(rd.xz,rd.xz); float b =
dot(rd.xz,d.xz); float c = dot(d.xz,d.xz) - sp.wsp.w; t = bb - a*c; if( t > 0.0
) { t = (-b-sqrt(t)*sign(sp.w))/a; r = (t > 0.0) && (t < tm); } return r; }

bool intPlane( in vec4 pl, in vec3 ro, in vec3 rd, in float tm, out float t ) {
t = -(dot(pl.xyz,ro)+pl.w)/dot(pl.xyz,rd); return (t > 0.0) && (t < tm); }

vec3 calcnor(in vec4 ob,in vec4 ot,in vec3 po,out vec2 uv ) { vec3 no;

if(ot.w>2.5)
{
    no.xz = po.xz-ob.xz;
    no.y = 0.0;
    no = no/ob.w;
    uv = vec2(no.x,po.y+fpar00[12].w);
}
else if(ot.w>1.5)
{
    no = ob.xyz;
    uv = po.xz*.2;
}
else
{
    no = po-ob.xyz;
    no = no/ob.w;
    uv = no.xy;
}

return no;

}

float calcinter(in vec3 ro,in vec3 rd,out vec4 ob,out vec4 co) { float
tm=10000.0; float t;

if( intSphere(  fpar00[0],ro,rd,tm,t) ) { ob = fpar00[0]; co = fpar00[ 6]; tm = t; }
if( intSphere(  fpar00[1],ro,rd,tm,t) ) { ob = fpar00[1]; co = fpar00[ 7]; tm = t; }
if( intCylinder(fpar00[2],ro,rd,tm,t) ) { ob = fpar00[2]; co = fpar00[ 8]; tm = t; }
if( intCylinder(fpar00[3],ro,rd,tm,t) ) { ob = fpar00[3]; co = fpar00[ 9]; tm = t; }
if( intPlane(   fpar00[4],ro,rd,tm,t) ) { ob = fpar00[4]; co = fpar00[10]; tm = t; }
if( intPlane(   fpar00[5],ro,rd,tm,t) ) { ob = fpar00[5]; co = fpar00[11]; tm = t; }

return tm;

}

bool calcshadow(in vec3 ro,in vec3 rd,in float l) { float t;

bvec4 ss = bvec4(intSphere(  fpar00[0],ro,rd,l,t),
                 intSphere(  fpar00[1],ro,rd,l,t),
                 intCylinder(fpar00[2],ro,rd,l,t),
                 intCylinder(fpar00[3],ro,rd,l,t));
return any(ss);

}

vec4 calcshade(in vec3 po,in vec4 ob,in vec4 co,in vec3 rd,out vec4 re) { float
di,sp; vec2 uv; vec3 no; vec4 lz;

lz.xyz = vec3(0.0,1.5,-3.0) - po;
lz.w = length(lz.xyz);
lz.xyz = lz.xyz/lz.w;

no = calcnor(ob,co,po,uv);

di = dot(no,lz.xyz);
re.xyz = reflect(rd,no);
sp = dot(re.xyz,lz.xyz);
sp = max(sp,0.0);
sp = sp*sp;
sp = sp*sp;

if( calcshadow(po,lz.xyz,lz.w) )
    di = 0.0;

di = max(di,0.0);
co = co*texture2D(tex00,uv)*(vec4(.21,.28,.3,1) + .5*vec4(1.0,.9,.65,1.0)*di) + sp;

di = dot(no,-rd);
re.w = di;
di = 1.0-di*di;
di = di*di;

return(co+0.6*vec4(di));

}

void main( void ) { float tm; vec4 ob,co,co2,re,re2; vec3 no,po; vec3 ro =
fpar00[12].xyz; vec3 rd = normalize(raydir);

tm = calcinter(ro,rd,ob,co);

po = ro + rd*tm;
co = calcshade(po,ob,co,rd,re);

tm = calcinter(po,re.xyz,ob,co2);
po += re.xyz*tm;
co2 = calcshade(po,ob,co2,re.xyz,re2);
gl_FragColor=mix(co,co2,.5-.5*re.w) + vec4(fpar00[13].w);

};

Vertex shader:

varying vec3 raydir; uniform vec4 fpar00[16];

void main( void ) { vec3 r;

gl_Position=gl_Vertex;

r = gl_Vertex.xyz*vec3(1.3333,1.0,0.0)+vec3(0.0,0.0,-1.0);

raydir.x=dot(r,fpar00[13].xyz);
raydir.y=dot(r,fpar00[14].xyz);
raydir.z=dot(r,fpar00[15].xyz);

};

inigo quilez - learning computer graphics since 1994

Inigo Quilez :: articles :: volumetric sort - 2006

Intro Let's say you have a large set of small objects that you want to alpha
blend in a front to back or back to front order. Let's say the position of these
objects is constant. And let's also assume that your objects are all positioned
along a 2d or 3d grid, or close to a grid. This can be the case when rendering a
point cloud of gaussian splats for example, as we usually do for laser scanned
data. Then, you can very easily sort this objects at virtually zero performance
cost, and this article will explain you how.

It might look like the premises are too restrictive, not applicable to "real
life". However, imagine you have a field of grass, and you want to draw the
blades in back-to-front order. You can probably afford aligning them into a 2D
grid, and might be apply random scale and orientation to break the regularity.
Or may be you have a cloud rendering engine using billboards, and you have to
sort them also to properly alpha blend the particles. Or you could even have a
point-cloud viewer showing some nice Julia sets coming from a 1024^3 voxel (like
the one in the end of this article).

In 2D To explain the technique, let's first think about the problem in 2D. Let's
say you have a grid of objects like the one below. Now let's say you are looking
at this grid of objects from the view point indicated by the orange arrow in the
diagram. Now, try to agree with the fact that given this situation you could
draw your objects line by line, starting from the line a, b-... until the line
p, q-...

That's obviously because we are looking roughly from bottom to top. Note also
that because we are a bit skewed to the left, we better draw the objects from
"left to right" within each line; ie, a, c, d, ... instead of ..., e, d, c, b,
a.

Good. In a very similar way, the correct order to render the objects in the grid
for a viewpoint like the one indicated with the green arrow should be left to
right for the columns, and then bottom to top within each column: ..., p, k, f,
a, ..., q, l, g, b, ...

So it's quite simple. We can determine the order just based on the view vector.
If instead of "left to right" and "bottom to up" we use +x, -x, +y and -y, we
can easily see that there 8 different possible orders: { +x+y, +x-y, -x+y, +x-y,
+y+x, +y-x, -y+x, -y-x } (basically we have 4 options for the first axis (+x,
-x, +y, -y) and the there is only 2 remaining for the second (+x, -x or +y, -y,
depending on the first option).

The transition between one order and another is done on half quadrants, as you
can see. The figure showing 2D square split in 8 sections shows the areas where
the same order is valid (you can see the orange and green areas corresponding to
the arrows we used as example in the previous diagram). The trick now is clear:
precompute 8 index arrays and save the in video memory, one for each possible
order. For each rendering pass, take the view direction and compute wich of the
orders is the good one, and use it to render. So, we basically skip any sorting
time, and also bus trafic between the CPU and the GPU. The only drawback is that
we need 8 copies of the index arrays in memory instead of 8, and as we will see
immediately, it is even worse in 3D... But again, is so cool to have zero
sorting cost!

When the view vector stays in any point of a colored area, the order is the
same.

In 3D In 3D the situation is quite the same, we only have one axis more. The
difference is that now the amount of possible orders is quite bigger. For the
first axis's order we have 6 options (-x,+x,-y,+y,-z,+z), for the second we
have 4 (assume we chose -x, the we still have -y,+y,-z,+z) and 2 for the last
axis (assuming we chose +z, we still have -y and +y). So that's a total of 48
possibilities! This can be a lot of video memory depending on the application.
There is some simple tricks to help of course. For example, we keep the 48
copies in system memory and just upload the one we need. Assuming frame to frame
coherence, this should happen not to often. We can even have a small thread
running in parallel to the rendering just calculating the index array instead of
precalculating and storing it in system memory. We can even anticipate the
camera movement and precompute (asynchronously) the next expected index array.

Another trick is to have a top level grid to sort cells of objects, and then let
random ordered drawing of the objects in the cell. If the objects were a field
of grass, this can work pretty well. Or even, if we already have an octree data
structure to do frustum culling and occlusion queries on the dataset, we can
sort the octree nodes with this technique and then do standard CPU sorting in
the visible node, or even have precomputed index arrays per-node.

Now the view vector can belong to 48 possible sections in the surface of a cube,
as shown in the picture below.

In 3D, we have 48 areas for the view vector.

Implementation To finish the article, a bit of code to show how you can get the
order index (from 0 to 47) from the 3D view vector. There is probably a more
simple (read compact) way to do it.

int calcOrder( const vec3 & dir ) { int signs;

const int   sx = dir.x<0.0f;
const int   sy = dir.y<0.0f;
const int   sz = dir.z<0.0f;
const float ax = fabsf( dir.x );
const float ay = fabsf( dir.y );
const float az = fabsf( dir.z );

if( ax>ay && ax>az )
{
    if( ay>az ) signs = 0 + ((sx<<2)|(sy<<1)|sz);
    else        signs = 8 + ((sx<<2)|(sz<<1)|sy);
}
else if( ay>az )
{
    if( ax>az ) signs = 16 + ((sy<<2)|(sx<<1)|sz);
    else        signs = 24 + ((sy<<2)|(sz<<1)|sx);
}
else
{
    if( ax>ay ) signs = 32 + ((sz<<2)|(sx<<1)|sy);
    else        signs = 40 + ((sz<<2)|(sy<<1)|sx);
}

return signs;

}

Examples These are a few images rendered in realtime with gaussian splatting,
using the technique presented in this article for sorting the splats for correct
blending. I ran these experiments in 2005, and all splat positions were
precalculated and, and stored in large static vertex buffers. There were 48
static index buffers precomputed as well and the right one was chosen at run
time based on view angle, as explained in this article.

3D Noise volume

Julia set

3D Noise

SDF primitives

inigo quilez - learning computer graphics since 1994

Inigo Quilez :: articles :: voxel lines and occlusion - 2013

Intro Voxels are fun. There's something mesmerizing about their appearance, so
it's natural that people use them a lot in their demoscene production, Shadertoy
creations and even in their visual effects demo reels. Their XYZ plane aligned
geometry makes for a good candidate for all sort of rendering optimizations. And
also makes it kind of easy to perform some operations that would otherwise be
difficult in other contexts. Among such things, there's the computation of shape
edges and occupancy.

Voxel edges computed on the fly per pixel (see
https://www.shadertoy.com/view/4dfGzs)

Edges So, you have your voxel volume which is full of either empty or filled
cells/voxels, with values 0 or 1 representing the state. You are now shading the
face of a voxel cell and you want to mark the pixels belonging to the edges of
the voxel shape as such. I will assume that:

you know if this face is point up, down, left, right, front or back you
therefore can access the neighbor voxels around this face in "local space" you
have a vec4 called va with the values of the right, left, front and back side
voxels you have a vec4 called vb with the values of the front right, front left,
back left and back right corner voxels you have a vec4 called vc with the values
of the right, left, front and back of the voxel above the current you have a
vec4 called vd with the values of the front right, front left, back left and
back right corner voxels of the cell above the current you have normalized uv
parametrization for the face Note that va, vb, vc and vd capture the information
of the neighbors in "local" space. You can see these cells represented in the
figure to the right. The blue grid cell is the current cell undergoing shading.
The cell above it (in local space) is naturally guaranteed to be empty if we are
to see this cell at all from our current camera's point of view (that is the
cell the ray intersecting this voxel cell arrived from).

Labeling of the voxel cells involved (in local space)

Now that we have all this information, we can tag the pixels that belong to the
edges (within some range in UV space, like 0.85 to 0.95 for example) as actual
edges for the voxel solid by looking at the neighbor voxel grid cells stored in
va, vb, vc and vd. For example, any of the edges of the blue cell will be an
actual geometric edge if the cell to that side va is empty. It will also be an
edge if va is not empty but vc, the voxel to the side and above, is also not
empty. So, for the right edge of our voxel face we have:

float rightEdge = smoothstep( 0.85, 0.95, uv.x) * ( ((1.0-va.x)||(vc.x))?1.0:0.0
);

We can perform the OR operation above directly with floating point signals by
using or(a,b) = a + b - a*b:

float rightEdge = smoothstep( 0.85, 0.95, uv.x) * (1.0-va.x + va.x*vc.x));

We can also do this for the whole four sides of the face at once, if only to
bring clarity to the code:

float edges = maxcomp( smoothstep( 0.85, 0.95, vec4(uv.x,1.0-uv.x,uv.y,1.0-uv.y)
) * (1.0-vc*(1.0-va)) );

where maxcomp() returns the largest component of a vector.

This code will work, except that due to the thickness of the edge at rendering
time we also need to take care of the corners. We can proceed similarly with the
corners though, and flag a corner as belonging to an edge if the vb cell is
empty or the vd cell is solid. Putting both edge and corner detectors together
produces the following code:

float isEdge( vec2 uv, vec4 va, vec4 vb, vec4 vc, vec4 vd ) { // float maxcomp(
in vec4 v ) { return max( max(v.x,v.y), max(v.z,v.w) ); }

vec2 st = 1.0 - uv;

// sides    
vec4 wb = smoothstep( 0.85, 0.95, vec4(uv.x,
                                       st.x,
                                       uv.y,
                                       st.y) ) * (1.0 - va + va*vc) );
// corners
vec4 wc = smoothstep( 0.85, 0.95, vec4(uv.x*uv.y,
                                       st.x*uv.y,
                                       st.x*st.y,
                                       uv.x*st.y) ) * (1.0 - vb + vd*vb);
return maxcomp( max(wb,wc) );

}

Fake Occlusion A very similar technique can be used to compute fake occlusion on
the face of a voxel from the immediate neighbor voxels. In this case we only
care about the cells above the current cell undergoing shading, vc and vd. For
the edges, we can simply use a linear grayscale UV gradient modulated by the
neighbor cell occupancy. Or in other words, if the voxel is solid, it produces
occlusion. Something like

float rightOcclusion = uv.x * vc.x;

For corners, we want to occlude when the corner cell vd is solid but the edge
cells vc are empty, since the case of solid vc cells has already been captured
with the previous test. For example, for the front right corner, we'd have

float frontRightOcclusion = uv.x * uv.y * vd.x * (1.0-vc.x)*(1.0-vc.z);

We can put it all together to get

float calcOcc( vec2 uv, vec4 va, vec4 vb, vec4 vc, vec4 vd ) { vec2 st = 1.0 -
uv;

// edges
vec4 wa = vec4( uv.x, st.x, uv.y, st.y ) * vc;

// corners
vec4 wb = vec4(uv.x*uv.y,
               st.x*uv.y,
               st.x*st.y,
               uv.x*st.y)*vd*(1.0-vc.xzyw)*(1.0-vc.zywx);

return wa.x + wa.y + wa.z + wa.w +
       wb.x + wb.y + wb.z + wb.w;

}

which produces kind of decent cheap approximation to short distance occlusion.
See the images comparing some simple rendering of a voxel without and with the
occlusion approximation enabled:

Ambient + Diffuse lighting

Same as left, with fake occlusion based on neighbor occupancy

There's a live example in Shadertoy of this code (click play to watch it move,
or follow https://www.shadertoy.com/view/4dfGzs)

inigo quilez - learning computer graphics since 1994

Inigo Quilez :: articles :: 3D models generation - 2005

When making a 4 kilobyte or a 64 kilobyte demo one needs to be imaginative in
the way geometry is stored. In my experience, in a 4 kilobyte demo as
Kinderplomber or Stiletto one cannot affort more than one kilobyte of geometry.
In 64 kilobytes demos I guess something like twelve kilobytes is ok (as
in 195/95/256 or Paradise. Of course, you can also use full procedural geometry
and spend just few hundred bytes to build your scenes. You can also combine pure
geometry with procedural techniques, and have the best of both worlds. So, let's
quickly have an overview of some of the techniques I used with my colleagues at
rgba for some of our demos.

Primitives

The simplest way to display some geometry is to use simple primitives. Like
cubes, spheres, cylinders, toruses, planes. Many famous and successfull 4
kilobyte demos are done this way (Atrium and Micropolis being probably the best
examples of this practice. Most 3d APIs like OpenGL or DirectX will give you
some interface to render such primitives for free, but in any case, generating
them is trivial and small.

Personally I have never being in favour of these methodes. It's boring, looks
simple and has zero challenge. Also, in our productions, even if it seems we
have being using them like the Image 1 taken from 195/95/256 might sugest, we
have never done so. That scene shows a cube, a sphere, a plane, a cylinder and a
cone, yet there is no code in the demo to generate such primitives. There are
other techniques that can be used to generate such shapes without writing one
single line of extra code or using any external DLL or API. For example,
revolution shapes.

Revolution shapes

Since the very beggining ot my 64 kilobyte demo making experience, like with
Weektro and rare back in 1998, I have always used revolution shapes to generate
geometry. Revolving geometry is a very cheap way to generate interesting
geometry: given a 2d contour made of line segments, a surface is constructed as
the 2d profile turns around a given axis. The only thing you need to store is
few 2d coordinates for your profile (probably stored as 8 bit numbers) and then
a little piece of code to generate triangles based on that data and few sinus
and cosinus (around 300 bytes of code). The technique is quite flexible, it
allows to build many types of objects, litle columns, cups, mushrooms,
tentacles, skydomes... Also, of course, cylinders, cubes and spheres.... So,
that's why I never write one single byte of code to generate those primitives -
if you have revolution shapes in your geometry engine, you have those ones too.

Clone, scale, rotate, bend

No need to say that you can get variations of your geometry by instanciating it
many times (cloning) and changing the scale, orientation and bending of the
clones. For example you can use procedural techinques to do so and build trees
or cities, or you can even use more sophisticated algorithms to build something
like Atrium or even manual placement of clones to shape a robot.

Displacement

However the technique I like the most is displacement of geometry. This very
easily hides the simplicity of your data, and makes it look rich. We used it a
lot both in Paradise and 195/95/256. The idea is, once you have a 3d object, say
a revolution shape, then process it's vertices one by one and apply some formula
to it's vertex coordinates and move them. That way you can achieve scale,
rotation and bending for free without any extra code, but also much more complex
effects like fractal detail growth, ripples, bumps, etc. For example, in
Paradise there are some mountains in the background of one of the scene. Those
where not done with some special code for fractal mountains, but it was just a
displacement modifier applied to a 2d horizontal plane. Another example are the
columns shown on Image 3, taken from 195/95/256, where the details of the
columns where doing by displacement. Another example is visible in Image 1 - the
potatoe in the background is a revolution object (a sphere) where the vertices
where displaced with some perlin noise. Image 4 shows another use of
displacement, in this case the teeths of the perimeter of the wheel where grown
by displacement of vertices too.

Subdivision, softeing

Of course there are algorithms to tesselate geometry and smooth it. For example,
we have used very often the well knwon Catmul Clark subdivision shceme to make
surfaces more detailed and smooth. Catmul Clark works best with quadrilaterals;
for pure triangular meshes you can try the Loop subdivision shema, although I
really recommend Catmul Clark since most artists are used to it (that's what
most modeling applications use today).

Also you can smooth a surface without generating more polygons. You can do that
by low pass filtering the mesh, or in other words, doing some sort of bluring on
the vertices. You can see it like this: for a given vertex, find the vertices
that are conected to it (through an edge), and do some averaging on vertex
coordinates, and asign it to the coordinates of the current vertex. Just like
you blur images.

Mesh compression

Of course, you can also make your 3d models in any 3d application and then
export them to your demo. We have used this a lot in the last 4 and 64 kilobyte
demos. Basically you normally want to store a low resolution version of you
mesh, with as less polygons as possible, and store it as a C array of data in
your demo. Usually vertices are quantified to 8 bits for example, and geometry
converted to quad strips for compression. Well, there are many techniques to
compress, other articles and presentations in this site explain the techniques
in detail. You can easily get down to an average of 2 bytes per triangle in your
models. For example, each of the statues in the 195/95/256 demo, like the one
shown in Image 7, was compressed in around 750 bytes.

Metaballs

Another way to build smooth surfaces, although it's very difficult to control,
are metaballs. I still have not seen a single demo where metaball based objects
look anything near "not horrible", but hopefully somebody will improve the
technique one day or another. The advantage is that once you place the
metaprimitives (balls, cylinders, planes, whatever) either by manual placement
by an artist or by procedural techniques (like ifs, fractals, or any other), you
get a smooth surface without cracks. You can marching cubes or marhching
tethaedra for that, the only problem is you will probably need some
simplification algorithm to reduce the huge amount of polygons that you will
generate.

Image 1. A shoot from 195/95/256 showing some "primitives"...

Image 2. A shoot from 195/95/256 showing revolution shapes

Image 3. Displacement applied to the geometry of Image 2

Image 4. Displacement used to grow some teeth in a wheel

Image 5. A mesh before and after CC subdivision

Image 6. A compressed mesh

Image 7. A compressed mesh after CC subdivision

inigo quilez - learning computer graphics since 1994

Inigo Quilez :: articles :: minimal code for splines - 2003

This is an coding trick I developed for the Paradise 64k demo in 2004. Before
Paradise, all the camera movents of our 4k and 64k intros had been done with
simple combinations of sinus and cosinus functions. However we wanted more
camera control for Paradise, so we decided to implement some minimal spline
functionality. We decided that Catmull-Rom splines, even if they don't provide
tangent controls, would suffice for our little production. So we went for it and
we ended up with a cute tiny implementation that allowed us to interpolate
both 3D and n-dimensional control points.

Catmull-Rom splines The beauty of Catmull-Rom splines over Bezier or other types
of Splines and other cubic polynomial interpolation functions in general is that
with this type of spline control over the shape of the curve is super simple and
can't be more intuitive. Simply choose a few point in space, a time for every
one of them, et voilá, you are done, your path will pass thru those points at
those moments in time. Can't be more easy.

I'm not going to explain how Catmull-Rom splines are constructed, you can simply
go to google for that, or derive the formulas yourself, it's very easy because
you simply have to:

1.  build a generic cubic polynomial p(t) = a + b⋅t + c⋅t2 + d⋅t3, where p(t) is
    your path in space and a,b,c,d are coefficients (points in space too)
2.  take it's derivative p'(t) = b + 2c⋅t + 3d⋅t2
3.  make sure that at t=0 the curve passes thru your first point p1 (you got the
    first coefficient already, a = p1)
4.  make sure that at t=0 the curve has tangent (p2-p0)/2 (and you got the
    second coefficient, b = (p2-p0)/2)
5.  make sure that at t=1 the curve passes thru the second point p2
6.  make sure that at t=1 the curve has tangent (p3-p1)/2
7.  solve 5 and 6 to get c and d

If you follow these steps you will arrive to something like:

a = (2⋅p1) b = (p2-p0) c = (2⋅p0 - 5⋅p1 + 4⋅p2 - p3) d = (-p0 + 3⋅p1 - 3⋅p2 +
p3)

or if you want as

a = < { 0, 2, 0, 0}, {p0,p1,p2,p3} > b = < {-1, 0, 1, 0}, {p0,p1,p2,p3} > c = <
{ 2,-5, 4,-1}, {p0,p1,p2,p3} > d = < {-1, 3,-3, 1}, {p0,p1,p2,p3} >

Mathematicians will tell me you cannot dot a vector with a vector of vectors,
but well, you get the idea.

Of course one first needs to know the segment (p1,p2) in which we want to
perform the cubic interpolation above. Then, the formula for p(t) have to be
used after normalizing t to the proper (t1,t2) interval.

The code This code is written using standard C types only, so it should be
pretty much CopyPaste-able, and easily adapted to your needs. There might be
some corner cases you might want to add checks for, like the cases where t is
less than zero or bigger than the time of the last key of the spline.

static signed char coefs[16] = { -1, 2,-1, 0, 3,-5, 0, 2, -3, 4, 1, 0,
1,-1, 0, 0 };

void spline( const float *key, int num, int dim, float t, float v ) { const int
size = dim + 1; // find key int k = 0; while( key[ksize]<t ) k++; // interpolant
const float h = (t-key[(k-1)size])/(key[ksize]-key[(k-1)size]); // init result
for( int i=0; i<dim; i++ ) v[i] = 0.0f; // add basis functions for( int i=0;
i<4; i++ ) { int kn = k+i-2; if( kn<0 ) kn=0; else if( kn>(num-1) ) kn=num-1;
const signed char co = coefs + 4i; const float b = 0.5f(((co[0]*h + co[1])h +
co[2])h + co[3]); for( int j=0; j<dim; j++ ) v[j] += bkey[knsize+j+1]; } }

This might compile to something between 100 or 150 bytes depending on your
compiler settings. It can probably be optimized by computing k*size only once
and reusing it in the rest of the code, and using offsets to index in the kn
keys, but I don't want to make the code that unreadable here, although I would
certainly do it if I was using this for a 4k intro.

As you can guess from the code, the format for the path key is: { t0, x1, y1,
z1, t1, x1, y1, z1, t2, ,x2, y2, z2, ... }, that is not the best option
regarding data compression, but works allows a small implementation of the
spline code. You probably want to store the splines in a more compressible
format (like independant streams for the different t, x, y, z signals, and then
apply some linear prediction on them and then quantify them for example, or any
other thing you might come with) and then convert it to this format before
performing the spline evaluation.

inigo quilez - learning computer graphics since 1994

Inigo Quilez :: articles :: raymarching distance fields - 2008

Raymarching SDFs (Signed Distance Fields) is slowly getting popular, because
it's a simple, elegant and powerful way to represent 3D objects and even
render 3D scenes. SDFs are a type of implicit surface representation, and so its
origins can be tracked back very far in the history of computer graphics. While
not SDFs per-se, the first paper that's been brought to my attention showing
boolean operations in implicit surfaces by combining them through the min() and
max() operators dates back to 1972, by A.Ricci and later by B.Wyvill and
G.Wyvill in 1989. The first mention of raymarching SDFs seems to be also from
that time - the paper by Sandin, Hart and Kauffman, which used it for
rendering 3D fractals, although they never coined the term "SDF". Seven years
later in 1995 C.Hart himself documented the raymarching technique once again
(but unfortunately miscalled it "Sphere Tracing" - usually we trace rays,
cylinders or cones, but definitely not spheres!).

My story with SDFs starts around 2001. Like many other demosceners of that time,
I experimented a lot with raymarching Signed Fields (without the "D") to render
fractals and metaballs as part of our demos. However it was only after I read
Alex Evan's work from 2006 and Keenan Crane's work from 2005 that I got excited
about exploring the consequences of constraining a Signed Field to be an
Euclidean Distance. And so, in 2007 I dived in, and explored, and then produced
the first results that we'd recognizable as modern SDF, including the four
images at the bottom of this page that demonstrated non-trivial but art directed
content, and demonstrated techniques for efficient soft shadows, smooth blending
and domain repetition.

A raymarched procedural SDF, step by step modeling, shading and lighting

Now, although raymarching SDFs can definitely render regular polygonal meshes,
as a lover of compression and procedural content myself, my focus became to
implement the SDFs directly as list of primitives rather than as volumetric
data, which is where many folks' first instinct seem to go, despite the
unresolvable precision issues, probably because their mindset is that of
rendering in video games. However, primitive based SDF offer the infinite
sharpness and precision that is crucial both for CAD and professional graphic
design. Unlike volume based approaches of constant access time, rendering
unlimited number of primitives and their CSG (boolean) operations that define
the SDF requires some acceleration structures and more complex techniques than
volumes. But coincidentally, back in 2007, the timing for such a procedural/math
and compute based approach was great - First, GPUs were evolving their
compute/ALU capabilities faster than their memory bandwidth, a trend that has
only continued in the last 20 years, so pure mathematical SDFs started to be
competitive against a backed representations. Second, I personal had an interest
in doodling around and finding the required maths to express different shapes
and effects. Third, I was also very much interested in improving my artistic
skills, so procedural content was a great fit. And fourth, as a demoscener, I
was naturally captivated by procedural content generation and its potential for
reduced memory footprint (RAM and disk). As it turns, pretty rendering,
procedural content and small memory footprint was also in high demand at Pixar,
I learnt a couple of years later, but that's a story for another day.

And so, for the next 5 years, the demoscene and the Shadertoy communities we
pushed the SDF raymarching technique further, creating more and better looking
content. With each new demo and Shadertoy piece, we proved SDFs did not have to
be limited to just funny fractals or metaballs or toy cities of ever repeating
buildings, but content could be created that was art directed, and included
sophisticated modeling, proper shading with filtering, lighting and animation.
As the volume of works grew, a case started to emerge for SDFs as a legit
professional representation for 3D graphics, at least in certain applications.
And about 5 years later academia noticed this, and started working on the area
as well. Another 5 years or so later the industry itself also joined in, and we
started seeing the emergence of professional modeling and design tools based on
SDFs.

Anyways, to end this already long post, since 1998 I've been documenting some of
the discoveries/inventions I did around SDFs, or just tricks or simple
observations that could be interesting to others, both in this very website and
my Youtube channel. So here are some pointers if you want to start learning the
basics of SDFs: Articles on Raymarching on this site My video tutorials and
streams about raymarching SDFs on Youtube The lecture called "Rendering Worlds
with Two Triangles" that I have 2008 on the topic All my Shadertoy examples,
some of which I list below:

Sea Creature, 2022

This was a quick one-nighter. It uses a classic KIFS recursive fractal schema
exactly like the Angels painting below, but here I used smooth-minimum and
smooth-abs to blend all spheres together into a single organic shape. The rest
is color tweaking. The shader runs slow because I'm doing volumetric rendering
so I get some sweet transparencies( as usual I'm okey with doing something 4x
slower if that makes it look 10% better!)

Source code: https://www.shadertoy.com/view/csB3zy

Selfie Girl, 2020

This was my second human SDF and my first attempt at facial animation. When I
finished it I was as proud of it as I was critical of the many flaws it had. But
of course I think it's still a great example of using SDFs for organic modeling.
I think the whole primitive count for the whole painting is 32, so that gives
you a sense on how little data is needed to represent 3D scenes and how
efficiently it can be stored and processed.

Source code: https://www.shadertoy.com/view/WsSBzh Tutorial:
https://www.youtube.com/watch?v=8--5LwHRhjk

Sphere Gears, 2019

This was an improv session, based on some other Shadertoy user's idea, which I
started as an optimization exercise. In particular, thanks to the symmetry of
the object you can evaluate only 4 gear pieces instead of the 18 that make the
whole object. Similarly, within each gear, only a single tooth/dent is evaluated
instead of 12, making the whole SDF pretty cheap to evaluate.

Source code: https://www.shadertoy.com/view/tt2XzG Tutorial (part 1):
https://www.youtube.com/watch?v=sl9x19EnKng Tutorial (part 2):
https://www.youtube.com/watch?v=bdICU2uvOdU

Happy Jumping, 2019

Happy Jumping was a one-day effort to animate a character, which I had never
done before really. I designed a super simple character for it and the animation
was rough - in fact, it gets out of model very often. But it was a fine first
attempt. The lighting is my usual 3 light rig (key, fill, bounce) and some fake
subsurface.

Source code: https://www.shadertoy.com/view/3lsSzf Tutorial:
https://www.youtube.com/watch?v=Cfe5UQ-1L9Q

Planet Fall, 2018

Planet Fall was an improv/jam session, although it turned out pretty well. It's
a level recursion of voronoi distributed cylinders on a sphere. The lighting is
cheated as usual and the occlusion mostly painted by hand. But the color palette
makes it look so pretty.

Source code: https://www.shadertoy.com/view/lltBWB

Surfer Boy, 2018

This was my first attempt at creating a human face completely
procedurally/mathematically. It's made mostly of ellipsoids, cones and quadratic
bezier curves. It was mostly modeled to camera - it doesn't look that good from
other angles. In order to keep it realtime I did a pretty simple lighting setup
(no ambient occlusion, no GI, just painted colors here and there).

Source code: https://www.shadertoy.com/view/ldd3DX

Greek Temple, 2017

This image was born as a live coding improv session for the students of UPenn,
although I spend a couple of days working after the live coding was done. The
temple is made of basic domain repetition of 6 or 7 boxes and cylinders. The
most interesting bit is probably the lack of global illumination - instead, the
rich bounce lighting in the temple is achieved by colorzing the relevant areas
by hand positioned colors. Also, for the sake of an interesting composition, the
light direction that illuminates the ocean and terrain is different from the
direction of the light that illuminates the temple.

Source code: https://www.shadertoy.com/view/ldScDh Tutorial:
https://www.youtube.com/watch?v=-pdSjBPH3zM

Ladybug, 2017

I made this one after another live coding session for the students of the
University of Washington. I made a mushroom for them, and then the two weeks
that followed, in small chuncks of night time, I slowly completed the drawings
until I got here. The modeling is weak and I only pushed it enough to sell the
image at medium resolution. It could get a lot more love, but I had to stop
somewhere before I'd take too many nights. Lighting is mostly faked (no ambient
occlusion, no GI) and very much painted by hand to look good (especially
everything that has to do with bounce lighting and subsurface looking
materials).

Source code: https://www.shadertoy.com/view/4tByz3

Fractal Cave, 2016

The was a menger sponge fractal with some variations and a heavy domain
distortion. The image was path-traced by raymarching the signed distance field.

Source code: https://www.shadertoy.com/view/Xtt3Wn

Rainforest - 2016

This was an attempt to raymarch a signed distance field of a tree-populated
terrain. Trees are mere spheres with some basic noise displacement distributed
on a voronoi pattern. The most interesting part is the heavy use of analytical
derivatives of noise to compute fast normals and slopes, which helped a lot
speed up the raymarching process of both terrain (the bigger the slope, the
smaller the step size - you can read on the Lipschitz constant).

Tutorial: https://www.youtube.com/watch?v=BFld4EBO2RE Source code:
https://www.shadertoy.com/view/4ttSWf

Elephants - 2016

Since the Snail worked so well, I tried to make elephants too, with the same
exact techniques. This proved to be more difficult, and the results are only
mediocre. In particular, the modeling was difficult to make for me by just
typing formulas, so I left it unfinished and only added detail that would be
seen from this camera angle. The trees are ellipsoids with some domain
distortion, and the terrain a distance field to a plane with displacement.

Source code: https://www.shadertoy.com/view/4dKGWm

Snail, 2015

The Snail was one of my all time favorite images, because it looks good and
tells a little story. It was mostly a modeling exercise, by using simple
distance functions blended together smoothly. The shell was interesting to make,
and the translucent feeling of the snail itself took me a while to get right.

Source code: https://www.shadertoy.com/view/ld3Gz2

Sculpture III, 2015

Just like Sculpture II below, this was an (domain) deformed sphere through here.
The distortion was made with four octaves of sine waves.

Source code: https://www.shadertoy.com/view/XtjSDK

Arlo - 2015

This was mostly a procedural modeling exercise with signed distance field. By
using simple distance functions blended together smoothly, I was able to
replicate the main character of Pixar's movie The Good Dinosaur. I didn't have
time to pose it in a more interesting action, so he's looking forward and stiff.
The most interesting part in this drawing was the use of ray differentials to do
proper texture filtering in a raymarching context.

Source code: https://www.shadertoy.com/view/4dtGWM

Eve arrives, 2015

Eve was my first attempt at modeling an actual character. And I chose her
because she is simple - she's made of basic ellipsoids and sphere that are
smooth blended together. The background detail is done with domain repetition of
one box.

Source code: https://www.shadertoy.com/view/llsXRX

Mushroom - 2015

For this drawing I tried to create rocks by a technique that I saw the user TDM
use for one of his shaders: start with a sphere and cut bigger spheres of it to
create a semi sharp rock. It worked more or less fine, I got some interesting
rocks.

Source code: https://www.shadertoy.com/view/4tBXR1

Antialias, sort of - 2014

This was an attempt to antialiasing the screen-space edges of the raymarched
distance fields. By using ray differentials, one can estimate how much of the
pixel footprint (pixel frustum) does approximately intersect the geometry, given
that at each marching step we heave the distance to the closest surface. That
was used in this image to compute partial coverage per near-intersection, and
them they were all composited front to back to give the correct smooth image
without edge pixelization.

Source code: https://www.shadertoy.com/view/llXGR4.

Sculpture II - 2014

This was another quick experiment on domain distortion through here. In this
case, a sphere got distorted by a sine wave of a sine wave of a sin wave of a
sin wave... of a sin wave. The result: a beautiful shape.

Source code: https://www.shadertoy.com/view/4ssSRX

Canyon - 2014

A simple terrain in this case. Some bits are floating in the air since it was
defined as a true 3D field, not as a height-map based distance field. Adding
white snow on a red/yellow terrain always works and it's a cheap way to get good
results without doing much effort on actual shading/texturing/surfacing.

Source code: https://www.shadertoy.com/view/MdBGzG

Worms - 2014

This one was improvised over a night, and it featured some domain repetition
with only three cylinders modeled, and domain distortion to make them curve
along the vertical axis.

Source code: https://www.shadertoy.com/view/XsjXR1

Fish Swimming - 2014

The interesting aspect of this signed distance field image is the way the
texturing was done. Since there are no meshes or vertices for the modeling of
the fish, there is no place to store UV coordinates that can deform with the
model itself. But I was able to make the animation function of the fish
invertible such that each point on the surface would be able to know where it
came from in the rest pose of the fish and do the noise and pattern lookups
there. That way the textures didn't swim on the surface of the fish as they did
in the Insect drawing below. The fish did swim actually, with some simple sine
waves and noise.

Source code: https://www.shadertoy.com/view/ldj3Dm

Dolphin - 2014

This was my attempt at doing water splashes. That was done by doing some
displacement of the water plane with an exponential shape based on the proximity
of the dolphin to the water. Indeed, the signed distance field representation
can be used beyond finding marching and shadow or occlusion computation - in
this case helps the water know about the dolphin. The dolphin itself was made
with a few cylinders and some other smoothly blended primitives.

Source: https://www.shadertoy.com/view/4sS3zG

Woods - 2013

In this case a simple camera rotation was enough to hide the domain repetition
of the trees. The interesting part is that I had to fake the lighting to the
point of bending the light itself along the depth of the drawing: the direction
of the sun light is different in the foreground and in the background
(interpolated between those two poses). Tha allowed me to get the patch of light
in the red mushroom by the camera and the nice side lighting in the tree in the
background. Modeling wise, the trees are cylinders with an exponentially
decaying radius along its axis, all smoothly blended together with the smooth
minimun formula, and some noise on top as a displacement.

Source code: https://www.shadertoy.com/view/XsfGD4

Rounded Voxels - 2013

This was an exercise to integrate together fast ray-casting though a regular
grid and raymarching inside the grid cells. Basically, the ray marches quickly
though a volumetric regular grid, which can be done efficiently with a few
additions and integer mask operations, and when found a grid cell to be not
empty, a raymarcher takes over to find the intersection of the signed distance
field defining the rounded boxes. If no intersection happens, the control is
returned to the grid marcher.

Source code: https://www.shadertoy.com/view/4djGWR

Volcanic - 2013

This was a serious attempt at getting some decent outdoors lighting without
resorting to global illumination. I wrote an article about it here. Modeling was
based mostly on value noise, and there's also some smoke added as a second
raymarching pass.

Source code: https://www.shadertoy.com/view/XsX3RB

Wavy - 2013

This was an attempt to pretty colors. It used some domain deformation through
wrapping like the one I described here, but in 3D and with only one level of
indirection (the rendering is slow enough as it is).

Source code: https://www.shadertoy.com/view/ldX3zS

Angels - 2013

Angels used some basic domain repetition to create many flying creatures despite
I only had one. The differences in the animations between the different
creatures was made by taking the cell id to offset the animation formula (which
was based on cosine waves). The creatures themselves were made with some simple
fractal recursion of ellipsoids.

Source code: https://www.shadertoy.com/view/lssGRM

Insect - 2013

This shader was one of the first to use the smooth minimun formula to blend
shapes together (legs and body). It also performs some analytic inverse
kinematic to position the legs automatically in the terrain without simulation
or integration.

Source code: https://www.shadertoy.com/view/Mss3zM

Piano - 2013

This was an exercise in combining true lighting with painted lighting. By
"painted" I mean that it wasn't computed through shadow casting or pathtracing,
but that its color was injected artificially in specific areas and objects in
the scene to convey ("fake") expensive lighting effects. For example, some of
the light in the bottom part of the wall was added artificially, and the window
from which the light comes in doesn't exist, but was shaped by hand with the
formula of a square. On the other hand, this drawing performed true reflections.

Source code: https://www.shadertoy.com/view/ldl3zN

Rocks - 2013

This drawing was based on a voronoi pattern again, like Leizex (see below), and
was mostly an exercise on shading simple rocks. However it's interesting to see
how now this could be done in realt-time in the GPU while Leizex (2008) wasn't
real time at all!

Source code: https://www.shadertoy.com/view/MsXGzM

Fruxis - 2012

I made this one for the Trsac Demo party. It was my raymarched procedural
distance field that was implemented in the GPU (GLSL). It was also my first to
use a pathtracer (using this algorithm). You can find a modified version of it
that does not use pathtracing here: https://www.shadertoy.com/view/ldl3zl

Cell - 2013

This was my first procedural signed distance field exercise in Shadertoy
(WebGL). The interesting bit was my attempt to fake the subsurface scattering of
light by raymarching inside the volume of the geometry to measure thickness.

Source code here: https://www.shadertoy.com/view/Xdl3R4

Leizex - 2008

This was a one day production. I wanted to check how it is to directly raymarch
a three dimensional voronoi pattern. It did work, although it was very slow
them, but it is realtime this days of course. The lighting is fake again - there
aren't any light sources, neither there is ambient occlusion computations or
anything. Colors are procedurally assigned to points in space.

Source code here: https://www.shadertoy.com/view/XtycD1

Bridge - 2009

This is a sketch really, but it never got completed, due to the lack of artistic
inspiration at the time. The interesting part is probably the technique used for
creating the grass, which is based on voronoi-driven domain tiling.

Organix - 2008

Organix won the 1st position at the Function Demoparty in Hungary. It's a
raymarched SDF, but it run on the CPU in around 40 seconds at 1280x720
resolution. Needles to say this is realtime these days in the GPU. The lighting
is fake in this one, in that shadows and ambient occlusion are procedurally
painted on the geometry, not computer through raycasting or similar techniques.

Source Code: https://www.shadertoy.com/view/ldByDh

Slisesix - 2008

This image was the first image I generated by raymarching SDFs (Signed Distance
Functions). It contained smooth blending of geometric primitives, domain
repetition, ambient occlusion and soft shadows. It won the 1st prize of 4
Kilobytes Procedural Graphics of Euskal Party 2008 in Spain.

Source code here: https://www.shadertoy.com/view/lsf3zr

inigo quilez - learning computer graphics since 1994

Inigo Quilez :: articles :: smooth minimum - 2013

Intro

One of the basic building blocks of Signed Distance Field (SDF) modeling based
on basic primitives (as opposed to Grids or Neural Networks) is the
Smooth-Minimum operator or Smooth-Union. This is similar to a regular
Minimum/Union operator, but, well "smooth". This means that where the regular or
non-smooth Minimum/Union takes two input SDFs a and b and returns the closest
one of the two, effectively combining both into a single field (for the distance
to the combined shape is the distance to whichever piece is closest), the Smooth
version combines the two shapes by blending and melting them together, if they
are close enough. This helps not only aggregate shapes together, but sculpt
organic shapes with them, as if the primitives were made of clay.

Minimum

Smooth-minimum The images above are an example of using the smooth minimum to
sculpt a human face. First we have the regular minimum or union based field,
where all the base primitives are clearly visible. Then we can see the
Smooth-minimum or Smooth-union, where the shapes have blended together to form
one continuous surface. You can see it work in realtime and explore the code in
Shadertoy.

A list of Smooth-minimums

So the regular Minimum simply takes the lowest valued of two SDFs a and b:

float min( float a, float b ) { return (a<b) ? a : b; }

Most languages come with this function already defined, either as a hardware
instruction (most likely) or through a software emulation with an actual branch
like in the code above (most unlikely).

Now, the Smooth-minimum must achieve a smooth, non-binary transition between the
values of a and b for regions where a and b are close enough to each other
within some tolerance k (measured in regular, distance units). So, when
functions or surfaces are close enough to each other, the will blend or melt
together.

We can design an infinite number of such functions, each with different pros and
cons, but here's a list of some such possible smooth-minimum functions that I've
used at one point or another:

// exponential float smin( float a, float b, float k ) { k = 1.0; float r =
exp2(-a/k) + exp2(-b/k); return -klog2(r); } // root float smin( float a, float
b, float k ) { k = 2.0; float x = b-a; return 0.5( a+b-sqrt(xx+kk) ); } //
sigmoid float smin( float a, float b, float k ) { k = log(2.0); float x = b-a;
return a + x/(1.0-exp2(x/k)); } // quadratic polynomial float smin( float a,
float b, float k ) { k = 4.0; float h = max( k-abs(a-b), 0.0 )/k; return
min(a,b) - hhk*(1.0/4.0); } // cubic polynomial float smin( float a, float b,
float k ) { k = 6.0; float h = max( k-abs(a-b), 0.0 )/k; return min(a,b) -
hhhk*(1.0/6.0); } // quartic polynomial float smin( float a, float b, float k )
{ k = 16.0/3.0; float h = max( k-abs(a-b), 0.0 )/k; return min(a,b) -
hhh(4.0-h)k(1.0/16.0); } // circular float smin( float a, float b, float k ) { k
= 1.0/(1.0-sqrt(0.5)); float h = max( k-abs(a-b), 0.0 )/k; return min(a,b) -
k0.5*(1.0+h-sqrt(1.0-h*(h-2.0))); } // circular geometrical float smin( float a,
float b, float k ) { k *= 1.0/(1.0-sqrt(0.5)); return max(k,min(a,b)) -
length(max(k-vec2(a,b),0.0)); } You have the code for the functions above here
in this Shadertoy example.

Now let's just have a look to their behavior first with a(x)=e-x and
b(x)=sin(4x) (quadratic, cubic, quartic, circular, exponential, sigmoid, square
root, circular geometrical):

min(a(x),b(x))

smin(a(x),b(x))

When applied to shapes, they look like this:

Moving shapes blended with constant k

Stationary shapes blended with varying k On the left you can see a moving circle
and a rectangle blended together by the different smooth-minimum functions
defined above with a constant k>, and on the right you see the effect of
modifying the parameter k over two constant shapes. The different colors of the
shapes correspond to the different smooth-minimums above.

Something interesting thing to note is, perhaps, that for any given parameter k,
these smooth-minimum functions produce blending regions of approximately the
same size. This is thanks to the normalization factors I've introduced for k in
each variant. So let's talk about that next.

Normalization, thickness and bounds

If you've been using smooth-minimum functions before in Shadertoy or elsewhere,
you probably have seen some of these smooth-minimum functions above (most likely
the Quadratic one) but without the premultiplication of k that I have introduced
here. This is a "normalization" factor that I've computed so that the parameter
k always maps directly to the thickness of the blended area, in actual distance
units. This equalizes all smooth-minimums and makes them more or less compatible
and interchangeable with each other. You can see this effect in the animations
above, where the neck connecting the circle and the rectangle has constant
thickness regardless of the smooth-minimum in use.

The computation of this normalization factor is easy - you need to consider the
deviation of the smooth-minimum from the minimum when a = b. Later we'll learn
that this is precisely the value of each smooth-minimum's Kernel function,
evaluated at the origin. But for now, I hope it makes sense to you that when
normalized this way, the value of the parameter k matches exactly the maximum
inflation or thickening that the shapes a and b undergo due to the smooth
blending happening. Which means that in this normalized form, the parameter k is
also the bounding box expansion required to perfectly bound the smooth-minimum:

Here the circle a and the rectangle b have been expanded by k units to produce a
combined bounding volume in blue that touches exactly the shape of the
smooth-minimum. Being able to perform this computation is hence key to getting
good hierarchical acceleration structures like BVHs and grids for raycasting,
proximity queries and collision detection.

Mix factor

Besides smoothly blending values, it's often also useful to share the blending
factor to mix the materials of the two SDFs involved. As an example, in the
image below I'm mixing the red and a blue materials based on this blending
factor as computed by the code below, which returns the smooth-minimum in .x and
the blend factor in .y for the Quadratic and the Cubic smooth-minimums. Please
note I have propagated the g(0) normalization already into the expressions, to
get some extra simplifications and regularity. You can derive the blend factors
for the other smooth-minimum functions rather easily too, I leave it to you as
an exercise:

// quadratic polynomial vec2 smin( float a, float b, float k ) { float h = 1.0 -
min( abs(a-b)/(4.0k), 1.0 ); float w = hh; float m = w0.5; float s = wk; return
(a<b) ? vec2(a-s,m) : vec2(b-s,1.0-m); }

// cubic polynomial vec2 smin( float a, float b, float k ) { float h = 1.0 -
min( abs(a-b)/(6.0k), 1.0 ); float w = hhh; float m = w0.5; float s = w*k;
return (a<b) ? vec2(a-s,m) : vec2(b-s,1.0-m); }

Mixing materials with smooth-minimum Code in Shadertoy

The DD Family

Okey, time to look into the smooth-minimum functions more in depth. We'll begin
by recognizing that most of the functions I introduced at the beginning of the
article actually belong to one family that generalizes them. The code above has
been written in more or less optimized GPU code, which obfuscates the
similarities between them, but if we rewrite them a bit, we'll see that the
root, sigmod, quadratic, cubic, quartic and circular variants are very similar
and belong to the same family:

// root float smin( float a, float b, float k ) { k = 2.0; float x = (b-a)/k;
float g = 0.5(x+sqrt(xx+1.0)); return b - k * g; } // sigmoid float smin( float
a, float b, float k ) { k = log(2.0); float x = (b-a)/k; float g =
x/(1.0-exp2(-x)); return b - k * g; } // quadratic polynomial float smin( float
a, float b, float k ) { k = 4.0; float x = (b-a)/k; float g = (x> 1.0) ? x :
(x<-1.0) ? 0.0 : (x(2.0+x)+1.0)/4.0; return b - k * g; } // cubic polynomial
float smin( float a, float b, float k ) { k = 6.0; float x = (b-a)/k; float g =
(x> 1.0) ? x : (x<-1.0) ? 0.0 : (1.0+3.0x(x+1.0)-abs(xxx))/6.0; return b - k *
g; } // quartic polynomial float smin( float a, float b, float k ) { k
\= 16.0/3.0; float x = (b-a)/k; float g = (x> 1.0) ? x : (x<-1.0) ? 0.0 :
(x+1.0)(x+1.0)(3.0-x*(x-2.0))/16.0; return b - k * g; } // circular float smin(
float a, float b, float k ) { k = 1.0/(1.0-sqrt(0.5)); float x = (b-a)/k; float
g = (x> 1.0) ? x : (x<-1.0) ? 0.0 : 1.0+0.5(x-sqrt(2.0-x*x)); return b - k * g;
}

These are mathematically equivalent versions of the ones introduced earlier, you
can probably play a bit with pen and paper and rearrange the terms in the
formulas to convince yourself that this is true. Indeed they can be written in
either normal form (left) or the optimized form (right):

float smin( float a, float b, float k ) { k /= g(0.0); float x = (b-a)/k; return
b - kg(x); } float smin( float a, float b, float k ) { k /= g(0.0); float h =
max(k-abs(a-b),0.0)/k; return min(a,b) - kg(h-1.0); }

First note that all the smooth-minimums in this family are, through the use of
min() and the kernel, a function of the direct difference of the SDFs a and b.
That's why I called this the Direct Diffrence Family (or DD Family).

Now, the function g(x) is the only thing that's different among all the
smooth-minimum of this family. I call this function g(x) the Kernel of the
smooth-minimum. The Kernel plays the important role of defining many of the
characteristics of the smooth-minimum, and we'll later in the article design a
few interesting such kernels. For now, these are the graphs of a few of the
kernels above (quadratic, cubic, quartic, circular, sigmoid, square root):

A few DD kernels Kernel:

g(x) = (x·(2+x)+1)/4 g(x) = (1+3x·(x+1)-|x3|)/6 g(x) = (x+1)2(3-x·(x-2))/16 g(x)
= 1+½(x-√(2-x2)) g(x) = x/(1-2-x) g(x) = (x+√(x2+1))/2 Normalization factor:

g(0)=1/4 g(0)=1/6 g(0)=3/16 g(0)=1-√½ g(0)=1 g(0)=1/2

The shape of the kernel g(x) is very specific. When x tends towards infinity,
ie, when b is much larger than a, g(x) tends towards the identity x such that
the smooth-minimum returns a. On the other hand, when x tends towards negative
infinity, which happens when a is much larger than b, then g(x) tends
towards 0.0 such that the smooth-minimum tends towards b. Basically, the Kernal
behaves like the function max(x,0), or ReLU if you are a Machine Learning
person, but in a relaxed way. That's where the smoothness of the smooth-minimum
comes from. In fact, if we make g(x)=max(x,0), a perfect rectifier, then the
smooth-minimum reduces, mathematically, to a regular minimum.

The kernel plays a very important role in the normalization of the
smooth-minimum. Note that to the right of the graphs I've noted the value of the
kernel at the origin, that is, the height of its intersection with the y axis.
This is because that's the value we are subtractng from b when we are
equidistant to the two SDFs a and b, that is, when x = (b-a)/k = 0, because a=b.
That's the value we need to normalize all the smooth-minimums of this family
against, for it is the thickest part of the blend region created between the
source SDFs.

If you'd like to compute these normalization values yourself, please note that
in the case of the Sigmoid kernel you'll get a zero-divided-by-zero value, so
you'll need to use L'Hopital's rule to get the correct answer.

Now, some of the Kernels g(x) above never take the exact reach 0.0 nor x
exactly, no matter how large x is in absolute value. These are the exponential,
sigmoid and square root smooth-minimums, and that's the reason these three never
return exactly neither a nor b, no matter how far and close these shapes are to
each other. That in turn means that the shapes of the SDFs a and b will be
distorted everywhere in space, not just in the neighborhood of the region of
closest contact. I call this phenomenon "lack of Rigidity". On the other hand,
the Quadratic and Circular smooth-minimums use kernels to which we added extra
constraints that prevents this problem, and produce Rigid smooth-minimums,
putting them in a different sub-family of smooth-minimums: the "CD Family":

The CD Family

This is the subset of smooth-minimums from the DD Family that force the kernel
to connect perfectly with the identity and zero curves. Mathematically, their
kernels satisfy these equations:

g(-1) = g'(-1) = 0 g(1) = g'(1) = 1

These constraints effectively clamp the effect of the difference of distances,
so that the smooth-minimum doesn’t have any effect on the SDF shapes a nor b
once they are sufficiently far from each other, ie, k units apart or further.
This is why I’m calling this the "Clamped Differences Family" or "CD Family".

Naturally you can create an infinite variety of such kernels with these
constraints, and therefore there are an infinite amount of possible
smooth-minimums that you can use, but this article and the graph below show only
a few of the ones I think are most relevant (quadratic, cubic, quartic,
circular):

A few CD kernels Kernel:

g(x) = (x·(2+x)+1)/4 g(x) = (1+3x·(x+1)-|x3|)/6 g(x) = (x+1)2(3-x·(x-2))/16 g(x)
= 1+½(x-√(2-x2)) Normalization factor:

g(0)=1/4 g(0)=1/6 g(0)=3/16 g(0)=1-√½ For the sake of clarity, let me list again
some of the smooth-minimums in the the CD family:

// quadratic polynomial float smin( float a, float b, float k ) { k = 4.0; float
h = max( k-abs(a-b), 0.0 )/k; return min(a,b) - hhk(1.0/4.0); } // cubic
polynomial float smin( float a, float b, float k ) { k = 6.0; float h = max(
k-abs(a-b), 0.0 )/k; return min(a,b) - hhhk*(1.0/6.0); } // quartic polynomial
float smin( float a, float b, float k ) { k = 16.0/3.0; float h = max(
k-abs(a-b), 0.0 )/k; return min(a,b) - hhh(4.0-h)k(1.0/16.0); } // circular
float smin( float a, float b, float k ) { k = 1.0/(1.0-sqrt(0.5)); float h =
max( k-abs(a-b), 0.0 )/k; return min(a,b) - k0.5*(1.0+h-sqrt(1.0-h*(h-2.0))); }

And quickly as a historical note, I invented (deduced, I'd say) the
quadratic-polynomial smooth-minimum somewhere between 2011 and 2012, but I only
publish and shared the code in 2013 (in a few shaders in Shadertoy). Around 2015
the Media Molecule folks making the Dreams game optimized it (or maybe
re-developed it, should ask Dave Smith). I've been using their version ever
since they published the optimization. I'll however list the original
smooth-minimum code here below for refence, since I know it's still circulating
around in certain open source circles:

// quadratic polynomial float smin( float a, float b, float k ) { float h =
clamp( 0.5+0.5*(b-a)/k, 0.0, 1.0 ); return mix( b, a, h ) - kh(1.0-h); }

Now, looking at all the options within the CD family, you might be asking
yourself why would I ever consider using the circular variant above given it
uses a square root, which can be a bit too slow to evaluate in some situations.
We'll analyze and talk about this one later, but for now let's say that this
kernel is the only one that leads to a CD smooth-minimum that produces an
exactly circular connection between perpendicular objects. In that regard it is
equivalent to the circular-geometrical smooth-minimum presented in the first
listing in this article and which is used by some applications; but unlike the
circular-geometrical smooth-minimum, the kernel in the CD family does not
produce rendering and collision artifacts (more on this later).

Gradients

It is important that we stop here a bit to study the gradients of our
smooth-minimum functions. For one side, computing analytic gradients rather than
numerically can be a much faster way to do lighting or object placement and
alignment. But more importantly for this article, it helps us analyze the
behavior of our smooth-minimums mathematically. This is because a necessary (but
not sufficient) condition for a field to be a Distance Field (such as an SDF) is
that the length of its gradient is 1.0 exactly, everywhere. This can sometimes
not be achieved, but knowing when we fail at fulfilling this condition, and by
how much we fail to do so, is important information that helps us design our
rendering algorithms accordingly.

So, lets start the analysis by calling our smooth-minimum . Then its gradient
will depend on the gradients of the input SDFs a and b like this:

To check whether the resulting smooth-minimum f is also an SDF, we can look at
its gradient, see if it has a length of 1. We can check that by computing its
squared length:

which expands into a quite large expression. But luckily we can group terms and
get:

If the input SDFs a and b are both truly SDFs and not just approximated or
bounded SDFs, then

which simplifies the squared length of the smooth-minimum gradient to

Looking at this formula one thing is obvious - the length of the gradient
depends on the relative orientation of the two SDF's gradients ∇a and ∇b. Now,
armed with this information, let's analyze how the different smooth-minimum
functions behave:

Gradients of the DD and CD family

In the case of the DD family we can move a bit more further, since we know that

Therefore we can compute the partial derivatives in terms of the Kernel g(x)
like this:

This leads to the very compact expression for the gradient of the
smooth-minimum:

Remember that for our Kernels, we always have 0 ≤ g' ≤ 1. So according to the
first equation above, the gradient of the smooth-minimum is the linear
interpolation of the gradients of the input SDFs a and b. We know that other
than at the extremes, the linear interpolation of two vectors is shorter than
the two vectors (that's the reason we use slerp to interpolate
quaternions/rotations, rather than lerp). This means that while the
smooth-minimums in the DD family will never produce a correct SDF in the blended
region, because the length is shorter than 1 we know that they will never
overestimate the true distance, and therefore we can use them safely for
raymarching and collision detection!

We can also conclude the same thing by noting that in the second equation the
parabola 2g'(1-g') is positive in the interval 0 ≤ g' ≤ 1) and that the dot
product ∇a · ∇b ≤ 1, making the whole expression smaller than 1.

For the CD family of smooth-minimums, remember that outside the blending region
where |a-b|>k, the Kernel g(x) is exactly 0 or x, which has derivatives 0 and 1
respectively, making the length of the gradient |∇f|=1. This means that the DD
smooth-minimums are exact outside the blending regions, which we already knew
since we constructed them explicitly to be behave that way.

Since the Quadratic smooth-minimum is so common, it might be worth writing
explicitly its gradient. Now, since the quadratic polynomial's Kernel is
g(x)=(x·(2+x)+1)/4, we have that g'(x)=(x+1)/2. This results in the following
code which can be used to compute both the smooth-minimum and its gradient in a
single go, assuming now that the variables a and b bundle both the value of the
SDF in .x and its gradient in .yzw (note again the code below already accounts
for the normalization factor g(0)):

// .x = f(p) // .y = ∂s(p)/∂x // .z = ∂s(p)/∂y // .w = ∂s(p)/∂z // .yzw = ∇s(p)
with ∥∇s(p)∥<1 sadly vec4 smin( in vec4 a, in vec4 b, in float k ) { k = 4.0;
float h = max(k-abs(a.x-b.x),0.0)/(2.0k); return vec3(min(a.x,b.x)-hhk,
mix(a.yz,b.yz,(a.x<b.x)?h:1.0-h)); }

The gradient of smin() in 2D Code in Shadertoy

Having analytic gradients is more common than you might think for applications
where you want to compute surface normals during the raymarch itself (for
lighting volumetric shapes or for positioning elements in the scene), or for
Machine Learning, without having to implement a full auto-differentiation
framework. For examples you can visit the Distance+Gradient 2D SDF article where
I've compiled a list of primitive distances and their analytic gradients
computed efficiently by reusing expressions between both.

Circular Profile

Alright, earlier in the article I briefly mentioned that the particular shape of
the blended region can be an important thing to consider. Different
smooth-minimums will produce different profiles, even after normalization. And
when doing mechanical modeling in particular, it can be useful to ensure that
the profile of the blended region is perfectly circular. The Circular Geometric
smooth-minimum achieves this objective explicitly, by construction.
Unfortunately, it cannot be used for raymarching and collision detection as-is,
for it overestimates distances.

So this is where the Circular variant of the CD family comes to the rescue. It
achieves mathematically perfect circular profiles as well, but does not suffer
from the overestimation problem since it belongs in the CD family, and hence is
suitable for rendering. It is however a non-local blend, like all CD and DD
smooth-minimums. But let's see how I deduced the formula for the Circular
smooth-minimum:

First, we assume we are working with two SDFs a and b meet at a 90 degree angle,
and that k=1, so that:

Now, we want to match the profile of a perfect circle passing through (1,1) and
with radius of 1:

But because , we can substitute in the equation above and get a quadratic
polynomial in a:

which we can solve with the quadratic equation formula to get

Great. Now we need to match the gradient of f to that of the circle. From our
analysis on gradients we did earlier, we know that our CD Circular
smooth-minimum will have a gradient of the form

But since our SDFs a and b are vertical and horizontal planes, we have ∇a=(1,0)T
and ∇b=(0,1)T, so ∇f = ( g'(x), 1 - g'(x) )T. Also, the (unnormalized) gradient
of our circle ∇f points in the direction (1-a,1-b)T since it's a circle centered
at (1,1). So, we can write

where m is some constant of proportionality. We can isolate m from the first
equation

and replace it in the second to obtain an expression for g'(x) in terms of a and
b:

But we already computed both a and b in terms of x in the quadratic above, so we
can replace them and after a couple of simplifications, we get

This is fantastic, all we need to do now is integrate

and find the right constant C, to get our final result - a CD Kernel that
produces a perfectly circular shape when smoothly blending to perpendicular
SDFs:

This Kernel can also be written as g(x) = 1+sin(asin(x√½)-π/4), but there isn't
any reason to use inverse trigonometric functions when a square root will just
do. In fact, if you are a fellow oldschool programmer who got accustomed to CPU
coding, the use of a square root might still be a red flag for you. However, it
should not be - GPUs perform square roots almost in the same number of cycles as
multiplications, unlike the CPUs where the gap is at least an order of
magnitude. Still, if you are using CPUs after all for SDF evaluation and really
really want to squeeze every single clock cycle available, then you can probably
approximate the whole Circular kernel g(x) with a polynomial. For example, if we
take a degree 4 polynomial

then we have 5 coefficients/constraints to determine, from a4 to a0. But because
our approximation will be of CD family, we already have four such constraints
g(-1) = g'(-1) = 0 and g(1) = g'(1) = 1. The last constraint can be simply that
g(0) matches that of the circular kernel, that is, g(0) = 1 - √½. With that we
are ready to set the system of equations

which doesn't take much work to solve:

This gives us all we need to code the circular approximation smooth-minimum,
either in normal form (left) or in optimized form (right):

// circular approximation float smin( float a, float b, float k ) { k
\= 1.0/(1.0-sqrt(0.5)); const float a4 = 3.0/4.0 - sqrt(0.5); const float a2 =
-5.0/4.0 + 2.0sqrt(0.5) ; const float a0 = 1.0 - sqrt(0.5); float x = (b-a)/k;
float g = (x> 1.0) ? x : (x<-1.0) ? 0.0 : x*(x*(xxa4+a2)+0.5)+a0; return b - k *
g; } // circular approximation float smin( float a, float b, float k ) { k
\= 1.0/(1.0-sqrt(0.5)); float h = max( k-abs(a-b), 0.0 )/k; const float b2
= 13.0/4.0 - 4.0sqrt(0.5); const float b3 = 3.0/4.0 - 1.0sqrt(0.5); return
min(a,b) - khh(hb3(h-4.0)+b2); } To get the code for the optimized form, all you
need to do is propagate x=h-1 into the polynomial.

Now, how does this compare to the exact circular smooth-minimum? Let's test it
with two rectangles and a large blend band k. On the left you can see the
quadratic smooth-minimum, on the right is the circular (which is exact), and in
the middle you see the 4-th degree approximation that we just computed. The red
arc is the ideal circular blend section. If you look closely you'll see the
approximation doesn't conform to it exactly, there are a few white pixels that
leak outside the red arc. In applications where this is a problem you might want
to go to a degree 6 polynomial perhaps:

Quadratic smooth-minimum

Circular Approximation smooth-minimum

Circular smooth-minimum

Quadratic smooth-minimum

Circular Approximation smooth-minimum

Circular smooth-minimum

Gradients of the Circular Geometrical

The Circular Geometrical smooth-minimum, operates very differently to the DD
family. I saw it first in Shadertoy used by folks learning from the demoscene
group Mercury. Instead of relying on the difference of distances a and b like
the DD family, it assumes these two SDFs are perpendicular and it explicitly
constructs a circle connecting them, centered at (kk,k)T with radious k. So
let's do some math see how that works, and without loss of generality, assume
that k has already been normalized by 1-√½:

So the partial derivatives of f with respect to a and b are:

which means that the gradient of the Circular Geometrical smooth-minimum is:

and its length squared is

The only way for this quantity to be less that one (which is required to prevent
overestimating distances) is:

The denominator is positive and the numerator cannot be negative since inside
the blending region both a≤k and b≤k, making their product positive. So the only
regions of space where the gradient's length is less than one is given by

That is the concave regions of space, and only there the Circular Geometrical
smooth-minimum under-estimates. In other words, in the convex areas of the the
union of two shapes the Circular Geometrical smooth-minimum overestimates the
distance to the surface of the blended object, and vanilla raymarchers will fail
and produce rendering artifacts, as we are about to see.

Regions of distance under/over-estimation

Here's a diagram showing how the CD smooth-minimums and the Circular Geometrical
smooth-minimum behave. I've used the Quadratic Polynomial smooth-minimum as a
representative of the CD family, but they all have very similar behavior
diagrams.

Quadratic smooth-min

Circular Geometrical smooth-min Here the shadowed regions of the plane indicate
areas where the smooth-minimum returns a lower bound of the real distance to the
resulting shape (white boundary). Again, these are regions where the length of
the gradient is less than 1.0, where sampling the smooth-minimum (indicated by
the yellow dots) produces distance bounds (yellow circles) that do not touch the
shape. This means in these regions a raymacher will perform slower than in the
non-shadowed areas where it can traverse the scene at the Speed of Light.

To make things worse, the regions of under-estimation for ALL smooth-minimum
belonging on the DD and CD families span to infinity, ie, non-local. Indeed,
these shaded regions in the left image only get larger and larger as we move
away from the shapes, which is easy to prove: if you split the plane in a
Voronoi diagram where each point in the plane is associated with the shape a or
b that is closest to it, you will always find a band where |a-b|<k that extends
along the edges of this Voronoi graph.

This is bad news for our CD Family, which again, depending on the use of the
SDFs, can make certain rendering or collusion detection methods that rely on
bounding boxes be slightly more difficult than they could be otherwise. But not
all CD smooth-minimums are equally bad, here you can see a comparison of the
areas of sub-Speed-of-Light for different variants (the less dark shadowed
yellow, the better):

Quadratic

Cubic

Quartic

Circular

On the other hand, the Circular Geometrical smooth-minimum is locally supported
unlike the CD family, and has its underestimation and overestimation regions
bounded, which you can see by seeing that when far enough from the
smooth-minimum shape, the plane is all regular yellow and isolines are exact.
Leeping the slow raymarching regions contained to around the objects is a great
property of course.

However, here's the big drawback of the Circular Geometrical smooth-minimum - as
we mentioned earlier, it often overestimates distances. The red colors indicate
regions where the smooth-minimum overestimates the distance to the shape, which
is very dangerous. I've marked in light red those where the length of the
gradient is larger than one as discussed earlier, and in darker red regions
where the gradient has length one yet the smooth-minimum is returning the
distance to the underlying SDFs a or b rather than to the blended shape.
Sampling the smooth-minimum in those regions produce distance bounds (yellow
circles) that penetrate into the shape, or in other words, these are regions
where raymarcher will go faster than the Speed of Light and things will break
badly and renders will fail by showing holes on the objects, like seen in the
image on the right in the following comparison:

Circular smooth-min

Circular Geometrical smooth-min While this seems bad, it can sometimes be easy
to remedy by letting the raymarcher back-track when penetrating the surface,
which can be done by breaking the loop based on the absolute value of the
distance to the surface rather than the signed distance. The effort might be
worth it, considering the Locality of the Circular Geometrical smooth-minimum
which translates into less raymarching iterations to render the same picture.
Here in the image below blue regions indicate the intersection was found in less
than 15 iterations, yellow less than 25, and red 30 or more:

Circular smooth-min

Circular Geometrical smooth-min You can see that the CD Circular smooth-minimum
required more iterations to resolve than the Circular Geometrical, specially
around the center of the image.

Properties

Okey, let's gather all the things we've talked about so far about
smooth-minimum, and make a list of the properties we care most about, one by
one.

Rigidity : Some smooth-minimum will distort the SDFs a and b no matter how far
they are from each other. However, the CD family of smooth-minimums and the
Circular Geometrical do preserve the shape of the SDFs a and b everywhere but in
the blending region.

Locality : all smooth-minimum will always produce non-exact SDF in some regions.
As we saw, for the CD Family there are always infinitely large regions where the
smooth-minimum under-estimates the distance, no matter how far we move away from
the shapes or how far apart these shapes are. On the other hand, the Circular
Geometric smooth-minimum has Local Support and doesn’t suffer from this problem,
making it an attractive option depending on the application.

Conservative : smooth-minimums that never overestimate distances are
Conservative. This is a very desirable property, for it makes algorithms like
raymarching and collision detection easier to implement in a robust manner
without producing visual artifacts. Fortunately, ALL the smooth-minimums in the
CD Family are guaranteed to produce underestimates by constriction, both because
the kernel g(x)≤0 and also because the length of their gradient is always less
than one. However, the Circular Geometrical smooth-minimum violates this
property in many regions of space as we saw earlier, making it a less attractive
option in some applications.

Associativity : the CD Family of smooth-minimums is not associative, in that
smin(a,smin(b,c)) ≠ smin(smin(a,b),c). That is, the order in which you blend
objects matters. This is certainly true for all members of the DD family.
However, the Exponential and the Circular Geometrical can be blended in any
order!

So here goes a summary:

Rigid	Local	Cons.	Asso. Quadratic	Yes	No	Yes	No Cubic	Yes	No	Yes	No
Quartic	Yes	No	Yes	No Circular	Yes	No	Yes	No Exponential	No	No	Yes	Yes
Sigmoid	No	No	Yes	No Root	No	No	Yes	No Circular Geometric	Yes	Yes	No	Yes

As you can see, there's no single smooth-minimum I know of that has all the
desired properties, so the choice will be depend on your and what are the
requirements of the application using it.

Results

In general I use the Quadratic polynomial smooth-min function because it's fast,
close enough to circular, never overestimates, and while not local it still not
too bad. Although the Circular is also a great option. Regardless, here go a few
examples of the many uses I've done of the Quadratic smooth-minimum in the past
to connect surfaces, such as snow and bridge in the images below.

Regular min()

Polynomial smooth-min() Note how the snow gently piling by the bridge thanks to
the smooth minimum, and how thanks to the mix factor described above we can
transition between the snow and stone materials (source code and realtime demo:
shadertoy.com/view/Mds3z2).

Regular min()

Polynomial smooth-min(). Note how the regular min() union cannot smoothly
connect the legs of the creature to its body (source code and realtime demo:
shadertoy.com/view/Mss3zM). Naturally, the technique is very handy for
connecting the different pieces of one same character, such as the arms, head
and body, which in the case of the following realtime shader are made of
spheres, ellipsoids and segment primitives):

inigo quilez - learning computer graphics since 1994

Inigo Quilez :: articles :: normals for an SDF - 2015

Intro And SDF is a Signed Distance Function, and in the context of computer
graphics, it's usualy employed to rapidly raymarch geometry and scenes. When
doing lighting on such scenes, or collision detection, having access to the
surface normals of the geometry is necessary. The surfaces that emerge from an
SDF f(p), where p is a point in space, are given by a particular iso-surface,
normally the f(p) = 0 isosurface. Computing the normal n of that isosurfaces can
be done through the gradient of the SDF at points located on the surface.

Remember that the gradient of a scalar field is always perpendicular to the
iso-lines or iso-surfaces described by the scalar field, and since the normals
to a surface need to be perpendicular, the normals must align with the gradient.

Surface normals filtered for antialaising, computed with the tetrahedron
technique below (realtime version: https://www.shadertoy.com/view/ldScDh)

There are multiple ways to compute such gradient. Some are numerical, others are
analytical, all with different advangates and disavantages. This article is
about numerically computing them, which requires the least code writing, but
might not be the fastest or most accurate. Still, its simplicity make it the
most popular way to compute normals in realtime raymarched demos and games.

Classic technique - forward and central differences

In its simplest form, we take the definition of gradient.

Those partial derivatices can be computed with small differences, as we know
from the definition of derivative. For example,

where h is as small as possible. This is called forward differentiation, since
it takes the point under consideration p=(x,y,z) and evaluates f at
p'=(x+h,y,z), a point in the possitive x direction. Backwards and central
differences are also possible, which take the following forms:

Actually central differences are often preferred since they don't favor neither
possitive nor negative axes, which is important in order to make sure the
normals follow the surface without any directional bias. Note the division by
two times h in the central differences method, which is the distance between the
function sampling points.

Using central differences, the normal therefore takes the following form

Note how the division by 2h can be simplified away since the normalization will
rescale the normal anyways to unit length. When encoding this in a program, it
looks like this:

vec3 calcNormal( in vec3 p ) // for function f(p) { const float eps = 0.0001; //
or some other value const vec2 h = vec2(eps,0); return normalize(
vec3(f(p+h.xyy) - f(p-h.xyy), f(p+h.yxy) - f(p-h.yxy), f(p+h.yyx) - f(p-h.yyx) )
); }

This form can be found in many raymarching demos in places like Shadertoy or the
demoscene productions. However, if evaluating f(p) six times becomes too
expensive, one can use forward differences instead

since it only requires four evaluations instead of six. Even though the
following is not true due to numerical reasons, in some special cases we can
assume f(p) is zero though, since by definition we are looking for the zero
isosurface, and get away with the following three evaluations:

Tetrahedron technique

There's a nice alternative to the direct gradient definition technique that is
based also on central differences, which means the lighting done on the normals
will follow the surfaces without any shift, that uses only four evaluations
instead of six, making it as efficient as the forward differences. I first saw
this technique used by Paulo Falcao in Pouet around 2008 and then later by Paul
Malin in Shadertoy. I have embraced it in most of my shaders since then, and it
looks like this:

vec3 calcNormal( in vec3 ) // for function f(p) { const float h = 0.0001; //
replace by an appropriate value const vec2 k = vec2(1,-1); return normalize(
k.xyyf( p + k.xyyh ) + k.yyxf( p + k.yyxh ) + k.yxyf( p + k.yxyh ) + k.xxxf( p +
k.xxxh ) ); }

The reason this works is the following:

The four sampling points are arranged in a tetrahedron with vertices k0 =
{1,-1,-1}, k1 = {-1,-1,1}, k2 = {-1,1,-1} and k3 = {1,1,1}. Evaluating the sum

on those four vertices produces some nice cancellations that results in

which are four directional derivatives, meaning we can rewrite the sum as

We can now proceed to rewriting this using matrices, or we can keep looking at
one component at a time. Let's do the later. For x, we get

(this makes use of the fact that the dot product is a linear operator). The
results are similar for the y and z components, meaning that

which after normalization gives us our normal at the zero-isosurface.

An important implementation detail

I hope I can obsolete this paragraph some day in the future, but as of today,
your development platform might decide to take the above code and inline the
four calls to f() inside calcNormal(). If f() is long, as it often is, that can
create problems with the number of intructions available in your platform, and
often times it can crash the shader compiler (often happens in WebGL). One trick
to prevent the compiler from inlining f() is to make sure the loop depends on
some input uniform that the compiler doesn't know about. Thomas Hooper though of
this trick first and has since become pretty common among the Shadertoy
community, since it helps both prevent the crashes and also get the compilation
times down by a lot. This a variation of his trick made by Clément Baticle
(a.k.a. Klems):

vec3 calcNormal( in vec3 p ) // for function f(p) { const float h = 0.0001; //
replace by an appropriate value #define ZERO (min(iFrame,0)) // non-constant
zero vec3 n = vec3(0.0); for( int i=ZERO; i<4; i++ ) { vec3 e
\= 0.5773*(2.0vec3((((i+3)>>1)&1),((i>>1)&1),(i&1))-1.0); n += emap(pos+e*h).x;
} return normalize(n); }

Cetral differences for Terrains

When rendering raymarched terrains we usuaully have a heightmap of the form
g(x,z) that defines it through y = g(x,z). This can be rewriten as f(x,y,z) = y
- g(x,z) = 0, which means that while not a distance field it is still a scalar
field, so we can still use all the gradient logic in order to comptue its
normal. By doing so, we get

which means that the normal can be computed simply as

vec3 getNormal( in vec2 p ) // for terrain g(p) { const float eps = 0.0001; //
or some other value const vec2 h = vec2(eps,0); return normalize( vec3(
g(p-h.xy) - g(p+h.xy), 2.0*h.x, g(p-h.yx) - g(p+h.yx) ) ); }

Beware that because we are using central differences and the distance between
the sampling points is 2h, removing the division by 2h means we need to multiply
the y component (which is 1) by 2h. This is the code that you can also find in
article about terrain marching. You can arrive to the same formula if you
consider the four little triangles that you can create if you were
polygonalizing the terrain field with triangles of size h, and were computing
normals for those triangles and then averaging them, as you would be doing for a
mesh.

Some words about sampling and aliasing

You might have noticed that so far we haven't paid any attention to the value of
h. In theory, it needs to be as small as possible in order for the components of
the gradient to properly approximate the spatial derivatives. Of course too
small values will introduce numerical errors, so in practice there's a limit to
how small h can be.

However, there's another important consideration when it comes to picking a
value - geometry detail and aliasing. Indeed, when taking our central
differences at a point in space, we should consider how far it is from camera,
or more exactly, what's the footprint of the current pixel being rendered into
world space at the sampling point. The idea is that we want to know how tiny do
geometrical details get when projected to the screen in the neighborhood of the
sampling point - or equivalently, how big is the pixel footprint compared to the
geometrical detail in the area. We want this information so that we can do Level
of Detail (LoD) in the raymarched geometry - we don't need to compute SDFs for
objects/features that are really far in the distance and get too tiny due to
perspective projection. This is not only a performance optimization, but more
importantly, an image quality issue. By ensuring that we do not consider and
sample geometry that is not big enough to be reliably captured by a ray
representing the whole pixel, we are effectively implementing a type filtering
(band limiting) that prevents aliasing.

Now, under such circumstances, one needs to make sure that the h used to sample
the gradients is also about the size of the pixel footprint (or the size of the
biggest detail we decided to reject as part of the LoD system). This is going to
be something proportional to the distance from the sampling point to the camera,
often referred as t in raytracers and raymarchers.

The effect is very dramatic, as can be seen in the aniamted GIF below. Look to
the distant cliff walls on the left side of the image:

On the left you can see the stabilized/filtered normals computed by choosing h
proportionally to the pixel foorprint. To the right is the naive version that
uses a constant h (0.001 in this case, making it bigger makes the image lose
detail in the foreground). The difference is pretty big.

inigo quilez - learning computer graphics since 1994

Inigo Quilez :: articles :: ellipsoid SDF - 2008

Intro

One of the most useful primitives when modeling organic shapes with SDF us the
ellipsoid. However, unlike spheres, cones, boxes and even torii, ellipsoids
don't have an analytic distance function that can be evaluated by using roots
and/or trigonometric functions. This is bad because that means that in principle
one cannot use ellipsoids for modeling, we need to use a numerical method to
implement their distance function. That, of course, is pretty slow most (a
bisection method would require around 10 to 20 iterations to get good results).
So, instead, we need to resort to approximate distance functions. Luckily for
us, we CAN find bound functions that at least have a zero iso-surface in the
shape on an exact ellipsoid. This means that such bound functions will produce
renders of exact ellipsoids, but will report non euclidean distances when
queried. That means that a raymarcher will have a harder time finding them and
will require more steps until it finds an intersection. It also means occlusion
and shadow queries will be wrong for ellipsoids, to different degree based on
which bound function we are using. This is an article about two such bound
functions.

Shadows produced by an exact SDF

Too large shadows due to a naive bound SDF

Improved shadows with improved bound SDF

Disclaimer As a matter of fact, it is possible to compute the exact distance to
an ellipsoid in closed from when the ellipsoid is symmetrical in one axis. In
that case, the shape cam be evaluated as a 2D ellipse that is revolved along one
perpendicular axis in 3D. Since ellipses do have exact solution in the form of a
cubic, and since creating shapes of revolution is trivial and doesn't increase
the degree of the polynomials, ellipsoids that are revolved ellipses do have a
closed form. All images in this article that compare the two bound techniques
use symmetric ellipsoids so I can compare them to the ground truth that I know
is exact.

First approach

The simplest approach to bounding the distance to an ellipsoid is stretching the
space such that the ellipsoid becomes a sphere, computing the distance to a unit
sphere in that space, and then scale the distance back to the original space by
the largest of the scale factors. The code would be:

float sdbEllipsoidV1( in vec3 p, in vec3 r ) { float k1 = length(p/r); return
(k1-1.0)*min(min(r.x,r.y),r.z); }

Second approach

A simple way to improve the previous bound is to divide by the length of its
gradient. After some math on paper and nice rearrangement, one gets to this:

float sdbEllipsoidV2( in vec3 p, in vec3 r ) { float k1 = length(p/r); float k2
= length(p/(rr)); return k1(k1-1.0)/k2; }

This involves a second square root (or rather, an inverse-square root, which
happens to be faster in that a square root when computed in the GPU). But the
extra cost comes with benefits.

Third approach

Another idea that might pop in your head is to change the first technique a
little bit and see if that improves the distance estimate: stretch the space to
deform the ellipsoid into a unit sphere, find the closest point in the unit
sphere, transform the point back to the source space, and compute the distance
there. The technique actually works great in that does produce a better distance
estimate. However, it fluctuates between being an underestimate and
overestimate, and even with backtracking raymarchers it doesn't seem to work
well, and I won't discuss it further in this article, although it can be used
for 2D plotting.

float sdaEllipsoidV3( in vec3 p, in vec3 r ) { float k1 = length(p/r); return
length(p)*(1.0-1.0/k1); }

First vs Second techniques

Let's compare the two proposed techniques to the ground truth distance to an
ellipsoid in the context of raymarching.

These images below are a map of the number of steps that the raymarcher needed
to find the primary ray intersection in the scene rendered at the beginning of
the article. Blue means "few steps", around less than 32, green is "moderate
number of steps", around 64, and red means "many steps", around 128.

Ground Truth

First Technique

Second 2 As you can see, the First Technique does a poor job at bounding the
distance to ellipsoid tightly, while the Second Technique does a much better
job, resulting in an image that renders faster, or alternatively, has less
artifacts for the same render time. Speaking of that, here goes the comparison
of actually fixing the number of steps and seeing how well the different
techniques do at resolving the surface of the ellipsoid for that fixed compute
budget:

Ground Truth

First Technique

Second Technique Indeed, for a set maximum number of iterations of 90, the First
Technique is not able to find the surface of the ellipsoid (although it would
for 150 iterations), producing black pixels around the edges of the ellipsoid.
The Second Technique works much better.

You can find a real-time animated comparison of the two techniques here:
https://www.shadertoy.com/view/tdS3DG.

The most important benefit of the Second Technique over First Technique,
however, is not the more efficient raymarching. The main benefit is that it
produces a much closer distance estimate to the ground truth, and in particular,
it makes it much more Euclidean. That means that when combined with other
primitives that produce exact Euclidean distances, the Second Technique produces
values that play along nicely with those produced by the other primitives. That
means that one can adjusting shadow softness parameters, occlusion thresholds
and many other values globally for the whole SDF and keep consistent results.
With the First Technique the ellipsoids always belong in a different distance,
such to speak, universe and are difficult to control.

This is an example of using the Second Technique (left image) vs First Technique
(right image) - while both produce the same geometry, the shadows produced by
the cap, which is made of two ellipsoid one of which is very flat, are pretty
broken and difficult to art direct. The Second Technique produces a sensible
distance field and plausible shadows:

Second Technique

First Technique You can find the real-time animated image and source code that
contains the ellipsoid SDF here: https://www.shadertoy.com/view/ldd3DX

inigo quilez - learning computer graphics since 1994

Inigo Quilez :: articles :: FBM detail in SDFs - 2019

Intro

As Signed Distance Functions start making it into mainstream and commercial
applications, it's important to find replacements or alternatives to common
things artists used to do in polygon-land. One such thing is displacement as a
means to enhance or add detail to a shape. While much of what I'll say here
applies to any kind of displacement pattern or SDF based "detail mapping", I'll
focus on pure fBM style displacement in this article, the one we use to make
procedural terrains for example (you might know it as "Fractal Noise" too). And
the reason to find an alternative to it is that the traditional way of
constructing and applying fBM (and displacement/detail) does not work well with
SDFs. But, I've found an alternative that I think is competitive. So keep
reading!

fBM SDF detail (not as a regular fBM displacement)

The problem

The problem with tradicional noise/fBM displacements on SDFs is this: the
(regular, arithmetic) addition of two SDFs is not an SDF. Furthermore, the
addition of an SDF and some other field/function (SDF or not) is also not an SDF
(except for very carefully manufactured functions, that is). So, when adding a
regular fBM, sine wave or any other displacement function to a "host" SDF, we
don't get a valid SDF anymore (we violate the principle that the gradient of an
SDF must have length 1.0).

That of course doesn't stop us from still trying, and we often do add arbitrary
functions to our SDFs in the hopes of changing their shape. Sometimes we can
achieve some level of success when done gently, but it all breaks rapidly when
we push it a little bit. For example, when rendering SDFs through a raymarcher,
adding a sine wave to a sphere can work for small amplitudes and frequencies, if
the raymarcher happens to be designed to tolerate small deviations from true
SDFs (always at the cost of performance). However it will sooner or later break
as we make the sine wave larger or its wavelength shorter (since that pushes the
length of the field's gradient away from 1.0 quickly).

That said, since we haven't really had an alternative for it, this method is
still used for things that traditionally have been solved through displacement
in polygon-land, such as the fractal terrains I mentioned in the intro; even
though they don't work well or efficiently simultaneously. And this is
unfortunate because fBM signals are very popular, well understood, widely
implemented by all sorts of modeling, texturing and painting software, and
widely used by artists. It's really a pity we can't simply use them with SDFs in
a reliable manner.

Or can we?

A solution

Well, we know addition of functions doesn't work well with SDFs, so let's try to
workaround it by redefining addition, see if we can repurpose and save fBMs.

The most important aspect of the addition in an fBM or fractal process, is not
its arithmetic aspect or computation per-se. What's important is that waves of
different amplitudes and wavelengths are being additively combined together,
meaning, they sit on top of each other. That corresponds to how shapes in nature
organize themselves too (hence the success of fBM in procedural modeling and
texturing). So what we need is to define a new "addition" of SDFs that allows us
to combine shapes together and grow them on top each other.

The "combining" part of our requirement is easy and we can do it with a simple
union or smooth-union operation. That would be a min() or smoothmin() probably.
In other words, as we generate SDFs of shorter and smaller amplitudes and
wavelength, we can combine them with the "host" shape and to each other through
the regular SDF union operations.

The "on top of each other" part of the addition can be achieved by making sure
that these SDF layers (called "octaves" traditionally in standard/vanilla fBM
implementations) only exist in the vicinity of the previous layer (or "host"
object that we are applying our fBM displacement to). That way only the surface
of our object will be augmented with the higher frequency detail, and no new
surfaces will be created elsewhere.

One easy way to accomplish this is by clipping the SDFs layers against a
slightly inflated version of the "host" SDF, ideally through a
smooth-intersection to keep the smoothness of the final shape. That'd be a max()
or smax() operator. Depending on the shape of the SDF layers we can still have
flyovers (disconnected pieces of surface), so this is not a bullet proof method,
but it works well in practice.

An implementation

So, let's put all these ideas together. First we need a random and smooth SDF to
use as base function for our fBM. Since traditional 3D noise() is not a distance
function (it's an Signed Field/Function, but doesn't measure distances), we
cannot use it. Instead, we'll use an infinite but simple grid of spheres of
random sizes. Spheres have simple SDFs, are isotropic and so they feel like
natural candidates. Making an infinite grid of them is easy as well with some
basic domain repetition. If we restrict the radius of our random spheres to be
smaller than half the edge-length of the grid, then for a given point in space
we only need to evaluate the SDF of the 8 spheres at the corners of the grid
cell the point belongs to.

The code below is a possible implementation of such sdBase(), and to the right
is a direct rendering of it:

float sph( ivec3 i, vec3 f, ivec3 c ) { // random radius at grid vertex i+c
float rad = 0.5*hash(i+c); // distance to sphere at grid vertex i+c return
length(f-vec3(c)) - rad; }

float sdBase( vec3 p ) { ivec3 i = ivec3(floor(p)); vec3 f = fract(p); //
distance to the 8 corners spheres return min(min(min(sph(i,f,ivec3(0,0,0)),
sph(i,f,ivec3(0,0,1))), min(sph(i,f,ivec3(0,1,0)), sph(i,f,ivec3(0,1,1)))),
min(min(sph(i,f,ivec3(1,0,0)), sph(i,f,ivec3(1,0,1))),
min(sph(i,f,ivec3(1,1,0)), sph(i,f,ivec3(1,1,1))))); }

sdBase(), an infinite grid of spheres with random radius Once we have the base
SDF sdBase(), we can start using it in our additive fractal construction of fBM
with the redefined "addition" described above:

float sdFbm( vec3 p, float d ) { float s = 1.0; for( int i=0; i<11; i++ ) { //
evaluate new octave float n = s*sdBase(p);

   // add
   n = smax(n,d-0.1*s,0.3*s);
   d = smin(n,d      ,0.3*s);

   // prepare next octave
   p = mat3( 0.00, 1.60, 1.20,
            -1.60, 0.72,-0.96,
            -1.20,-0.96, 1.28 )*p;
   s = 0.5*s;

} return d; }

sdFBM(), adding 11 octaves of sdBase() together Here we are adding 11 layers of
sdBase(), each of them with twice the frequency (of half the wavelength) of the
previous one. The 2.0 scaling factor is hidden inside the 3x3 matrix that
transforms p in each iteration. Because of this frequency doubling for each
additional layer, we call these layers "octaves", like in music (where moving
from one scale to the next doubles its pitch). The variable s keeps track of
this scaling factor and applies it to sdBase() in order to bring the distances
to the same metric space. Finally, the rotation is there to break up the
otherwise obvious alignment of the different sdBase() layers. Any rotation will
work, although some will do better than others and I must confess I used this
one because it's rational and therefore short when written down rather than
because it's the best possible one, so by all means experiment and try to find a
better one.

So far this construction is identical to that of a basic fBM. However, instead
of simply adding each layer n to the "host" SDF (passed as d to the function),
we perform the modified "sdf addition": first we smoothly clamp the noise SDF n
to the an inflated version of the host surface d. The inflation factor is 0.1s
in this case, but again, you should play with that. Just remember to make it
proportional to s. The smoothness factor 0.3s should also be subject to
customization, but it's probably a good idea to keep it proportional to s as
well, so we keep fractal detail at all scales.

The second part of our "sdf addition" is to combine the current layer n with the
"host" sdf d. We do this, as we said, with a regular smin() operation with
certain smoothness factor, 0.3*s in our case, which you should also experiment
with and customize for your desired look.

In the image to the right side of the code snippet you can see what adding each
one of these sdBase() layers does to the "host" SDF, a plane in this case. As
you can see the process works great, and most importantly, the method produces a
valid SDF, within the capabilities of smin() and smax() to do so, that is way
better suited for raymarching and also for distance based lighting techniques
(ambient occlusion, soft shadows) and collision detection than naive addition of
fBM to the "host" SDF.

And the following is a video showing the fBM SDF construction, slightly modified
for cosmetic reasons, together with some rendering of the resulting surfaces:

And the following is a realtime version of it in Shadertoy where you can see the
technique in action, that comes with reference code so you can study and modify
it directly: https://www.shadertoy.com/view/3dGSWR

LOD

As with most fractal constructions, one can easily band pass filter the geometry
at construction time, and filter out detail that is not needed for a given pixel
in the screen (or shadow-ray cone). To implement something like this, all we
need to do is measure the maximum size the following sdBase() call can
contribute with, which is given by the current scale factor s, and if it is
below a given threshold th then break the loop. That's threshold th should be
based on the size of a pixel in world space at the location of the fFM
invocation, which you can compute easily with ray differentials. Basically,

float sdFbm( vec3 p, float d, float th ) { float s = 1.0; for( int i=0; i<11;
i++ ) { // evaluate new octave float n = s*sdBase(p);

   // add
   n = smax(n,d-0.1*s,0.3*s);
   d = smin(n,d      ,0.3*s);

   // prepare next octave
   p = mat3( 0.00, 1.60, 1.20,
            -1.60, 0.72,-0.96,
            -1.20,-0.96, 1.28 )*p;
   s = 0.5*s;

   // lod
   if( s<th ) break;

} return d; }

Variations

This fBM SDF technique accepts a million variations, of course. For example,
sdBase() could be built with spheres that are not of random sizes but are
instead located at random positions within the grid. Or both random sizes and
random positions could be implemented, like in a "voronoise" pattern. They could
also be smoothly blended together rather than being independent spheres. They
could be arranged in tetrahedral simplexes rather than in a grid lattice for
better performance too. Or the grid could be defined in polar coordinates rather
than rectilinear, which can be useful if the "host" surface has cylindrical
symmetry. Or it could be arranged in a logarithmic spiral or some other shape.
Lastly, of course, we could replace the spheres with some other completely
different primitive, such as cubes, torii or even ellipsoids to give the noise
some anisotropic properties, Gabor-noise style.

Besides variations to sdBase(), there's plenty of room to play and be creative
with the fBM SDF construction itself. For example, in the video above I
displaced the grid by a small amount in each iteration, which helped increase
the number of concavities in the field that looked like cliffs. Of course, the
scaling factor could be a function of the SDF itself, or of space, or a
combination. And lastly, the fBM could be using sdBase() in a subtractive manner
rather than additively, effectively carving out fractal patterns from solids:

float sdFbm( vec3 p, float d ) { float s = 1.0; for( int i=0; i<7; i++ ) { //
evaluate new octave float n = s*noiseSDF(p);

   // subtract
   d = smax( d, -n, 0.2*s );

   // prepare next octave
   p = mat3(0.00, 1.60, 1.20, 
           -1.60, 0.72,-0.96,
           -1.20,-0.96, 1.28)*p;
   s = 0.5*s;

} return d; }

Subtracting 7 octaves of sdBase() from a box SDF A realtime version and
reference code used to generate this subtractive picture above can be found here
in Shadertoy: https://www.shadertoy.com/view/Ws3XWl.

Naturally the possibilities and endless and only limited by your imagination.

Problems

While the technique works great and has lots of advantages over adding
traditional fBM signals to the "host" surface, it also comes with some problems.
The main one is that it can sometimes produce "flyovers" - small pieces of
surface that are disconnected from the "host" SDF in the case of additive fBMs
(little holes will equivalently be created at undesired locations for
subtractive fBM). This also happens with traditional 3D fBM as soon as we try to
generate cliffs, so it's not a problem specific to the fBM proposed in this
article. But it's nevertheless something we could try to alleviate. In some
circumstances, it is possible to guarantee no flyovers will be generated, but
making the clip (smax) and combine (smin) steps more conservative, at the cost
of reducing the amount of detail on the surface.

Conclusion

The technique is awesome as far as I can tell and produces beautiful pictures
not only for terrains, but generally for growing/displacing any type of SDF
detail on top of other SDFs.

inigo quilez - learning computer graphics since 1994

Inigo Quilez :: articles :: directional derivative - 2013

Intro When we learn the concept of gradient at school it is presented to us as
the primary way to describe and deal with directions and orientations, probably
because it describes the local change of a function in a very general way. And
so, a related concept called the "directional derivatives" is barely talked
about, except perhaps as an intermediate concept to arrive at that of a
gradient. And while the generality of the gradient probably grants the almost
exclusive focus we give it, we shouldn't forget that sometimes we don't want the
general solution or implementation, but the specialized and optimized one. Like
when you want to compute lighting from a single light source for 3D objects
efficiently without computing normals, or when you are painting volumetric
clouds.

Diffuse lighting without surface normals (code:
https://www.shadertoy.com/view/Xl23Wy

Fast (realtime) dynamic lighting on volumetric clouds with directional
derivatives

Say you are doing some real-time rendering of volumetric clouds, and that you
need to do some lighting and shaping without scattering and self shadowing
computations. You need to go cheap, but even extracting the gradient or normal
from the cloud volume that you need to do your regular Lambertian lighting is
already expensive. You are probably evaluating your gradient by taking 4 or 6
samples of the volume, depending your implementation, only to then dot it with
the light direction. Which works, but is very slow because evaluating your
(possibly procedural) volumetric field 4 or 6 times is your bottleneck.

The idea So now forget what your teacher told you about gradients and have a
look to this article on the directional derivatives in the Wikipedia. In
particular, look at this formula:

Now, if x was the point in space we are shading/lighting, and f was out SDF or
cloud density field, then f(x) would be the density at that point we are
shading, and ∇f(x) the gradient (or 'normal'). At the same time, if v was the
light direction, then the right side of the equation ∇f(x)⋅v/|v| would be
nothing but our regular N⋅L Lambertian lighting... which according to the
equation is equal to the directional derivative of the field taken in the
direction of the light (left side of the equation)!

So basically, instead of extracting a general derivative in all possible
directions and dot with the one direction of interest, you can measure the
change (derivative) directly in that direction of interest. Or in other words,
rather than taking 4 or 6 samples to extract a generic derivative or gradient,
and then dot it with the light direction to do our lighting, we could simply
sample the field no more than 2 times, at the current point and at a point a
small distance away in the direction of the light (and divide by that distance
of course). So, something that is 4 or 6 evaluations can be reduced to one.
Since one evaluations has been already done for computing the opacity of the
volume, we are now really doing two evaluations rather than 5 or 7. Which is a
massive speedup.

The code So, let's say we have an SDF or a volumetric function called map(). On
the left you can see the traditional way of doing your lighting based on
gradients. To the right you can see the new way of performing lighting:

// map : SDF or density function // eps: differential unit, base on required LOD
vec3 calcNormal( in vec3 x, in float eps ) { vec2 e = vec2( eps, 0.0 ); return
normalize(vec3(map(x+e.xyy)-map(x-e.xyy), map(x+e.yxy)-map(x-e.yxy),
map(x+e.yyx)-map(x-e.yyx))); }

void render( void ) { // ... float den = map( pos ); vec3 nor = calcNormal( pos,
eps ); float dif = clamp(dot(nor,light),0.0,1.0); // ... } // map : SDF or
density function // eps: differential unit, based on required LOD void render(
void ) { // ... float den = map( pos ); float dif =
clamp((map(pos+eps*light)-den)/eps,0.0,1.0); // ... } If this code is called
hundreds or thousands of times during a raymarch process because it's core to
the volumetric raymarching process, then the gains can be massive, since the
traditional method requires 2 evaluations per point while the new method only
involves 7 evaluations per point. The code not only is 3.5 times faster, but
also smaller, which is great if you are doing some size-coding based demo or
shader.

Of course, the drawback is that this is only an advantage for a small number of
light sources. So computing the normal might be advantageous anyways after 3
or 4 light sources, which is the most likely scenario (for example, to lit
clouds you will want at least three light sources: the sun, the sky dome and the
bounce coming from the ground).

Here are some pictures that show the new directional derivative based lighting
versus the gradient based, and also to no lighting at all, for comparison.

No lighting:

Directional derivative based lighting (3.5 times faster):

Traditional gradient based lighting:

If you are trying to see the differences between the directional derivative and
the gradient method but don't spot them, that's because there are no
differences.

It's also worth mentioning that the technique works also for solid surfaces of
course, as illustrated in the first picture in this article. Here below I leave
another example of the technique used with a solid SDFs, showing how to use this
technique to get some lighting in a few bytes of code - click in the title to
access the source code:

inigo quilez - learning computer graphics since 1994

Inigo Quilez :: articles :: screen space ambient occlusion - 2007

Disclaimer - I wrote this article almost 20 years ago, and it was probably the
first public documentation exploring and describing the technique. Naturally the
technique has evolved since then (although its main problem still remains
unresolved), so read this article in its historical context.

SSAO stands for Screen Space Ambient Occlusion, and it partially makes reality
one of the deepest dreams of computer graphics programmers: ambient occlusion in
realtime, mine included (see here). The term was first used by Crytek when they
introduced it in a small paragraph of a paper called "Finding Next Gen:
CryEngine 2" (just google for that sentence).

Since then many CG programing enthusiast tried to decipher how the technique
works, and each one got different results with varying quality and performances.
I did my own investigations and I arrived to a method that being not optimal,
still gives cool results and is quite usable. Of course I went thru many
revisions of the algorithm, but well, this is the technique I used for
Kindernoiser and Kindercrasher 4 kilobyte demos.

Screen Space Ambient Occlusion term applied to a complex shape

The trick: Ambient occlusion, as other direct lighting techniques (and indirect
too of course) is based on a non-local computations. This means it's not enough
to know the surface properties of the point to be shaded, but one needs some
description of the surrounding geometry as well. Since this information is not
accessible on modern rasterization hardware (that's why we will never see good
realtime shadows in OpenglGL or Directx), the Crytek team (as many other guy
somehow before them) came with the idea to use the zbuffer to partially recover
such information. Zbuffer can be seen as a small repository of geometry
information: from each pixel on the buffer one can recover the 3d position of
the geometry (well, the closest to the camera surface) projected on that pixel.

Thus the idea is to use that information in a two (or more) pass algorithm.
First render the scene normally, or almost, and in a second full screen quad
pass compute the ambient occlusion at each pixel and use it to modify the
already computed lighting. For that, for each pixel for which we compute the AO
we construct few 3d points around it and see if these points are occluded from
camera's point of view. This is not ambient occlusion as in the usual
definition, but it indeed gives some kind of concavity for the shaded point,
what can be interpreted as an (ambient) occlusion factor.

To simplify computations on the second pass, the first pass outputs a linear eye
space z distance (instead of the 1/z used on the regular zbuffers). This is done
per vertex since z, being linear, can be safely interpolated on the surface of
the polygons. By using multiple render targets one can output this buffer at the
same time as the regular color buffer.

The second pass draws a screen space polygon covering the complete viewport and
performs the ambient occlusion computation. For that it first recovers the eye
space position of each pixel by unprojection: it reads the z value from the
previously prepared texture, and given the eye space view vector (computed by
interpolation from the vertex shader) it computes the eye space position. Say
gl_TexCoord[0] contains the eye view vector (not necessarily normalized), tex0
the linear zbuffer, and gl_Color the 2d pixel coordinates (from 0 to 1 for the
complete viewport), then:

float ez = texture2D( tex0, gl_Color.xy ); // eye z distance vec3 ep =
ez*gl_TexCoord[0].xyz/gl_TexCoord[0].z; // eye point

next we have to generate N 3d points. I believe Crytek uses 8 points for low end
machines (pixel shaders 2.0) and 16 for more powerful machines. It's a trade off
between speed and quality, so definitively a parameter to play with. I generated
the points around the current shading point in a sphere (inside the sphere, not
just on the surface) from a small random lookup table (passed as constants),
with a constant radius (scene dependent, and feature dependent - you can make
the AO more local or global by adjusting this parameter).

for( int i=0; i<32; i++ ) { vec3 se = ep + rad*fk3f[i].xyz;

Next we project these points back into clip space with the usual perspective
division and look up on the zbuffer for the scene's eye z distance at that
pixel, as in shadow mapping:

vec2 ss = (se.xy/se.z)*vec2(0.75,1.0);
vec2 sn = ss*0.5 + vec2(0.5);
vec4 sz = texture2D(tex0,sn);

or alternatively

vec3 ss = se.xyz*vec3(0.75,1.0,1.0);
vec4 sz = texture2DProj( tex0, ss*0.5+ss.z*vec3(0.5) );

Now the most tricky part of the algorithm comes. Unlike in shadow mapping where
a simple comparison yields a binary value, in this case we have to be more
careful because we have to account for occlusion, and occlusion factor are
distance dependent while shadows are not. For example, a surface element that is
far from the point under consideration will occlude less that point than if it
was closer, with a quadratic attenuation (have a look here). So this means it
should be a bit like a step() curve so that for negative values it does not
occlude, but it should then slowly attenuate back to zero also. The attenuation
factor, again, depends on the scale of the scene and aesthetical factors. I call
this function "blocking or occlusion function". The idea is then to accumulate
the occlusion factor, like:

float zd = 50.0*max( se.z-sz.x, 0.0 );
bl += 1.0/(1.0+zd*zd);

and to finish we just have to average to get the total estimated occlusion.

} gl_FragColor = vec4(bl/32.0);

The second trick: Doing it as just described creates some ugly banding artifacts
derived from the low sampling rate of the occlusion (32 in the example above, 8
or 16 in Crytek's implementation). So the next step is to apply some dithering
to the sampling pattern. Crytek suggests to use a per pixel random plane to do a
reflection on the sampling point around the shading point, what works very well
in practice and is very fast. For that we have to prepare a small random normal
map, accessible thru tex1 on the following modified code:

vec3 se = ep + rad*reflect(fk3f[i].xyz,pl.xyz);

so the complete shader looks like:

uniform vec4 fk3f[32]; uniform vec4 fres; uniform sampler2D tex0; uniform
sampler2D tex1;

void main( void ) { vec4 zbu = texture2D( tex0, gl_Color.xy );

vec3 ep = zbu.x*gl_TexCoord[0].xyz/gl_TexCoord[0].z;

vec4 pl = texture2D( tex1, gl_Color.xy*fres.xy );
pl = pl*2.0 - 1.0;

float bl = 0.0;
for( int i=0; i<32; i++ )
{
    vec3 se = ep + rad*reflect(fk3f[i].xyz,pl.xyz);

    vec2 ss = (se.xy/se.z)*vec2(0.75,1.0);
    vec2 sn = ss*0.5 + vec2(0.5);
    vec4 sz = texture2D(tex0,sn);

    float zd = 50.0*max( se.z-sz.x, 0.0 );
    bl += 1.0/(1.0+zd*zd);

} gl_FragColor = vec4(bl/32.0); }

The big secret trick is to apply next a blurring to the ambient occlusion, that
we stored in a texture (occlusion map). It's easy to avoid blurring across
object edges by checking the difference in z between the blurring sampling
points, and the eye space normal too (that we can output as with the eye linear
z distance on the very first pass).

Optimziations: The shader above does not execute in pixel shaders 2.0 hardware,
because of the amount of instructions, even with just 8 sampling points, while
Crytek's does. So, the thing is to simplify the inner loop code. The first this
one can do is to remove the perspective projection applied to the sampling
points. This has a nice side effect, and it's that the sampling sphere is
constant size in screen space regardless the distance to the camera, what allows
for ambient occlusion both in close and distant objects at the same time. That's
what Crytek guys do, I believe. Once could play with the blocking factor to
remove few instructions too.

Results: I added few reference images below of this realtime Screen Space
Ambient Occlusion implementation above:

inigo quilez - learning computer graphics since 1994

Inigo Quilez :: articles :: cppencapsulation - 2012

Intro So here's a little coding trick I've developed in 2012 to workaround the
lack of encapsulation in the C++ language. I've edited the article with some
more modern constructs because I still find the technique useful, and I hope
others do too.

As we know, C++ classes don't provide much encapsulation - many implementation
details leak to the user. Consider this class:

number_bucket.h

#pragma once

#include  #include 

class NumberBucket { public: NumberBucket(); ~NumberBucket();

void addNumber( int n );
int  pickNumber( void );

private: std::vector mNumbers; std::minstd_rand0 mRandoms; }; number_buket.cpp

#include  #include  #include  #include "number_bucket.h"

NumberBucket::NumberBucket():mNumbers(),mRandoms(12345) {}
NumberBucket::~NumberBucket() {}

void NumberBucket::addNumber( int n ) { mNumbers.push_back(n); }

int NumberBucket::pickNumber( void ) { const size_t size = mNumbers.size();
assert( size>=1 ); const size_t id = mRandoms() % size; return mNumbers[ id ]; }
Side notes - [1] this is not how you pick uniformly from a set of integers. [2]
I'd never write a class with constructors, I'm writing traditional C++ here only
to keep the conversation focused on encapsulation.

There are a few problems with this class. First, many implementation details and
data types are leaking. In particular code that uses NumberBucket needs to pull 
and  even though interacting with this class' API does not require any of those
data types. The involvement of vectors and random number generators are internal
implementation details of the class and should not have never concerned the
caller.

To make things worse, code that uses NumberBucket is also forced its own client
code to include  and . This dependency chain propagates quickly into a cascade
of include files, where data types leak upstream all the way to the root of the
project, making build times explode to 5 minutes for projects that with manual
encapsulation and basic include hygiene actually only take 5 seconds to build.
This is specially bad for std:: library, which is very leaky. Forward-declaring
classes can mitigate some issues, at the cost of introducing indirections, but
is still not proper encapsulation.

The problem of encapsulation is larger than just data type and include file
leakage though. Imagine NumberBucket used a plain mNumbers[] array instead of a
std::vector and some inlined mRandom generator instead of std::minstd_rand0. The
class would still reveal implementation details, such as the existence of
mNumbers and mRandom. This adds cognitive load when inspecting and exploring
this class' interface to learn how to interact with it. And also triggers
recompilation of all upstream code when modifying the private members of the
class, even though they do not affect the class API at all.

The traditional solution - PIMPL

The most common way to fight the lack of encapsulation in C++ is the PIMPL
pattern ("Pointer to IMPLementation"). PIMPL hides all implementation details
from the class' interface, including data types. This is an example of PIMPL
applied to the NumberBucket class above:

number_bucket.h

#pragma once

#include 

class NumberBucket { public: NumberBucket(); ~NumberBucket();

void addNumber( int n );
int  pickNumber( void );

private: struct Impl; std::unique_ptr mImpl; }; number_bucket.cpp

#include  #include  #include  #include "number_bucket.h"

struct NumberBucket::Impl { Impl():mNumbers(),mRandoms(12345) {} ~Impl() {}

std::vector<int>  mNumbers;
std::minstd_rand0 mRandoms;

};

NumberBucket::NumberBucket() : mImpl(std::make_unique()) {}
NumberBucket::~NumberBucket() {}

void NumberBucket::addNumber( int n ) { mImpl->mNumbers.push_back(n); }

int NumberBucket::pickNumber( void ) { const size_t size =
mImpl->mNumbers.size(); assert( size>=1 ); const size_t id = mImpl->mRandoms() %
size; return mImpl->mNumbers[ id ]; }

This works well and I used it a lot in the 2000s (with void * instead of
unique_ptr). However it has one undesirable feature - it requires an extra heap
allocation at initialization time, and an indirection for every object
operation. The heap allocation happens at unique_ptr construction time. If you
want an even cleaner API where  header is not leaked all over the place, you can
use a raw pointer and new()/delete(). This also has the benefit that you can
return errors on initialization instead of throwing from constructors ala RAII,
but your philosophy here might vary.

In my case, the reason I stopped using this vanilla variant of PIMPL is that the
heap allocation creates memory fragmentation. The actual meat of the
implementation operates in a memory location that is far from where the actual
class sits. This can create performance problems in many situations, which of
course can be solved by introducing a memory pool that manages all
IMPLementations instances of the same type. But at that point the amount of work
and boilerplate required to achieve performant encapsulation starts to feel
ridiculous. So here's where I landed in a different implementation of the PIMPL
pattern:

The modified solution - In-Place PIMPL So let's finally explore the solution I
landed on years ago: just use PIMPL but instead of allocating space in the heap
for our private implementation, do it in the class itself in an opaque block of
memory:

number_bucket.h

#pragma once

class NumberBucket { public: NumberBucket(); ~NumberBucket();

void addNumber( int n );
int  pickNumber( void );

private: int opaque[8]; }; number_bucket.cpp

#include  #include  #include  #include  #include "number_bucket.h"

struct Impl { Impl():mNumbers(),mGenerator(12345) {} ~Impl() {}

std::vector<int>  mNumbers;
std::minstd_rand0 mGenerator;

};

#define mImpl ((Impl*)opaque)

NumberBucket::NumberBucket() { new(opaque)Impl(); }
NumberBucket::~NumberBucket() { mImpl->~Impl(); }

void NumberBucket::addNumber( int n ) { static_assert(
sizeof(opaque)>=sizeof(Impl) ); mImpl->mNumbers.push_back(n); }

int NumberBucket::pickNumber( void ) { const size_t size =
mImpl->mNumbers.size(); assert( size>=1 ); const size_t id = mImpl->mGenerator()
% size; return mImpl->mNumbers[ id ]; }

As you can see, the member "int opaque[8]" in where we instantiate our private
implementation class. We do so by constructing/initializing it in place (with
placement new or otherwise). Internally, all the accesses to the private
implementation require casting the opaque blob to the implementation class, but
this does not result in an indirection at runtime. Instead this is just
accessing members as a fixed offset known at compile time, just like in regular
classes. I like using int[size/4] instead of char[size] for the opaque blob so I
ensure proper memory alignment.

Obviously the size of the opaque blob needs to be large enough to host the
implementation class. We make sure that's the case with the static_assert which
compares the size of the blob and the size of the implementation class. I often
check for equality to prevent wasting space, unless the class needs different
implementations in different platforms.

Lastly, there are no include files required, and as a result files only using
NumberBucket and classes written in the same idiom can now be compiled at the
speed of light.

But now let's address the major drawback of this strategy - the size of the
opaque block has to be maintained by hand in the class definition. You might
call this "a hack", but for me this has become more an idiom, and the pros of
this In-Place PIMPLE approach largely outweigh the cons: it achieves true
encapsulation (no include explosion, no implementation detail leaks) without
performance impact (no heaps, no indirections), and with less code and
complexity than PIMPL (no constructors and factories required).

Obviously I'd like C++ to provide encapsulation and not have to do any of this,
but I've been a fan of the technique since I developed it and I've used it in
professional products and hobby projects alike. If you want to have a cleaner
programs and shorter build times, give it a try!

inigo quilez - learning computer graphics since 1994

Inigo Quilez :: articles :: working with ellipses - 2006

Planar ellipses (I add "planar" to make it clear it's not a 3D ellipsoid) are
constructed by a 3D position c in space and two perpendicular orientation
vectors u and v that define both the plane where the ellipse lays and the size
of the axes. So you can store a 3D ellipse in 9 floats (8, if you are careful).
In the case of the ellipse being degenerated to a disk, then a single
orientation vector is needed (perpendicular to the plane containing the disk)
and a radious (so, 6 floats). Planar ellipses can become very useful for
computer graphics. For example, they appear when you cut a cylinder with a
plane. They also appear when rendering point clouds with a splatting algorithm,
or when raytracing point clouds. They can also help in realtime ambient
occlusion and indirect lighting computations, where occluders can be
approximated by a pointcloud.

Here I will show how to do two of the most basic operations on planar ellipses:
bounding box calculation and ray intersection (sounds like this is what you need
for a fast kd-tree based raytracer, uh?). Let's see the bounding box first:

The Bounding box of an ellipse computer analytically

Bounding Box

As we know any point in the ellipse boundary is described by the following
parametric equation:

p(ω) = c + u⋅cos(ω) + v⋅sin(ω)

With c, u and v defined as described above and as illustrated below:

The bounding box of the ellipse has to be tangent to this boundary. This
tangential points will be the maximum and minimum x, y and z coordinates of the
boundary equation. So, we need find the minima/maxima of the equation. We know
that we can get them by finding where the derivative equal zero. So,

p'(ω) = -u⋅sin(ω) + v⋅cos(ω)

must equal zero for each of the three coordinates. Let's rename first
λ=cos(ω)and solve for the x coordinate:

to get

meaning that

In summary, we can calculate our bounding box corners like this:

In (GLSL style) code this would become simply // disk :: c:center, u: 1st axis,
v: 2nd axis bound3 EllipseAABB( in vec3 c, in vec3 u, in vec3 v ) { vec3 e =
sqrt( uu + vv ); return bound3( c-e, c+e ); }

You can find the source code and realtime demo using this code here:
https://www.shadertoy.com/view/Xtjczw

Ready for building your acceleration kd-tree for point clouds?

Ray-ellipse intersection

Similarly to the ellipse border, the interior can be defined by

p(λ,γ) = c + u⋅λ + v⋅γ and λ2 + γ2 ≤ 1

where the equality holds for the border. Now, if we define the ray with the
equation

p(t) = ro + t⋅rd

where ro is the ray origin and rd is the (not necessarily normalized) ray
direction, then we must make both expressions equal in order to get the
intersection point, thus we must solve the equation ro + t⋅rd = c + u⋅λ + v⋅γ
that actually is a system of three equations (one for each of the x, y and z
coordinates) with three unknowns. Re-arranging these three equations we get the
following system:

We can solve this by Cramer's law:

Note that de will be zero when the ray is parallel to the plane containing the
ellipse, so that needs special care. Of course, t corresponds to the ray-plane
intersection distance. Finally, to test that the intersection point is in the
ellipse, check for

λ2 + γ2 ≤ 1

In code, this would mean

float iEllipse( in vec3 ro, in vec3 rd, // ray: origin, direction in vec3 c, in
vec3 u, in vec3 v ) // disk: center, 1st axis, 2nd axis { vec3 q = ro - c; vec3
r = vec3( dot( cross(u,v), q ), dot( cross(q,u), rd ), dot( cross(v,q), rd ) ) /
dot( cross(v,u), rd ); return (dot(r.yz,r.yz)<1.0) ? r.x : -1.0; }

You can find the source code in the same location as the bbox code:
https://www.shadertoy.com/view/Xtjczw

I hope all this is helpful for somebody :)

inigo quilez - learning computer graphics since 1994

Inigo Quilez :: articles :: don't flip, reflect and clip - 2014

Intro

So, there's this simple problem in computer graphics that I've encountered
multiple times: a vector that I've generated through some computation is facing
the wrong side of the hemisphere that I need it to be in. The naive way to fix
it is to simply flip or negate it so it falls in the correct hemisphere, but
that can create all sorts of problems.

Left: flipping normals introduces wrong colors (note brown pixels in dark
areas). Right: clipping normals keeps the image stable

Let me give you two examples where the fix fails:

Example 1: say we are dynamically generating procedural grass on the surface of
a rock. So, for many points on its surface we generate blades with random
length, thickness, stiffness, color and orientation. The orientation of each
blade needs to be in the outer hemisphere around the rock's surface normal, so
we generate a random point in space and if it's in the wrong side of the tangent
plane we flip it. So far so good, that's easy and the render looks great. But
when rendering the next frame because of the renderer's precision limitations
the rock's surface normal changes a tiny little bit. This happens more often
than you think with subdivision surfaces for example due to their view dependent
refinement, or with procedural meshes that are themselves the result of
computations, or with SDF objects, or even with plain static meshes when their
transforms have been baked. Whatever the reason, even though the normals look
stable to our naked eye, their numerical value can slightly change frame to
frame. And with it, the hemispheres of valid directions for our grass blades
change too. This means that some number of the blades of grass that got
generated in the correct hemisphere in the previous frame now fall in the wrong
hemisphere in the current frame, and need a flip. Suddenly we have an animation
with flicker and popping errors. Now, while this is infrequent in a shot with
millions of blades of grass it is guaranteed it's going to happen to a few
blades... and we only need a single one them to be visible to the camera to ruin
the whole shot.

Example 2: imagine we are rendering an SDF through raymarching in GLSL, and we
have computed normals for it through finite differences. We are now performing
lighting but we notice that every now and then the lighting blows up and gets a
white pixel in the image, especially around object silhouettes. We realize it's
due to normals facing away from the ray direction, so we force them to look
towards the camera with faceforward(), which essentially flips the normal. The
white pixels are gone, but their color is still not right, because we've
effectively turned the surface inside out, so we are getting totally picking the
wrong lighting. And we do not like wrong pixels if we can have the correct
pixels.

The problem

So, the problem in that vector flipping or negation is no a continuous map. If
we look at the naive solution

// flip v if it's in the negative half plane defined by r vec flipVec( vec v,
vec r ) { float k = dot(v,r); return (k>0.0) ? v : -v; }

we see that even the tiniest change in v or r can send v all the way in the
opposite direction. So, what we need to do is find a method to bring v to the
correct hemisphere through a function that is continuous rather than
discontinuous. Luckily there are two easy ways to do it:

Solution

The simplest solution is to reflect the vector v along the plane defined by r.
This works and produces a continuous transformation because a vector that is
very close to the plane separating the two hemispheres will reflect to the other
side into a location that is also very close to the plane. In other words, small
changes in the input always produce small changes in the output, which is what
brings stability to this approach. This is the code that implements it (I left
the dimensionality of the vector types unspecified since this works in any
number of dimensions):

// reflect v if it's in the negative half plane defined by r vec reflVec( vec v,
vec r ) { float k = dot(v,r); return (k>0.0) ? v : v-2.0rk; }

Alternatively, if for some reason you prefer to snap v to the half-plane, you
can use the following function, although it is a bit slower than the reflection:

// clip v if it's in the negative half plane defined by r vec3 clipVec( vec v,
vec r ) { float k = dot(v,r); return (k>0.0) ? v :
(v-r*k)inversesqrt(1.0-kk/dot(v,v)); }

Note that if v is normalized, the expression simplifies a little bit and you can
remove the division by dot(v,v). And/or if you only need to catch vectors that
penetrate slightly in the negative hemisphere then you can assume kk≈0 and
replace inversesqrt(1-kk) by it's Taylor approximation 1.0-0.5kk, or even by 1.
You can also remove the inversesqrt() completely if you plan to normalize v
after the clip rather than before, leaving you with:

// clip v if in the negative half plane defined by r. NOTE - doesn't preserve
length vec clipVecNoLength( vec v, vec r ) { float k = dot(v,r); return (k>0.0)
? v : v-r*k; }

Conclusion

The following animation is a comparison of the three methods. The yellow arrows
are the result of applying the hemisphere fix to the grey arrow. Pay attention
to the movement of the yellow arrows, and note how the "Flip" version jumps from
one quadrant to the other when the arrow gets into the negative hemisphere. The
"Reflect" and "Clip" in the other hand never jump, which is exactly what we
want:

You can see both techniques in action and the reference code in the following
realtime shader in Shadertoy: https://www.shadertoy.com/view/4dBXz3

inigo quilez - learning computer graphics since 1994
