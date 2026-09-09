# 3D Point Cloud Rendering: Shading and Shadows

This document analyzes the approaches for rendering a large, dynamically updating 3D point cloud (50 million points at 60Hz, with up to 1M points/sec insertion rate) with shading and shadows. It also details the implementation of a hybrid Screen-Space Shadows (SSS) and Eye-Dome Lighting (EDL) shader in [gen2.lisp](file:///home/kiel/stage/cl-cpp-generator2/example/197_shadertoy/gen2.lisp).

---

## 1. Comparison of Shadow Rendering Approaches

To render shadows directly on a 3D point cloud, three main approaches exist. Their characteristics under the scanner's performance and update constraints are summarized below:

| Approach | Rendering Overhead | Complexity | Dynamic Update Suitability | Handling of Gaps/Holes | Recommended? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Screen-Space Shadows (SSS)** | **Extremely Low** (Post-processing) | Medium | **Excellent** (No CPU/GPU rebuilds needed) | Natural interpolation in screen-space | **Yes (Highly Recommended)** |
| **Shadow Mapping** | **High** (Requires rendering all 50M points twice) | Low | Medium (Must re-render depth map) | Requires splat size adjustment to prevent holes | No |
| **Voxel Grid / Raymarching** | **High** (Voxel grid construction & raymarching) | High | Poor (Updating voxel tree at 500Hz is slow) | Natural (voxel density) | No |

### Detailed Evaluation

#### A. Screen-Space Shadows (SSS)
* **Concept**: First, render the point cloud normally to populate the depth and color framebuffer. Then, run a screen-space post-processing pass. For each pixel, reconstruct its 3D position from depth and raymarch in screen space towards the light source, checking for depth occlusion.
* **Why it fits the 50M Point Scanner**:
  * You only render the point cloud **once**. The shadow raymarching is done entirely in screen space and its cost is completely decoupled from the number of points (it only depends on screen resolution and ray steps).
  * It integrates seamlessly with your existing screen-space EDL shader! You can do normal estimation, EDL outlining, and SSS tracing in the same post-processing pass using the same depth texture.

#### B. Shadow Mapping
* **Concept**: Render the point cloud from the light source's perspective into a depth texture (shadow map), then render it from the camera's perspective, checking if each point projects behind the shadow map.
* **Limitations**:
  * **Double Vertex Load**: You have to pass the 50 million points through the vertex pipeline twice per frame (once for the shadow map, once for the screen). This is likely to drop the framerate below 60Hz.
  * **Holes**: Point clouds are not solid. Gaps between points will cause the light's depth pass to leak through, resulting in "shadow acne" or missing shadows. To prevent this, you would need to render points as larger splats/surfels, further increasing fragment shader overhead.

#### C. Voxel Grid / SVO Raymarching
* **Concept**: Voxelize the point cloud into a 3D texture or Sparse Voxel Octree (SVO), then raymarch the voxels.
* **Limitations**: Rebuilding or updating a 3D texture or octree at 500Hz while inserting 2000 points per step is extremely memory-intensive and will introduce severe CPU-to-GPU bandwidth bottlenecks.

---

## 2. Hybrid SSS + EDL Implementation in `gen2.lisp`

To demonstrate the hybrid Screen-Space Shadows + Eye-Dome Lighting pipeline, we created [gen2.lisp](file:///home/kiel/stage/cl-cpp-generator2/example/197_shadertoy/gen2.lisp). The shaders generated are:

1. [buf0.glsl](file:///home/kiel/stage/cl-cpp-generator2/example/197_shadertoy/vulkan-shadertoy-x11/launcher/shaders/shadertoy/buf0.glsl): Simulates the point cloud rendering pass. It renders a solid checkerboard ground plane and rasterizes a rotating 3D Fibonacci sphere point cloud (800 points), combining them in the color/depth texture buffer (iChannel0).
2. [main_image.glsl](file:///home/kiel/stage/cl-cpp-generator2/example/197_shadertoy/vulkan-shadertoy-x11/launcher/shaders/shadertoy/main_image.glsl): The post-processing pass. It reads the color and depth buffer, performs screen-space reconstruction, normal estimation, diffuse shading, screen-space raymarched shadows, and EDL outlining.

Here are the key algorithms implemented in S-Expressions:

### A. 3D Position Reconstruction
To compute shading and trace shadow rays, we reconstruct the 3D camera-space position $P$ of each pixel using its screen coordinate and depth value:
```lisp
(defun reconstructP (pixel depth)
  (declare (type ivec2 pixel)
           (type float depth)
           (values vec3))
  (let (uv proj)
    (declare (type vec2 uv proj))
    ;; Centered and aspect-ratio corrected UV coordinates
    (setf uv (/ (- (vec2 pixel) (* 0.5f0 (dot iResolution xy))) (dot iResolution y))
          proj (* uv 2.0f0))
    (return (vec3 (* proj depth) depth))))
```

### B. Screen-Space Normal Estimation
Since raw point clouds do not have surface normals, we estimate the normal vector at each pixel using the spatial derivatives of the reconstructed 3D positions of its neighbors:
```lisp
;; Reconstruct 3D positions of the right and up neighbors
(setf depth_R (dot (texelFetch iChannel0 (+ (ivec2 fragCoord) (ivec2 1 0)) 0) w)
      depth_U (dot (texelFetch iChannel0 (+ (ivec2 fragCoord) (ivec2 0 1)) 0) w))
(if (> depth_R 10000.0f0) (setf depth_R depth))
(if (> depth_U 10000.0f0) (setf depth_U depth))

(setf P_R (reconstructP (+ (ivec2 fragCoord) (ivec2 1 0)) depth_R)
      P_U (reconstructP (+ (ivec2 fragCoord) (ivec2 0 1)) depth_U)
      dPdx (- P_R P)
      dPdy (- P_U P)
      normal_cross (cross dPdx dPdy))
;; Avoid division-by-zero artifacts
(if (< (length normal_cross) 0.0001f0)
    (setf normal (vec3 0.0f0 0.0f0 -1.0f0))
    (setf normal (normalize normal_cross)))
```

### C. Screen-Space Raymarched Shadows (SSS)
From the point's 3D position $P$, we step along a 3D ray pointing towards the light source. At each step, we project the 3D position back to the screen and compare its depth with the depth stored in the buffer:
```lisp
(let ((ray_dir (normalize (- light_pos P)))
      (light_dist (length (- light_pos P)))
      (t_max (min light_dist 5.0f0))
      (steps 24))
  (declare (type float light_dist t_max)
           (type vec3 ray_dir)
           (type int steps))
  (setf shadow_factor 1.0f0)
  
  (for ("int step_idx = 1" (<= step_idx steps) (incf step_idx))
    (let (tVal P_curr proj_curr uv_curr pixel_curr map_depth)
      (declare (type float tVal map_depth)
               (type vec3 P_curr)
               (type vec2 proj_curr uv_curr)
               (type ivec2 pixel_curr))
      ;; Step forward along the 3D ray
      (setf tVal (* (/ (float step_idx) (float steps)) t_max)
            P_curr (+ P (* ray_dir tVal))
            ;; Project back to normalized screen space
            proj_curr (/ (dot P_curr xy) (dot P_curr z))
            uv_curr (* proj_curr 0.5f0)
            pixel_curr (ivec2 (+ (* uv_curr (dot iResolution y)) (* 0.5f0 (dot iResolution xy)))))
      
      ;; Stop ray if it goes off-screen
      (when (logior (< (dot pixel_curr x) 0) (>= (dot pixel_curr x) (dot iResolution x))
                    (< (dot pixel_curr y) 0) (>= (dot pixel_curr y) (dot iResolution y)))
        break)
      
      ;; Read buffer depth
      (setf map_depth (dot (texelFetch iChannel0 pixel_curr 0) w))
      
      ;; If the ray is behind a rendered point (occlusion), apply shadow factor and stop
      (when (logand (< map_depth 1000.0f0) (> (dot P_curr z) (+ map_depth 0.08f0)))
        (setf shadow_factor (- 1.0f0 shadow_strength))
        break))))
```

### D. Eye-Dome Lighting (EDL) Outlining
In the same shader, we run EDL to enhance the edges of the points. We compare the depth of the current pixel with its 4-neighborhood offsets:
```lisp
(let ((sum 0.0f0)
      (edl_radius 2.0f0))
  (declare (type float sum edl_radius))
  "vec2 offsets[4] = vec2[](vec2(0.0f, 1.0f), vec2(0.0f, -1.0f), vec2(1.0f, 0.0f), vec2(-1.0f, 0.0f));"
  (for ("int idx = 0" (< idx 4) (incf idx))
    (let (neighborDepth)
      (declare (type float neighborDepth))
      (setf neighborDepth (dot (texelFetch iChannel0 (clamp (ivec2 (+ (ivec2 fragCoord) (ivec2 (* (aref offsets idx) edl_radius)))) (ivec2 0) (- (ivec2 (dot iResolution xy)) 1)) 0) w))
      (when (> neighborDepth 10000.0f0)
        (setf neighborDepth depth))
      (setf sum (+ sum (max 0.0f0 (- depth neighborDepth))))))
  ;; Apply EDL shadow term
  (setf col (* col (exp (* (- sum) 150.0f0 edl_strength)))))
```

---

## 3. Adaption Guide for Your Scanner Software

To integrate Screen-Space Shadows into your scanner pipeline:

1. **Keep Your Current Rendering Pipeline**:
   Keep rendering your point cloud to a Framebuffer Object (FBO) containing a color texture and a floating-point depth texture (e.g., format `GL_DEPTH_COMPONENT32F`).
2. **Modify the Post-Processing Shader**:
   Currently, your post-processing shader only does EDL on the depth texture. Update it to include:
   * **Reconstruction**: Convert depth to view-space/camera-space position $P$ using the inverse projection matrix or the fast mathematical projection approach shown above.
   * **Normal Estimation**: Approximate normals using derivative of positions of neighbor pixels.
   * **Raymarch Shadows**: Step along a ray in view-space from $P$ towards the light position (transformed to view-space). Project the stepped position back into screen space, sample your depth texture, and check if the ray depth is greater than the sampled depth.
3. **Control Parameters**:
   Expose parameters like `shadow_strength` and `bias` to fine-tune the shadows depending on point size and scan noise.
