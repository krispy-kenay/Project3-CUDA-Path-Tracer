**University of Pennsylvania, CIS 5650: GPU Programming and Architecture, Project 3**

* Yannick Gachnang
  * [LinkedIn](https://www.linkedin.com/in/yannickga/)
* Tested on: Windows 10, EPYC 9354 @ 3.25GHz, 16GB RAM, RTX 2000 Ada 23GB VRAM

Project 3 CUDA Path Tracer
====================

## Basic Features

### Diffuse BSDF

<br><br>

---

### Stream Compaction

This path tracer implements stream compaction which moves active rays to the front of the array and dead rays to the back, allowing later kernel launches to process only the active rays rather than the full array.

---
<br><br>

### Material Sorting

---
<br><br>

### Anti-Aliasing

---
<br><br>

## Visual Features

### Specular and Refractive Materials

This path tracer implements specular reflective materials and basic refractive materials (no glossy materials or roughness).

#### Implementation

For specular reflection, the standard mirror reflection formula to compute the outgoing direction as a pure reflection about the surface normal are used.
For refractive materials, Snell's law with Fresnel reflection coefficients computed via Schlick's approximation are used.
At each surface interaction, the code chooses between reflection and refraction based on the Fresnel term and when total internal reflection occurs the ray reflects specularly.
Otherwise it refracts according to the ratio of refractive indices.

Both specular and refractive interactions are marked as delta distributions for the MIS implementation later.


| Diffuse Sphere | Refractive Sphere | Specular Sphere |
| --- | --- | --- |
| ![Diffuse sphere](img/baseline_simple.png) | ![Refractive sphere](img/baseline_refractive.png) | ![Specular sphere](img/baseline_specular.png) |

#### Performance

The frametimes for the different materials for this simple scene are within 5%, with specular doing marginally better probably because of its deterministic, non-branching computation path.
Scaling this up to 3 scenes, reveals that they all scale at approximately the same rate for increasing scene complexity.
This is the expected behavior, since in all cases the ray gets manipulated (scattered, reflected, transmitted) and then continues onward.
Performance is expected to get worse as more geometry is added into a scene and it appears that the material type does not substantial affect this scaling.

![Material Performance Differences](img/perf_materials.png)
<br>
<sub>*Bar chart showing the performance differences between the material types and how they scale with adding more objects of the same type*</sub>

<details>
  <summary>Raw Data</summary>
  

| Number of spheres | Diffuse (ms/frame) | Refractive (ms/frame) | Specular (ms/frame) |
|-------|-------------------|---------------------|---------------------|
| 1     | 22.70             | 23.15               | 22.66               |
| 2     | 23.547            | 24.01               | 23.51               |
| 3     | 24.531            | 25.02               | 24.42               |

</details>

#### GPU vs CPU

Diffuse shading only needs to accumulate the color throughput and calculate a new direction for the ray which is easily parallelizable with little memory contention. The two new material type have the same scope shading work, so aside from the divergence problems mentioned above they benefit similarly from parallel execution. A CPU implementation would process these rays sequentially and lose the massive throughput advantage and even the branching problems on the GPU are greatly overshadowed by processing thousands of rays at the same time.

#### Further Optimizations

While scaling in a simple scene appears to not be affected by material type, there are cases where this can be problematic. As long as shading is done in a single monolithic kernel, there will be divergence between the threads due to different branching conditions that could cause additional slowdown. 
This can be counteracted by sorting the rays by material type before shading or using a queue based system where each material gets sorted into its own queue (wavefront). Both of these will be investigated further below.

Implementing glossy and rough specular materials would greatly enhance the variety of scenes that can be represented, though it would also introduce additional calculations and branching that could negatively impact performance

---
<br><br>

### Sampling Methods

This path tracer implements stratified sampling with cranley-patterson rotation and low-discrepancy Sobol sequences, both of which aim to reduce variance and improve sample distribution.
Uniform random sampling can sometimes struggle with subtle low-contrast features like penumbras, an area that both methods aim to address.

#### Implementation

Stratified sampling divides each sample dimension into temporal strata across iterations rather than spatial strata within a single pixel (since we're shooting one ray per pixel per iteration).
Each iteration draws from a different stratum in the sequence, with Cranley-Patterson rotation applied per-pixel to prevent structured patterns from forming across the image.
This ensures more even coverage of the sample space over time.

Sobol sampling is a quasi-random low-discrepancy sequence that theoretically provides better convergence than pseudorandom sampling.
The same Sobol sampling is also used in the diffuse shader for hemisphere sampling and in shadow ray generation for light sampling, so the toggle affects sampling throughout the entire path tracer.

For the random number generator, the `thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);` was upgraded to an xorshift32 based generator for slightly faster random sampling.

| Uniform random sampling | Stratified + Sobol sampling |
| --- | --- |
| ![Sobol off](img/sobol_off.png) | ![Sobol on](img/sobol_on.png) |

#### Performance

The performance impact is negligible as all sampling strategies stay within 1% frametime of each other at various sample counts. Therefore, as long as Sobol (+ stratified) sampling generally produces an equivalent or better result than uniform random sampling, it can always be enabled without a meaningful negative impact.

More interestingly, variance calculations also show no meaningful difference between Sobol sampling and uniform random sampling (<1% difference) in most cases.
This can likely be attributed to the simplicity of most of the tested scenarios.
Only in more complicated scene setups do some of the subtle differences show up like the penumbra becoming slightly more visible in the pictures above.
This particular scene was setup on purpose to produce soft shadows with an area light to highlight where Sobol (+ stratified) sampling can help, but also showcases that this feature becomes more important as scene complexity increases.

#### GPU vs CPU

Both sampling methods work well on GPU. The sampling computation is uniform across all rays with no divergence. A CPU implementation would be slower due to sequential processing with no architectural advantage.

#### Further Optimizations

Adaptive sampling which concentrates samples in high-variance regions or other spatially aware sampling methods would likely much more explicitly lower variance faster than Sobol sampling.

---
<br><br>

### Depth of Field

This path tracer implements various real world camera properties like f-stop, focal length and focal distance to simulate depth of field (colloquially known as bokeh effect).

#### Implementation

Each camera ray originates from a randomly sampled point on a circular aperture rather than a single pinhole.
The code first computes the standard pinhole ray direction, then calculates where this ray intersects the focal plane.
The ray origin then gets offset to a point on the lens, and the direction gets recomputed to pass through the focal point.
The f-stop controls the lens radius, with lower f-numbers producing shallower depth of field. The focal length values were scaled arbitrarily for this demonstration.
With proper physical units the scaling factor would need to be adjusted to match real-world camera behavior, but the f-stop values themselves don't really mean anything with unitless scene dimensions.

| Pinhole Camera | f/8 | f/2.8 |
| --- | --- | --- |
| ![DOF baseline](img/intermed_simple.png) | ![DOF f8](img/intermed_dof_f8.png) | ![DOF f2.8](img/intermed_dof_f2-8.png) |

#### Performance

The depth of field calculation introduces negligible overhead with a <1% measured performance difference, likely because the additional per-ray calculations are lightweight compared to ray-scene intersection and shading.

#### GPU vs CPU

Every ray undergoes identical lens sampling calculations with no branching, which results in perfect coherence across warps. The math is simple and uniform so the GPU can process all rays in parallel. A CPU version would offer no advantages and would be significantly slower due to sequential processing.

#### Further Optimizations

More sophisticated lens models (thick lens simulation or realistic optical systems) could enhance realism, but would incur slightly increased computational cost.

---
<br><br>

## Mesh Improvements

### OBJ File Loading

---
<br><br>

## Performance Features

### Russian Roulette

---
<br><br>

### Hierarchical Spatial Data Structure

---
<br><br>

### Wavefront Path Tracing

---
<br><br>
