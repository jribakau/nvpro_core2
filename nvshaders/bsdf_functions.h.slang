/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NVVKHL_BSDF_FUNCTIONS_H
#define NVVKHL_BSDF_FUNCTIONS_H 1

/*-------------------------------------------------------------------------------------------------
> Implements bidirectional scattering distribution functions (BSDFs) for
> physically-based rendering systems.

To use this code, include `bsdf_functions.h`. When a ray hits a surface:
* Create a `PbrMaterial` describing the material at that point
* Call `bsdfEvaluate()` to evaluate the scattering from one direction to
  another, or `bsdfSample()` to choose the next ray in the light path.
  See these functions' documentation for more information on their parameters
  and return values.

See the vk_gltf_renderer and vk_mini_samples/gltf_raytrace samples for examples
where these functions are used in a Monte Carlo path tracer.

The returned BSDF values and weights have the cosine term from the rendering
equation included; e.g. the Lambert lobe returns `max(0.0f, cos(N, k1)) / pi`.

Advanced users can use some of the other functions in this file to evaluate
and sample individual lobes, or subsets of lobes.

`bsdfEvaluate()` picks one lobe at random, as `bsdfSample()` does, and returns
that lobe's own pdf; the two are a pair for Multiple Importance Sampling.
`bsdfEvaluateAll()` evaluates every lobe, each by its weight (the mean of
`bsdfEvaluate()` over `xi.z`), and returns the lobe mixture's pdf; its MIS
partner is `bsdfSampleMixture()` (or `bsdfSampleCore(..., true)`), which samples
exactly as `bsdfSample()` does but reports that mixture pdf. Mixing one of one
pair with one of the other biases the estimate.

This file also provides `bsdfEvaluateSimple()` and `bsdfSampleSimple()`, which
implement a simpler and faster, though less fully-featured BSDF model. The
simple model only has diffuse, specular, and metallic lobes, while the full
model includes diffuse, transmission, specular, metal, sheen, and clearcoat
lobes (plus support for most glTF extensions).

Since GLSL doesn't have a distinction between public and private functions,
only functions that are part of the "public API" will have annotations to
appear in README.md.

This file (and files that it depends on) should be a GLSL/C++ polyglot, and
work so long as GLM has been included:

```cpp
#undef GLM_FORCE_XYZW_ONLY
#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>
using namespace glm;

#include <nvshaders/bsdf_functions.h>
```

## Technical Notes

NVVK's BSDFs are based on [the glTF 2.0 specification](https://github.com/KhronosGroup/glTF) and
[the NVIDIA MDL SDK's BSDF implementations](https://github.com/NVIDIA/MDL-SDK/blob/203d5140b1dee89de17b26e828c4333571878629/src/mdl/jit/libbsdf/libbsdf.cpp). 

The largest divergence from the above is that NVVK's BSDF model uses a
Fresnel term that depends only on the view and normal vectors, instead of the
half vector. This allows it to compute weights for lobes independently, while
BSDF code would normally need to sample a half vector for layer `i` to
determine the Fresnel weight for layer `i+1`. This can result in slightly
different glossy/diffuse blend weights (e.g. slightly differently shaped
highlights on leather surfaces).

All lobes are energy conserving (their integral over the sphere is at most 1)
and probability distribution functions (PDFs) integrate to 1 (except for areas
where the sampled direction results in an absorption event).

All lobes use single-scattering BSDFs. Multiple-scattering lobes are a
potential future improvement.

Most lobes use GGX normal distribution functions (NDFs) and the uncorrelated
Smith shadowing-maskiung function, except for the diffuse lobe (Lambert BRDF)
and the sheen lobe ([Conty and Kulla's "Charlie" sheen](https://blog.selfshadow.com/publications/s2017-shading-course/#course_content)
and a V-cavity shadowing-masking function).

-------------------------------------------------------------------------------------------------*/

#include "slang_types.h"  // NVSHADERS_INOUT_TYPE

#include "functions.h.slang"           // cosineSampleHemisphere
#include "pbr_ggx_microfacet.h.slang"  // brdfLambertian, ..
#include "ggx_albedo_table.h.slang"
#include "bsdf_types.h.slang"
#include "pbr_material_types.h.slang"


#ifdef __cplusplus
#define ARRAY_TYPE(T, N, name) std::array<T, N> name
#elif defined(GLSL)
#define ARRAY_TYPE(T, N, name) T name[N]
#endif

#ifndef ARRAY_TYPE
#define ARRAY_TYPE(T, N, name) T name[N]
#endif


/*-------------------------------------------------------------------------------------------------
# Function `volumeExtinctionCoefficient`
> Returns the volumetric extinction coefficient `sigma_t = -log(c) / d` of the material, per
> KHR_materials_volume / KHR_materials_scatter.

Per spec: when only `KHR_materials_volume` is active, scattering is assumed to be zero
(`sigma_s = 0`), so `sigma_t = sigma_a`. When `KHR_materials_scatter` is also active, the
same formula `-log(c)/d` continues to give the **total** extinction (`sigma_t = sigma_a +
sigma_s`); the scattering portion is recovered as `sigma_s = sigma_t * rho_ss`. Beer's-law
transmittance over a path of length x is therefore `T(x) = exp(-sigma_t * x)` regardless of
whether scattering is enabled.

Caller intent: use this for the extinction in Beer's law, delta tracking, and any per-channel
attenuation. Do NOT add `pbrMat.scatterCoefficient` on top -- it is already included in this
result.
-------------------------------------------------------------------------------------------------*/
float3 volumeExtinctionCoefficient(PbrMaterial mat)
{
  // one definition (pbr_material_types.h.slang); its distance floor is a micrometre, so a medium
  // measured in tenths of a millimetre (skin) keeps its density
  return volumeExtinctionCoefficient(mat.attenuationColor, mat.attenuationDistance);

}

/*-------------------------------------------------------------------------------------------------
# Function `absorptionCoefficient` (deprecated alias)
> DEPRECATED: kept for backward compatibility. The function name is misleading -- it actually
> returns the **extinction** coefficient `sigma_t`, not the absorption coefficient `sigma_a`.
> They coincide only when scattering is disabled (KHR_materials_volume without
> KHR_materials_scatter). New code should call `volumeExtinctionCoefficient` instead.
-------------------------------------------------------------------------------------------------*/
float3 absorptionCoefficient(PbrMaterial mat)
{
  return volumeExtinctionCoefficient(mat);
}

/*-------------------------------------------------------------------------------------------------
# Function `applyIorToRoughness`
> Scales transmission roughness by IOR, per KHR_materials_transmission.

Maps the incoming roughness so that ior=1.0 produces no microfacet refraction and ior=1.5
produces the default amount. Matches the Khronos reference renderer's `functions.glsl`.
-------------------------------------------------------------------------------------------------*/
float applyIorToRoughness(float roughness, float ior)
{
  return roughness * clamp(ior * 2.0F - 2.0F, 0.0F, 1.0F);
}

/*-------------------------------------------------------------------------------------------------
# Function `applyVolumeAttenuation`
> Beer's law over a transmissive segment, per KHR_materials_volume.

`attenuationDistance == 0.0` means infinite attenuation distance (no absorption). Otherwise the
returned radiance is `attenuationColor ^ (distance/attenuationDistance) * radiance`.
-------------------------------------------------------------------------------------------------*/
float3 applyVolumeAttenuation(float3 radiance, float distance, float3 attenuationColor, float attenuationDistance)
{
  if(attenuationDistance == 0.0F)
    return radiance;
  float3 transmittance = pow(attenuationColor, float3(distance / attenuationDistance));
  return transmittance * radiance;
}

/*-------------------------------------------------------------------------------------------------
# Function `henyeyGreenstein`
> Henyey-Greenstein phase function for volumetric scattering.

Parameters:
- cosTheta: cosine of the angle between incident and scattered directions
- g: anisotropy parameter in (-1, 1). g > 0 = forward scatter, g < 0 = back scatter, g = 0 = isotropic
Returns the phase function value (probability density).
Ref: https://www.pbr-book.org/4ed/Volume_Scattering/Phase_Functions#TheHenyeyndashGreensteinPhaseFunction
-------------------------------------------------------------------------------------------------*/
float henyeyGreenstein(float cosTheta, float g)
{
  float g2    = g * g;
  float denom = 1.0 + g2 - 2.0 * g * cosTheta;
  return M_1_PI * 0.25 * (1.0 - g2) / (denom * sqrt(denom));
}

/*-------------------------------------------------------------------------------------------------
# Function `sampleHenyeyGreenstein`
> Samples a direction from the Henyey-Greenstein phase function distribution.

Parameters:
- xi: two random numbers in [0,1]
- g: anisotropy parameter in (-1, 1)
- wi: normalized incident direction (the direction light/ray is traveling)

Returns the sampled scattered direction.
For forward scattering (g > 0), result tends to align with wi (continue forward).
For back scattering (g < 0), result tends to point opposite to wi (bounce back).
Ref: https://www.pbr-book.org/4ed/Volume_Scattering/Phase_Functions#TheHenyeyndashGreensteinPhaseFunction
-------------------------------------------------------------------------------------------------*/
float3 sampleHenyeyGreenstein(float2 xi, float g, float3 wi)
{
  float cosTheta;
  if(abs(g) < 0.001)
  {
    // Isotropic scattering
    cosTheta = 1.0 - 2.0 * xi.x;
  }
  else
  {
    float t  = (1.0 - g * g) / (1.0 - g + 2.0 * g * xi.x);
    cosTheta = (1.0 + g * g - t * t) / (2.0 * g);
  }

  float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
  float phi      = 2.0 * M_PI * xi.y;

  // Build orthonormal basis around wi (the direction light is traveling)
  // For forward scattering (g > 0), result will be close to wi
  float3 u, v;
  orthonormalBasis(wi, u, v);

  // Convert spherical to Cartesian in the local frame
  return sinTheta * cos(phi) * u + sinTheta * sin(phi) * v + cosTheta * wi;
}

/*-------------------------------------------------------------------------------------------------
# Function `henyeyGreensteinPdf`
> Returns the PDF for sampling a direction using the Henyey-Greenstein phase function.

This is the same as henyeyGreenstein() since sampleHenyeyGreenstein() does perfect sampling and 
henyeyGreenstein() is already normalized.
-------------------------------------------------------------------------------------------------*/
float henyeyGreensteinPdf(float cosTheta, float g)
{
  float pdf = henyeyGreenstein(cosTheta, g);
  // Near |g| = 1, HG approaches a delta distribution;
  // catch numerical edge cases (0, NaN, inf) to avoid division by zero or NaN.
  if(pdf == 0.0f || isnan(pdf) || isinf(pdf))
    pdf = DIRAC;
  return pdf;
}

/*-------------------------------------------------------------------------------------------------
# `LOBE_*` Defines
> Indices for lobe weights returned by `computeLobeWeights()`.
-------------------------------------------------------------------------------------------------*/
#define LOBE_DIFFUSE_REFLECTION 0
#define LOBE_SPECULAR_TRANSMISSION 1
#define LOBE_SPECULAR_REFLECTION 2
#define LOBE_METAL_REFLECTION 3
#define LOBE_SHEEN_REFLECTION 4
#define LOBE_CLEARCOAT_REFLECTION 5
#define LOBE_DIFFUSE_TRANSMISSION 6
#define LOBE_COUNT 7

// KHR_materials_retroreflection (MRM) excludes transmission, clearcoat, and sheen from the
// mirrored-V blend -- only the base metal/dielectric specular lobes participate.
bool isRetroreflectionLobe(int lobe)
{
  return lobe == LOBE_DIFFUSE_REFLECTION || lobe == LOBE_SPECULAR_REFLECTION || lobe == LOBE_METAL_REFLECTION;
}

// The Fresnel factor depends on the cosine between the view vector k1 and the
// half vector, H = normalize(k1 + k2). But during sampling, we don't have k2
// until we sample a microfacet. So instead, we approximate it.
// For a mirror surface, we have H == N. For a perfectly diffuse surface, k2
// is sampled in a cosine distribution around N, so H ~ normalize(k1 + N).
// We ad-hoc interpolate between them using the roughness.
float fresnelCosineApproximation(float VdotN, float roughness)
{
  return lerp(VdotN, sqrt(0.5F + 0.5F * VdotN), sqrt(roughness));
}

// How a reflection lobe weighs its microfacets (brdf_ggx_smith_eval / _sample `fresnel`)
#define BSDF_FRESNEL_NONE 0        // not at all: the lobe weights carry it
#define BSDF_FRESNEL_DIELECTRIC 1  // ior_fresnel(eta, k1.h), the exact dielectric's
#define BSDF_FRESNEL_SCHLICK 2     // Schlick's from the tint (a metal's base colour) to white at k1.h

/*-------------------------------------------------------------------------------------------------
# Function `ggxAlbedoScaleBias`
> The directional albedo of the GGX reflection lobe below (height-correlated Smith) of roughness
> `alpha` at view cosine `NoV`, with Schlick's Fresnel from f0 to 1, as f0 * x + y: the baked table
> (ggx_albedo_table.h.slang), bilinear in perceptual roughness and sqrt(NoV). The fit of Ray Tracing
> Gems chapter 32 it replaces was the uncorrelated lobe's, and up to 0.27 off the correlated one at
> grazing.
-------------------------------------------------------------------------------------------------*/
float2 ggxAlbedoScaleBias(float alpha, float NoV)
{
  const int   last = GGX_ALBEDO_TABLE_RES - 1;
  const float x    = sqrt(clamp(alpha, 0.0F, 1.0F)) * float(last);
  const float y    = sqrt(clamp(NoV, 0.0F, 1.0F)) * float(last);
  const int   i    = min(int(x), last - 1);
  const int   j    = min(int(y), last - 1);
  const float fx   = x - float(i);
  const float fy   = y - float(j);
  const int   k    = i * GGX_ALBEDO_TABLE_RES + j;
  const int   kx   = k + GGX_ALBEDO_TABLE_RES;  // the next roughness
  const float s0   = GGX_ALBEDO_SCALE[k] + (GGX_ALBEDO_SCALE[kx] - GGX_ALBEDO_SCALE[k]) * fx;
  const float s1   = GGX_ALBEDO_SCALE[k + 1] + (GGX_ALBEDO_SCALE[kx + 1] - GGX_ALBEDO_SCALE[k + 1]) * fx;
  const float b0   = GGX_ALBEDO_BIAS[k] + (GGX_ALBEDO_BIAS[kx] - GGX_ALBEDO_BIAS[k]) * fx;
  const float b1   = GGX_ALBEDO_BIAS[k + 1] + (GGX_ALBEDO_BIAS[kx + 1] - GGX_ALBEDO_BIAS[k + 1]) * fx;
  return float2(s0 + (s1 - s0) * fy, b0 + (b1 - b0) * fy);
}

// The albedo itself, for one f0
float ggxSchlickAlbedo(float f0, float alpha, float NoV)
{
  const float2 sb = ggxAlbedoScaleBias(alpha, NoV);
  return clamp(f0 * sb.x + sb.y, 0.0F, 1.0F);
}

/*-------------------------------------------------------------------------------------------------
# Function `ggxMultiScatterFactor`
> The light a GGX reflection lobe of Schlick's Fresnel from `f0` loses to single scattering, given
> back: 1 + f0 (1 / E - 1), E its albedo with no Fresnel at `NoV` (Turquin 2019). The lobe keeps
> its shape. A rough white metal reflects all the light again instead of 50-70% (alpha 1).
-------------------------------------------------------------------------------------------------*/
float3 ggxMultiScatterFactor(float3 f0, float alpha, float NoV)
{
  const float2 sb = ggxAlbedoScaleBias(alpha, NoV);
  const float  e  = max(sb.x + sb.y, 0.05F);
  return float3(1.0F) + f0 * (1.0F / e - 1.0F);
}

float dielectricF0(float eta)
{
  const float r = (eta - 1.0F) / (eta + 1.0F);
  return r * r;
}

/*-------------------------------------------------------------------------------------------------
# Function `bsdfFresnelInLobes`
> Whether the dielectric lobes weigh each microfacet by its own Fresnel (at k1.h), as a microfacet
> dielectric does, instead of the whole lobe by one Fresnel at an estimated angle. The reflection then
> rises at grazing light as it should, and what it leaves goes to the layers below by the lobe's
> albedo at the view (Unreal's and Kulla and Conty's albedo scaling). Not on the denser side of the
> boundary, where the albedo fit knows no total internal reflection, nor with iridescence, whose
> thin-film factor stays the estimated one: both keep the lobe weights' Fresnel.
-------------------------------------------------------------------------------------------------*/
bool bsdfFresnelInLobes(PbrMaterial mat)
{
  return mat.iridescence <= 0.0F && mat.ior2 >= mat.ior1;
}

// The share of the light at view cosine `VdotN` the dielectric specular lobe reflects
// (bsdfFresnelInLobes): specular, its colour's largest channel, and the lobe's albedo
float bsdfSpecularAlbedo(PbrMaterial mat, float VdotN)
{
  const float colour = min(max(mat.specularColor.x, max(mat.specularColor.y, mat.specularColor.z)), 1.0F);
  const float alpha  = 0.5F * (mat.roughness.x + mat.roughness.y);
  return clamp(mat.specular * colour * ggxSchlickAlbedo(dielectricF0(mat.ior2 / mat.ior1), alpha, VdotN), 0.0F, 1.0F);
}

// The clearcoat's, likewise: its index 1.5 against ior1, its own roughness
bool bsdfClearcoatFresnelInLobe(PbrMaterial mat)
{
  return 1.5F >= mat.ior1;
}

float bsdfClearcoatAlbedo(PbrMaterial mat, float VdotN)
{
  const float alpha = mat.clearcoatRoughness * mat.clearcoatRoughness;
  return clamp(mat.clearcoat * ggxSchlickAlbedo(dielectricF0(1.5F / mat.ior1), alpha, VdotN), 0.0F, 1.0F);
}

/*-------------------------------------------------------------------------------------------------
# Function `computeLobeWeights`
>Calculates the weights of the individual lobes inside the standard PBR material.

Returns an array which can be indexed using the `LOBE_*` defines. This can be
used to perform your own lobe sampling.

Note that `tint` will be changed if the material has iridescence! (It's
convenient to compute the iridescence factor here.) This means you should avoid
passing a material parameter to the `tint` field. Make a temporary instead:
```glsl
float3 tint = mat.baseColor;
float[LOBE_COUNT] weights = computeLobeWeights(mat, dot(k1, mat.N), tint);
```
-------------------------------------------------------------------------------------------------*/
ARRAY_TYPE(float, LOBE_COUNT, ) computeLobeWeights(PbrMaterial mat, float VdotN, NVSHADERS_INOUT_TYPE(float3) tint)
{
  float frCoat = 0.0F;
  if(mat.clearcoat > 0.0f)
  {
    if(bsdfClearcoatFresnelInLobe(mat))
    {
      frCoat = bsdfClearcoatAlbedo(mat, VdotN);
    }
    else
    {
      float frCosineClearcoat = fresnelCosineApproximation(VdotN, mat.clearcoatRoughness);
      frCoat                  = mat.clearcoat * ior_fresnel(1.5f / mat.ior1, frCosineClearcoat);
    }
  }

  // This Fresnel value defines the weighting between dielectric specular reflection and
  // the base dielectric BSDFs (diffuse reflection and specular transmission): the reflection lobe's
  // albedo at the view when its microfacets carry their own Fresnel (bsdfFresnelInLobes).
  float frDielectric = 0;
  if(mat.specular > 0)
  {
    if(bsdfFresnelInLobes(mat))
    {
      frDielectric = bsdfSpecularAlbedo(mat, VdotN);
    }
    else
    {
      float frCosineDielectric = fresnelCosineApproximation(VdotN, (mat.roughness.x + mat.roughness.y) * 0.5F);
      frDielectric             = ior_fresnel(mat.ior2 / mat.ior1, frCosineDielectric);
      frDielectric *= mat.specular;
    }
  }

  // Estimate the iridescence Fresnel factor with the angle to the normal, and
  // blend it in. That's good enough for specular reflections.
  if(mat.iridescence > 0.0f)
  {
    // When there is iridescence enabled, use the maximum of the estimated iridescence factor. (Estimated with VdotN, no half-vector H here.)
    // With the thinfilm decision this handles the mix between non-iridescence and iridescence strength automatically.
    float3 frIridescence = thin_film_factor(mat.iridescenceThickness, mat.iridescenceIor, mat.ior2, mat.ior1, VdotN);
    frDielectric = lerp(frDielectric, max(frIridescence.x, max(frIridescence.y, frIridescence.z)), mat.iridescence);
    // Modulate the dielectric base lobe (diffuse, transmission) colors by the inverse of the iridescence factor,
    // though use the maximum component to not actually generate inverse colors.
    tint = mix_rgb(tint, mat.specularColor, frIridescence * mat.iridescence);
  }

  float sheen = 0.0f;
  if((mat.sheenColor.r != 0.0F || mat.sheenColor.g != 0.0F || mat.sheenColor.b != 0.0F))
  {
    sheen = pow(1.0f - abs(VdotN), mat.sheenRoughness);  // * luminance(mat.sheenColor);
    sheen = sheen / (sheen + 0.5F);
  }

  /*
  Our model consists of 6 layers. Each layer handles a fraction
  of light that didn't hit any layers above it:

    - Clearcoat       : clearcoat * schlickFresnel(1.5, VdotN)
    - Sheen           : sheen
    - Metal           : metallic
    - Specular        : specular * schlickFresnel(ior, VdotN)
    - Transmission    : transmission }
    - Diffuse         : 1.0          } These two are technically parallel layers
  */

  ARRAY_TYPE(float, LOBE_COUNT, weightLobe);
  weightLobe[LOBE_CLEARCOAT_REFLECTION]  = 0;
  weightLobe[LOBE_SHEEN_REFLECTION]      = 0;
  weightLobe[LOBE_METAL_REFLECTION]      = 0;
  weightLobe[LOBE_SPECULAR_REFLECTION]   = 0;
  weightLobe[LOBE_SPECULAR_TRANSMISSION] = 0;
  weightLobe[LOBE_DIFFUSE_TRANSMISSION]  = 0;
  weightLobe[LOBE_DIFFUSE_REFLECTION]    = 0;

  float weightBase = 1.0F;

  weightLobe[LOBE_CLEARCOAT_REFLECTION] = frCoat;  // BRDF clearcoat reflection (GGX-Smith)
  weightBase *= 1.0f - frCoat;

  weightLobe[LOBE_SHEEN_REFLECTION] = weightBase * sheen;  // BRDF sheen reflection (Lambert)
  weightBase *= 1.0f - sheen;

  weightLobe[LOBE_METAL_REFLECTION] = weightBase * mat.metallic;  // BRDF metal (GGX-Smith)
  weightBase *= 1.0f - mat.metallic;

  weightLobe[LOBE_SPECULAR_REFLECTION] = weightBase * frDielectric;  // BRDF dielectric specular reflection (GGX-Smith)
  weightBase *= 1.0f - frDielectric;

  weightLobe[LOBE_SPECULAR_TRANSMISSION] = weightBase * mat.transmission;  // BTDF dielectric specular transmission (GGX-Smith)

  float remainingWeight                 = weightBase * (1.0f - mat.transmission);
  weightLobe[LOBE_DIFFUSE_TRANSMISSION] = remainingWeight * mat.diffuseTransmissionFactor;
  weightLobe[LOBE_DIFFUSE_REFLECTION]   = remainingWeight * (1.0f - mat.diffuseTransmissionFactor);

  return weightLobe;
}

// findLobe's selection, from weights already computed: walks from the last lobe down and returns lobe 0
// when nothing else was picked. findLobe = computeLobeWeights + this (bit-identical to before).
int lobeFromWeights(ARRAY_TYPE(float, LOBE_COUNT, weightLobe), float rndVal)
{
  int   lobe   = LOBE_COUNT;
  float weight = 0.0f;
  // Stops when lobe reaches 0! No need to look at weightLobe[0] since we know
  // light will always scatter on one lobe.
  while(--lobe > 0)
  {
    weight += weightLobe[lobe];
    if(rndVal < weight)
    {
      break;  // Sample and evaluate this lobe!
    }
  }

  return lobe;
}

/*-------------------------------------------------------------------------------------------------
# Function `findLobe`
> Calculates the weights of the individual lobes inside the standard PBR material
> and randomly selects one.
-------------------------------------------------------------------------------------------------*/
int findLobe(PbrMaterial mat, float VdotN, float rndVal, NVSHADERS_INOUT_TYPE(float3) tint)
{
  ARRAY_TYPE(float, LOBE_COUNT, weightLobe) = computeLobeWeights(mat, VdotN, tint);
  return lobeFromWeights(weightLobe, rndVal);
}

/*
Shadow-terminator softening for a bumped diffuse lobe (Conty Estevez, Lecocq and Stein 2019, "Taming
the Shadow Terminator"; Unreal's ShadowTerminatorTerm). A normal or detail map tilts `N` away from
the smooth normal `Ns`, and Lambert's cosine against the tilted normal reaches zero along a hard line
while the smooth surface is still lit. The term reads the tilt as microfacet roughness and masks the
light toward `L` with its Smith G1, so it fades out toward the terminator. It is 1 with no bump
(N = Ns) and for a light along N. Applied to evaluation and sampling alike, it leaves the pdf alone.
*/
float shadowTerminatorTerm(float3 L, float3 N, float3 Ns)
{
  const float cosD = min(abs(dot(N, Ns)), 1.0f);
  const float cosI = min(dot(N, L), 1.0f);
  if(cosI <= 0.0f)
    return 0.0f;
  if(cosD >= 1.0f)
    return 1.0f;
  const float alpha2 = min(0.125f * (1.0f - cosD * cosD) / max(cosD * cosD, 1e-6f), 1.0f);
  const float tan2I  = (1.0f - cosI * cosI) / (cosI * cosI);
  return 2.0f / (1.0f + sqrt(1.0f + alpha2 * tan2I));
}

/*
The specular lobes' normal ("don't flip", Quilez; Unreal's AdjustShadingNormal). A normal map can
tilt `N` so far that a view `V` above the surface (`Ng`) mirrors below it, where a reflection lobe
sends nothing: black pixels along bumped silhouettes. The normal is then bent back, depending on the
view, to the half vector between V and the mirror direction laid onto the surface, so the mirror
direction grazes the surface instead. T and B follow it. Only for a view on Ng's side (the caller
flips Ng to the ray); the diffuse lobe keeps the map's normal.
*/
PbrMaterial bsdfSpecularFrame(PbrMaterial mat, float3 V)
{
  if(dot(V, mat.Ng) <= 0.0f)
    return mat;
  const float3 R = 2.0f * dot(V, mat.N) * mat.N - V;
  const float  k = dot(R, mat.Ng);
  if(k >= 0.0f)
    return mat;
  const float3 h = R - k * mat.Ng;
  mat.N          = dot(h, h) > 1e-12f ? normalize(normalize(h) + V) : mat.Ng;
  const float3 t = mat.T - mat.N * dot(mat.N, mat.T);
  if(dot(t, t) > 1e-8f)
  {
    const float3 b = cross(mat.N, normalize(t));
    mat.B          = dot(b, mat.B) < 0.0f ? -b : b;
    mat.T          = normalize(t);
  }
  else
  {
    orthonormalBasis(mat.N, mat.T, mat.B);
  }
  return mat;
}

/*
The softened diffuse lobe (`mat.diffuseSoftening` s, per channel): a Lambert lobe about the map's
normal N with weight 1 - s and one about the smooth normal Ns with weight s, as light that spreads
under a surface evens out the bumps it entered by, its red the most. It is sampled from the same two
lobes mixed by the channels' mean weight, so the sample weight stays within max((1 - s) / pN,
s / (1 - pN)) of the colour.
*/
float diffuseSofteningMixN(PbrMaterial mat)
{
  return 1.0f - (mat.diffuseSoftening.x + mat.diffuseSoftening.y + mat.diffuseSoftening.z) / 3.0f;
}

// Value (cosine included) and pdf of the softened diffuse lobe toward `k2`.
void brdf_diffuse_softened(PbrMaterial mat, float3 tint, float3 k2, NVSHADERS_OUT_TYPE(float3) value, NVSHADERS_OUT_TYPE(float) pdf)
{
  const float  cosN = max(dot(k2, mat.N), 0.0f);
  const float  cosS = max(dot(k2, mat.Ns), 0.0f);
  const float  pN   = diffuseSofteningMixN(mat);
  const float3 s    = mat.diffuseSoftening;
  pdf               = (pN * cosN + (1.0f - pN) * cosS) * M_1_PI;
  value = tint * M_1_PI * ((float3(1.0f) - s) * (cosN * shadowTerminatorTerm(k2, mat.N, mat.Ns)) + s * cosS);
}

// Evaluates a diffuse lobe.
void brdf_diffuse_eval(NVSHADERS_INOUT_TYPE(BsdfEvaluateData) data, PbrMaterial mat, float3 tint)
{
  if(max(mat.diffuseSoftening.x, max(mat.diffuseSoftening.y, mat.diffuseSoftening.z)) > 0.0f)
  {
    float3 value;
    float  pdf;
    brdf_diffuse_softened(mat, tint, data.k2, value, pdf);
    if(pdf <= 0.0f)
      return;  // absorb
    data.pdf          = pdf;
    data.bsdf_diffuse = value;
    return;
  }

  // If the incoming light direction is on the backside, there is nothing to evaluate for a BRDF.
  // Note that the state normals have been flipped to the ray side by the caller.
  // Include edge-on (== 0.0f) as "no light" case.
  const float nk2 = dot(data.k2, mat.N);
  if(nk2 <= 0.0f)  // if (backside)
  {
    return;  // absorb
  }
  data.pdf = nk2 * M_1_PI;

  // For a white Lambert material, the bxdf components match the evaluation pdf. (See MDL_renderer.)
  data.bsdf_diffuse = tint * data.pdf * shadowTerminatorTerm(data.k2, mat.N, mat.Ns);
}

void brdf_diffuse_eval(NVSHADERS_INOUT_TYPE(BsdfEvaluateData) data, PbrMaterial mat)
{
  brdf_diffuse_eval(data, mat, mat.baseColor);
}

// Samples a diffuse lobe.
void brdf_diffuse_sample(NVSHADERS_INOUT_TYPE(BsdfSampleData) data, PbrMaterial mat, float3 tint)
{
  if(max(mat.diffuseSoftening.x, max(mat.diffuseSoftening.y, mat.diffuseSoftening.z)) > 0.0f)
  {
    // One of the two lobes by the mean weight, xi.x stretched back over [0, 1)
    const float pN     = diffuseSofteningMixN(mat);
    float       xi     = data.xi.x;
    const bool  aboutN = xi < pN;
    xi                 = aboutN ? xi / pN : (xi - pN) / max(1.0f - pN, 1e-6f);
    float3 T = mat.T, B = mat.B, N = mat.N;
    if(!aboutN)
    {
      N = mat.Ns;
      orthonormalBasis(N, T, B);
    }
    const float3 local = cosineSampleHemisphere(min(xi, 0.99999994f), data.xi.y);
    data.k2            = normalize(T * local.x + B * local.y + N * local.z);
    float3 value;
    brdf_diffuse_softened(mat, tint, data.k2, value, data.pdf);
    if(data.pdf <= 0.0f)
    {
      data.bsdf_over_pdf = float3(0.0f);
      data.event_type    = BSDF_EVENT_ABSORB;
      return;
    }
    data.bsdf_over_pdf = value / data.pdf;
    data.event_type    = BSDF_EVENT_DIFFUSE_REFLECTION;
    return;
  }

  data.k2  = cosineSampleHemisphere(data.xi.x, data.xi.y);
  data.k2  = mat.T * data.k2.x + mat.B * data.k2.y + mat.N * data.k2.z;
  data.k2  = normalize(data.k2);
  data.pdf = dot(data.k2, mat.N) * M_1_PI;

  data.bsdf_over_pdf = tint * shadowTerminatorTerm(data.k2, mat.N, mat.Ns);  // bsdf * dot(wi, normal) / pdf;
  data.event_type    = BSDF_EVENT_DIFFUSE_REFLECTION;
}

void brdf_diffuse_sample(NVSHADERS_INOUT_TYPE(BsdfSampleData) data, PbrMaterial mat)
{
  brdf_diffuse_sample(data, mat, mat.baseColor);
}

// Evaluates the diffuse-transmission lobe (Lambertian BTDF).
// Per KHR_materials_diffuse_transmission, the BTDF color is `tint` (the caller passes
// `mat.diffuseTransmissionColorFactor`). Independent of the surface's `baseColor` -- baseColor
// only affects the diffuse REFLECTION lobe. Symmetric with `brdf_diffuse_eval`.
void brdf_diffuse_transmission_eval(NVSHADERS_INOUT_TYPE(BsdfEvaluateData) data, PbrMaterial mat, float3 tint)
{
  // If the incoming light direction is on the same side as the surface normal,
  // there is no diffuse transmission to evaluate. Absorb light in this case.
  float nk2 = dot(data.k2, mat.N);
  if(-nk2 <= 0.0f)
  {
    return;  // absorb
  }

  data.pdf = max(0.0f, -nk2 * M_1_PI);

  // For a white Lambert material, the bxdf components match the evaluation pdf. (See MDL_renderer.)
  data.bsdf_diffuse = tint * data.pdf;
}

// Samples the diffuse-transmission lobe.
// Per KHR_materials_diffuse_transmission, the BTDF color is `tint` (the caller passes
// `mat.diffuseTransmissionColorFactor`). Independent of the surface's `baseColor`. Symmetric with
// `brdf_diffuse_sample`.
void brdf_diffuse_transmission_sample(NVSHADERS_INOUT_TYPE(BsdfSampleData) data, PbrMaterial mat, float3 tint)
{
  // Sample a direction on the opposite hemisphere (relative to -mat.N).
  float3 sampledDir = cosineSampleHemisphere(data.xi.x, data.xi.y);
  data.k2           = mat.T * sampledDir.x + mat.B * sampledDir.y - mat.N * sampledDir.z;  // Flip the normal
  data.k2           = normalize(data.k2);

  // Compute the PDF for this direction.
  data.pdf = dot(data.k2, -mat.N) * M_1_PI;

  // Lambertian BTDF / cosine-PDF; cancels to just `tint` (= diffuseTransmissionColorFactor).
  data.bsdf_over_pdf = tint;
  data.event_type    = BSDF_EVENT_DIFFUSE_TRANSMISSION;
}


// The Fresnel a reflection lobe weighs a microfacet by (BSDF_FRESNEL_*), at k1.h
float3 bsdfMicrofacetFresnel(int fresnel, float eta, float3 tint, float k1h)
{
  if(fresnel == BSDF_FRESNEL_DIELECTRIC)
    return tint * ior_fresnel(eta, k1h);
  if(fresnel == BSDF_FRESNEL_SCHLICK)
    return schlickFresnel(tint, float3(1.0F), k1h);
  return tint;
}

// Evaluates a reflective lobe with a GGX NDF and a height-correlated Smith
// scattering-masking function. `fresnel`, `eta`: how each microfacet's reflection is weighed.
void brdf_ggx_smith_eval(NVSHADERS_INOUT_TYPE(BsdfEvaluateData) data, PbrMaterial mat, const int lobe, float3 tint, int fresnel, float eta)
{
  mat = bsdfSpecularFrame(mat, data.k1);
  // A reflection lobe has nothing below its hemisphere on either side: the sampler absorbs a view
  // below N (nk1 <= 0) and never produces a direction below it (as the diffuse and sheen evals reject
  // those), so a light sample there had no BSDF-sample partner. A normal map bends N past the view at
  // silhouettes.
  if(dot(data.k1, mat.N) <= 0.0f || dot(data.k2, mat.N) <= 0.0f)
  {
    data.pdf         = 0.0f;
    data.bsdf_glossy = float3(0.0f);
    return;
  }

  const float nk1 = abs(dot(data.k1, mat.N));
  const float nk2 = abs(dot(data.k2, mat.N));

  // compute_half_vector() for scatter_reflect.
  const float3 h = normalize(data.k1 + data.k2);

  // Invalid for reflection / refraction?
  const float nh  = dot(mat.N, h);
  const float k1h = dot(data.k1, h);
  const float k2h = dot(data.k2, h);

  // nk1 and nh must not be 0.0f or state.pdf == NaN.
  if(nk1 <= 0.0f || nh <= 0.0f || k1h < 0.0f || k2h < 0.0f)
  {
    data.pdf         = 0.0f;
    data.bsdf_glossy = float3(0.0f);
    return;
  }

  // Compute BSDF and pdf.
  const float3 h0 = float3(dot(mat.T, h), dot(mat.B, h), nh);

  data.pdf = hvd_ggx_eval(1.0f / mat.roughness, h0);
  float G1;
  float G2;

  float G12;
  G12 = ggx_smith_shadow_mask_correlated(G1, G2, float3(dot(mat.T, data.k1), dot(mat.B, data.k1), nk1),
                                         float3(dot(mat.T, data.k2), dot(mat.B, data.k2), nk2), mat.roughness);

  data.pdf *= 0.25f / (nk1 * nh);

  float3 bsdf = float3(G12 * data.pdf);

  data.pdf *= G1;

  // A metal's lost multiple scattering, given back by its Fresnel's f0 (the base colour)
  if(lobe == LOBE_METAL_REFLECTION && fresnel == BSDF_FRESNEL_SCHLICK)
    bsdf *= ggxMultiScatterFactor(tint, 0.5f * (mat.roughness.x + mat.roughness.y), nk1);

  tint = bsdfMicrofacetFresnel(fresnel, eta, tint, k1h);

  if(mat.iridescence > 0.0f)
  {
    const float3 factor = thin_film_factor(mat.iridescenceThickness, mat.iridescenceIor, mat.ior2, mat.ior1, k1h);

    switch(lobe)
    {
      case LOBE_SPECULAR_REFLECTION:
        tint *= lerp(float3(1.f), factor, mat.iridescence);
        break;

      case LOBE_METAL_REFLECTION:
        tint = mix_rgb(tint, mat.specularColor, factor * mat.iridescence);
        break;
    }
  }

  // eval output: (glossy part of the) bsdf * dot(k2, normal)
  data.bsdf_glossy = bsdf * tint;
}

// Samples a reflective lobe with a GGX NDF and a height-correlated Smith
// scattering-masking function. `fresnel`, `eta`: as brdf_ggx_smith_eval's.
void brdf_ggx_smith_sample(NVSHADERS_INOUT_TYPE(BsdfSampleData) data, PbrMaterial mat, const int lobe, float3 tint, int fresnel, float eta)
{
  // When the sampling returns eventType = BSDF_EVENT_ABSORB, the path ends inside the ray generation program.
  // Make sure the returned values are valid numbers when manipulating the PRD.
  data.bsdf_over_pdf = float3(0.0f);
  data.pdf           = 0.0f;
  mat                = bsdfSpecularFrame(mat, data.k1);

  // Transform to local coordinate system
  const float nk1 = dot(data.k1, mat.N);
  if(nk1 <= 0.0f)
  {
    data.event_type = BSDF_EVENT_ABSORB;
    return;
  }
  const float3 k10 = float3(dot(data.k1, mat.T), dot(data.k1, mat.B), nk1);

  // Sample half-vector, microfacet normal.
  const float3 h0 = hvd_ggx_sample_vndf(k10, mat.roughness, data.xi.xy);
  if(h0.z == 0.0f)
  {
    data.event_type = BSDF_EVENT_ABSORB;
    return;
  }

  // Transform to world
  const float3 h  = h0.x * mat.T + h0.y * mat.B + h0.z * mat.N;
  const float  kh = dot(data.k1, h);

  if(kh <= 0.0f)
  {
    data.event_type = BSDF_EVENT_ABSORB;
    return;
  }

  // BRDF: reflect
  data.k2 = (2.0f * kh) * h - data.k1;

  // Check if the resulting direction is on the correct side of the shading normal
  const float nk2 = dot(data.k2, mat.N);

  if(nk2 <= 0.0f)
  {
    data.event_type = BSDF_EVENT_ABSORB;
    return;
  }

  const float k2h = abs(dot(data.k2, h));

  float G1;
  float G2;

  float G12 = ggx_smith_shadow_mask_correlated(G1, G2, k10, float3(dot(data.k2, mat.T), dot(data.k2, mat.B), nk2), mat.roughness);

  if(G12 <= 0.0f)
  {
    data.event_type = BSDF_EVENT_ABSORB;
    return;
  }

  // The VNDF sample's pdf has G1: what is left of the joint masking
  data.bsdf_over_pdf = float3(G12 / G1);
  data.event_type    = BSDF_EVENT_GLOSSY_REFLECTION;

  // Compute pdf
  data.pdf = hvd_ggx_eval(1.0f / mat.roughness, h0) * G1;
  data.pdf *= 0.25f / (nk1 * h0.z);

  if(lobe == LOBE_METAL_REFLECTION && fresnel == BSDF_FRESNEL_SCHLICK)
    data.bsdf_over_pdf *= ggxMultiScatterFactor(tint, 0.5f * (mat.roughness.x + mat.roughness.y), nk1);

  tint = bsdfMicrofacetFresnel(fresnel, eta, tint, kh);

  if(mat.iridescence > 0.0f)
  {
    const float3 factor = thin_film_factor(mat.iridescenceThickness, mat.iridescenceIor, mat.ior2, mat.ior1, kh);

    switch(lobe)
    {
      case LOBE_SPECULAR_REFLECTION:
        tint *= lerp(float3(1.f), factor, mat.iridescence);
        break;

      case LOBE_METAL_REFLECTION:
        tint = mix_rgb(tint, mat.specularColor, factor * mat.iridescence);
        break;
    }
  }

  data.bsdf_over_pdf *= tint;
}

/*
 * In rare cases (mainly dispersion), we need to get an additional random
 * number. This function takes a random number and applies the hash function
 * from pcg_output_rxs_m_xs_32_32 to it.
 * Its quality hasn't been tested and it's not very principled, but it's better
 * than using a random number that's correlated with material sampling.
*/
float rerandomize(float v)
{
  uint word = asuint(v);
  word      = ((word >> ((word >> 28) + 4)) ^ word) * 277803737U;
  word      = (word >> 22) ^ word;
  return float(word) / asfloat(0x4f800000U);
}

/*-------------------------------------------------------------------------------------------------
# Function `computeDispersedIOR`
> Calculates the IOR at a given wavelength (in nanometers) given the base IOR
> and glTF dispersion factor.

See https://github.com/KhronosGroup/glTF/tree/0251c5c0cce8daec69bd54f29f891e3d0cdb52c8/extensions/2.0/Khronos/KHR_materials_dispersion.
-------------------------------------------------------------------------------------------------*/
float computeDispersedIOR(float base_ior, float dispersion, float wavelength_nm)
{
  // Since the glTF extension stores 20 / V_d
  float abbeNumber = 20.0F / dispersion;
  // Last equation of https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_dispersion
  return max(base_ior + (base_ior - 1.0F) * (523655.0F / (wavelength_nm * wavelength_nm) - 1.5168F) / abbeNumber, 1.0F);
}

/*-------------------------------------------------------------------------------------------------
# Function `wavelengthToRGB`
> Given a wavelength of light, returns an approximation to the linear RGB
> color of a D65 illuminant (sRGB whitepoint) sampled at a wavelength of `x`
> nanometers, using the CIE 2015 2-degree Standard Observer Color Matching Functions.

This is normalized so that
`sum(wavelengthToRGB(i), {i, WAVELENGTH_MIN, WAVELENGTH_MAX}) == float3(1.)`,
which means that the values it returns are usually low. You'll need to multiply
by an appropriate normalization factor if you're randomly sampling it.

The colors here are clamped to only positive sRGB values, in case renderers
have problems with colors with negative sRGB components (i.e. are valid
colors but are out-of-gamut).
-------------------------------------------------------------------------------------------------*/
#define WAVELENGTH_MIN 399.43862850585765F
#define WAVELENGTH_MAX 668.6617899434457F
float3 wavelengthToRGB(float x)
{
  // This uses a piecewise-linear approximation generated using the CIE 2015
  // 2-degree standard observer times the D65 illuminant, minimizing the L2
  // norm, then normalized to have an integral of 1.
  float3 rgb = float3(0.);
  if(399.43862850585765 < x)
  {
    if(x < 435.3450352446586)
      rgb.r = 2.6268757476158464e-05 * x + -0.010492756458829732;
    else if(x < 452.7741480943567)
      rgb.r = -5.383671438883332e-05 * x + 0.024380763013525125;
    else if(x < 550.5919453498173)
      rgb.r = 1.2536207000814165e-07 * x + -5.187018452935683e-05;
    else if(x < 600.8694441891222)
      rgb.r = 0.00032842519537482 * x + -0.18081111406184644;
    else if(x < 668.6617899434457)
      rgb.r = -0.0002438262071743009 * x + 0.16303726812428945;
  }
  if(467.41924217251835 < x)
  {
    if(x < 532.3927928594046)
      rgb.g = 0.00020126149345609334 * x + -0.0940734947497564;
    else if(x < 552.5312202450474)
      rgb.g = -4.3718474429905034e-05 * x + 0.03635207454767751;
    else if(x < 605.5304635656746)
      rgb.g = -0.00023012125757884968 * x + 0.13934543177803685;
  }
  if(400.68666327204835 < x)
  {
    if(x < 447.59688835108466)
      rgb.b = 0.00042519082480799777 * x + -0.1703682928462067;
    else if(x < 501.2110070697423)
      rgb.b = -0.00037202508909921054 * x + 0.18646306956262593;
  }
  return rgb;
}


// Evaluates the transmission lobe.
// `reflected`: what each microfacet's reflection takes from what it transmits, times its Fresnel at
// k1.h (bsdfFresnelInLobes, the reflection being the specular lobe's); 0 when the lobe weights carry it.
void btdf_ggx_smith_eval(NVSHADERS_INOUT_TYPE(BsdfEvaluateData) data, PbrMaterial mat, float3 tint, float3 reflected)
{
  bool isThinWalled = (mat.thickness == 0.0F);

  float2 ior = float2(mat.ior1, mat.ior2);
  if(mat.dispersion > 0.0f)
  {
    // Randomly choose a wavelength; for now, uniformly choose from 399-669 nm.
    float wavelength = lerp(WAVELENGTH_MIN, WAVELENGTH_MAX, rerandomize(data.xi.z));
    ior.x            = computeDispersedIOR(ior.x, mat.dispersion, wavelength);
    tint *= (WAVELENGTH_MAX - WAVELENGTH_MIN) * wavelengthToRGB(wavelength);
  }

  const float nk1 = abs(dot(data.k1, mat.N));
  float       nk2 = dot(data.k2, mat.N);

  // BRDF or BTDF eval?
  // If the incoming light direction is on the backface.
  // Do NOT include edge-on (== 0.0f) as backside here to take the reflection path.
  const bool backside = (nk2 < 0.0f);
  nk2                 = abs(nk2);

  const float3 h = compute_half_vector(data.k1, data.k2, mat.N, ior, nk2, backside, isThinWalled);

  // Invalid for reflection / refraction?
  const float nh  = dot(mat.N, h);
  const float k1h = dot(data.k1, h);
  const float k2h = dot(data.k2, h) * (backside ? -1.0f : 1.0f);

  // nk1 and nh must not be 0.0f or state.pdf == NaN.
  if(nk1 <= 0.0f || nh <= 0.0f || k1h < 0.0f || k2h < 0.0f)
  {
    data.pdf         = 0.0f;  // absorb
    data.bsdf_glossy = float3(0.0f);
    return;
  }

  float fr;

  if(!backside)
  {
    // For scatter_transmit: Only allow TIR with BRDF eval.
    if(!isTIR(ior, k1h))
    {
      data.pdf         = 0.0f;  // absorb
      data.bsdf_glossy = float3(0.0f);
      return;
    }
    else
    {
      fr = 1.0f;
    }
  }
  else
  {
    fr = 0.0f;
  }

  // Compute BSDF and pdf
  const float3 h0 = float3(dot(mat.T, h), dot(mat.B, h), nh);
  data.pdf        = hvd_ggx_eval(1.0f / mat.roughness, h0);

  float G1;
  float G2;
  float G12 = ggx_smith_shadow_mask(G1, G2, float3(dot(mat.T, data.k1), dot(mat.B, data.k1), nk1),
                                    float3(dot(mat.T, data.k2), dot(mat.B, data.k2), nk2), mat.roughness);

  if(!isThinWalled && backside)  // Refraction?
  {
    // Refraction pdf and BTDF
    const float tmp = k1h * ior.x - k2h * ior.y;

    // The half-vector Jacobian carries eta_t^2 (ior.y), so the pdf integrates to 1 over the refracted
    // directions; the value is built from it, so both rise together.
    data.pdf *= ior.y * ior.y * k1h * k2h / (nk1 * nh * tmp * tmp);
  }
  else
  {
    // Reflection pdf and BRDF (and pseudo-BTDF for thin-walled)
    data.pdf *= 0.25f / (nk1 * nh);
  }

  const float prob = (backside) ? 1.0f - fr : fr;

  const float3 bsdf = float3(prob * G12 * data.pdf);

  data.pdf *= prob * G1;

  if(backside)
    tint *= max(float3(1.0F) - reflected * ior_fresnel(ior.y / ior.x, k1h), float3(0.0F));

  // eval output: (glossy part of the) bsdf * dot(k2, normal)
  data.bsdf_glossy = bsdf * tint;
}


// Samples the transmission lobe. `reflected`: as btdf_ggx_smith_eval's.
void btdf_ggx_smith_sample(NVSHADERS_INOUT_TYPE(BsdfSampleData) data, PbrMaterial mat, float3 tint, float3 reflected)
{
  bool isThinWalled = (mat.thickness == 0.0F);

  // When the sampling returns eventType = BSDF_EVENT_ABSORB, the path ends inside the ray generation program.
  // Make sure the returned values are valid numbers when manipulating the PRD.
  data.bsdf_over_pdf = float3(0.0f);
  data.pdf           = 0.0f;

  float2 ior = float2(mat.ior1, mat.ior2);
  if(mat.dispersion > 0.0f)
  {
    // Randomly choose a wavelength; for now, uniformly choose from 361-716 nm.
    float wavelength = lerp(WAVELENGTH_MIN, WAVELENGTH_MAX, rerandomize(data.xi.z));
    ior.x            = computeDispersedIOR(ior.x, mat.dispersion, wavelength);
    tint *= (WAVELENGTH_MAX - WAVELENGTH_MIN) * wavelengthToRGB(wavelength);
  }

  const float nk1 = abs(dot(data.k1, mat.N));

  const float3 k10 = float3(dot(data.k1, mat.T), dot(data.k1, mat.B), nk1);

  // Sample half-vector, microfacet normal.
  const float3 h0 = hvd_ggx_sample_vndf(k10, mat.roughness, data.xi.xy);

  if(abs(h0.z) == 0.0f)
  {
    data.event_type = BSDF_EVENT_ABSORB;
    return;
  }

  // Transform to world
  const float3 h = h0.x * mat.T + h0.y * mat.B + h0.z * mat.N;

  const float kh = dot(data.k1, h);

  if(kh <= 0.0f)
  {
    data.event_type = BSDF_EVENT_ABSORB;
    return;
  }

  // Case scatter_transmit
  bool tir = false;
  if(isThinWalled)  // No refraction!
  {
    // pseudo-BTDF: flip a reflected reflection direction to the back side
    data.k2 = (2.0f * kh) * h - data.k1;
    data.k2 = normalize(data.k2 - 2.0f * mat.N * dot(data.k2, mat.N));
  }
  else
  {
    // BTDF: refract
    data.k2 = refract(data.k1, h, ior.x / ior.y, kh, tir);
  }

  data.bsdf_over_pdf = float3(1.0f);  // Was: (float3(1.0f) - fr) / prob; // PERF Always white with the original setup.
  data.event_type    = (tir) ? BSDF_EVENT_GLOSSY_REFLECTION : BSDF_EVENT_GLOSSY_TRANSMISSION;

  // Check if the resulting direction is on the correct side of the shading normal
  const float nk2 = dot(data.k2, mat.N) * (tir ? 1.0f : -1.0f);

  if(nk2 <= 0.0f || isnan(data.k2.x))
  {
    data.event_type = BSDF_EVENT_ABSORB;
    return;
  }


  const float k2h = abs(dot(data.k2, h));

  float G1;
  float G2;
  float G12 = ggx_smith_shadow_mask(G1, G2, k10, float3(dot(data.k2, mat.T), dot(data.k2, mat.B), nk2), mat.roughness);

  if(G12 <= 0.0f)
  {
    data.event_type = BSDF_EVENT_ABSORB;
    return;
  }

  data.bsdf_over_pdf *= G2;

  // Compute pdf
  data.pdf = hvd_ggx_eval(1.0f / mat.roughness, h0) * G1;  // * prob;


  if(!isThinWalled && (data.event_type == BSDF_EVENT_GLOSSY_TRANSMISSION))  // if (refraction)
  {
    // As btdf_ggx_smith_eval's: eta_t^2 in the Jacobian. tmp is negative entering a denser medium and
    // only its square counts, so the sign must not drop the Jacobian.
    const float tmp = kh * ior.x - k2h * ior.y;
    if(tmp != 0.0f)
    {
      data.pdf *= ior.y * ior.y * kh * k2h / (nk1 * h0.z * tmp * tmp);
    }
  }
  else
  {
    data.pdf *= 0.25f / (nk1 * h0.z);
  }

  if(data.event_type == BSDF_EVENT_GLOSSY_TRANSMISSION)
    tint *= max(float3(1.0F) - reflected * ior_fresnel(ior.y / ior.x, kh), float3(0.0F));

  data.bsdf_over_pdf *= tint;
}


// Evaluates the sheen lobe.
void brdf_sheen_eval(NVSHADERS_INOUT_TYPE(BsdfEvaluateData) data, PbrMaterial mat)
{
  mat = bsdfSpecularFrame(mat, data.k1);
  // BRDF or BTDF eval?
  // If the incoming light direction is on the backface.
  // Include edge-on (== 0.0f) as "no light" case.
  const float nk2      = dot(data.k2, mat.N);
  const bool  backside = (nk2 <= 0.0f);
  // Nothing to evaluate for given directions?
  if(backside)  // && scatter_reflect
  {
    return;  // absorb
  }

  const float nk1 = abs(dot(data.k1, mat.N));

  // compute_half_vector() for scatter_reflect.
  const float3 h = normalize(data.k1 + data.k2);

  // Invalid for reflection / refraction?
  const float nh  = dot(mat.N, h);
  const float k1h = dot(data.k1, h);
  const float k2h = dot(data.k2, h);

  // nk1 and nh must not be 0.0f or state.pdf == NaN.
  if(nk1 <= 0.0f || nh <= 0.0f || k1h < 0.0f || k2h < 0.0f)
  {
    return;  // absorb
  }

  const float invRoughness = 1.0f / (mat.sheenRoughness * mat.sheenRoughness);  // Perceptual roughness to alpha G.

  // Compute BSDF and pdf
  const float3 h0 = float3(dot(mat.T, h), dot(mat.B, h), nh);

  data.pdf = hvd_sheen_eval(invRoughness, h0.z);

  float G1;
  float G2;

  const float G12 = vcavities_shadow_mask(G1, G2, h0.z, float3(dot(mat.T, data.k1), dot(mat.B, data.k1), nk1), k1h,
                                          float3(dot(mat.T, data.k2), dot(mat.B, data.k2), nk2), k2h);
  data.pdf *= 0.25f / (nk1 * nh);

  const float3 bsdf = float3(G12 * data.pdf);

  data.pdf *= G1;

  // eval output: (glossy part of the) bsdf * dot(k2, normal)
  data.bsdf_glossy = bsdf * mat.sheenColor;
}


// Samples the sheen lobe.
void brdf_sheen_sample(NVSHADERS_INOUT_TYPE(BsdfSampleData) data, PbrMaterial mat)
{
  // When the sampling returns eventType = BSDF_EVENT_ABSORB, the path ends inside the ray generation program.
  // Make sure the returned values are valid numbers when manipulating the PRD.
  data.bsdf_over_pdf = float3(0.0f);
  data.pdf           = 0.0f;
  mat                = bsdfSpecularFrame(mat, data.k1);

  const float invRoughness = 1.0f / (mat.sheenRoughness * mat.sheenRoughness);  // Perceptual roughness to alpha G.

  const float nk1 = abs(dot(data.k1, mat.N));

  const float3 k10 = float3(dot(data.k1, mat.T), dot(data.k1, mat.B), nk1);

  float        xiFlip = data.xi.z;
  const float3 h0     = flip(hvd_sheen_sample(data.xi.xy, invRoughness), k10, xiFlip);

  if(abs(h0.z) == 0.0f)
  {
    data.event_type = BSDF_EVENT_ABSORB;
    return;
  }

  // Transform to world
  const float3 h = h0.x * mat.T + h0.y * mat.B + h0.z * mat.N;

  const float k1h = dot(data.k1, h);

  if(k1h <= 0.0f)
  {
    data.event_type = BSDF_EVENT_ABSORB;
    return;
  }

  // BRDF: reflect
  data.k2            = (2.0f * k1h) * h - data.k1;
  data.bsdf_over_pdf = float3(1.0f);  // PERF Always white with the original setup.
  data.event_type    = BSDF_EVENT_GLOSSY_REFLECTION;

  // Check if the resulting reflection direction is on the correct side of the shading normal.
  const float nk2 = dot(data.k2, mat.N);

  if(nk2 <= 0.0f)
  {
    data.event_type = BSDF_EVENT_ABSORB;
    return;
  }

  const float k2h = abs(dot(data.k2, h));

  float G1;
  float G2;

  const float G12 = vcavities_shadow_mask(G1, G2, h0.z, k10, k1h, float3(dot(data.k2, mat.T), dot(data.k2, mat.B), nk2), k2h);
  if(G12 <= 0.0f)
  {
    data.event_type = BSDF_EVENT_ABSORB;
    return;
  }

  data.bsdf_over_pdf *= G12 / G1;

  // Compute pdf.
  data.pdf = hvd_sheen_eval(invRoughness, h0.z) * G1;

  data.pdf *= 0.25f / (nk1 * h0.z);

  data.bsdf_over_pdf *= mat.sheenColor;
}


// The clearcoat lobe's material: its normal Nc, its roughness, an orthonormal frame about Nc and no
// iridescence. Shared by evaluate and sample so their pdfs agree when Nc != N.
PbrMaterial bsdfClearcoatMaterial(PbrMaterial mat)
{
  mat.roughness   = float2(mat.clearcoatRoughness * mat.clearcoatRoughness);
  mat.N           = mat.Nc;
  mat.B           = normalize(cross(mat.N, mat.T));  // Assumes Nc and T are not collinear (as the sampler did)
  mat.T           = cross(mat.B, mat.N);
  mat.iridescence = 0.0f;
  return mat;
}

/*-------------------------------------------------------------------------------------------------
# Function `bsdfLobeFresnel`
> How a reflection or transmission lobe carries its Fresnel, at view cosine `VdotN` (the one
> computeLobeWeights was given), with computeLobeWeights' `tint`. Weighed per microfacet, the lobe's
> value is divided by the albedo its weight carries, so weight times value is the microfacet BSDF
> whatever the albedo fit's error: that error only moves light between the lobe's share and the
> layers' below.
-------------------------------------------------------------------------------------------------*/
struct BsdfLobeFresnel
{
  float3 tint;
  int    fresnel;    // BSDF_FRESNEL_*, for the reflection lobes
  float  eta;        // for BSDF_FRESNEL_DIELECTRIC
  float3 reflected;  // the transmission lobe's: what each microfacet's reflection takes from it
};

BsdfLobeFresnel bsdfLobeFresnel(PbrMaterial mat, int lobe, float VdotN, float3 tint)
{
  BsdfLobeFresnel f;
  f.tint      = tint;
  f.fresnel   = BSDF_FRESNEL_NONE;
  f.eta       = 1.0F;
  f.reflected = float3(0.0F);
  if(lobe == LOBE_SPECULAR_REFLECTION)
  {
    f.tint = mat.specularColor;
    if(bsdfFresnelInLobes(mat))
    {
      f.tint    = mat.specularColor * (mat.specular / max(bsdfSpecularAlbedo(mat, VdotN), 1e-6F));
      f.fresnel = BSDF_FRESNEL_DIELECTRIC;
      f.eta     = mat.ior2 / mat.ior1;
    }
  }
  else if(lobe == LOBE_SPECULAR_TRANSMISSION)
  {
    // `transmissionTint` is float3(1) for plain glTF materials; see PbrMaterial.
    f.tint = tint * mat.transmissionTint;
    if(bsdfFresnelInLobes(mat) && mat.specular > 0.0F)
    {
      f.tint      = f.tint / max(1.0F - bsdfSpecularAlbedo(mat, VdotN), 1e-6F);
      f.reflected = mat.specular * mat.specularColor;
    }
  }
  else if(lobe == LOBE_METAL_REFLECTION)
  {
    // glTF's metal: Schlick's from the base colour. Iridescence keeps the plain base colour.
    f.tint    = mat.baseColor;
    f.fresnel = mat.iridescence <= 0.0F ? BSDF_FRESNEL_SCHLICK : BSDF_FRESNEL_NONE;
  }
  else if(lobe == LOBE_CLEARCOAT_REFLECTION)
  {
    f.tint = float3(1.0F);
    if(bsdfClearcoatFresnelInLobe(mat))
    {
      f.tint    = float3(mat.clearcoat / max(bsdfClearcoatAlbedo(mat, VdotN), 1e-6F));
      f.fresnel = BSDF_FRESNEL_DIELECTRIC;
      f.eta     = 1.5F / mat.ior1;
    }
  }
  return f;
}

/*-------------------------------------------------------------------------------------------------
# Function `bsdfEvaluateLobe`
> One lobe of the standard material, unweighted, without occlusion or retroreflection: its value
> (cosine included) and its own pdf. `tint` is computeLobeWeights' (iridescence-modified) base
> colour, used by the diffuse reflection and specular transmission lobes only.
-------------------------------------------------------------------------------------------------*/
void bsdfEvaluateLobe(NVSHADERS_INOUT_TYPE(BsdfEvaluateData) data, PbrMaterial mat, int lobe, float3 tint)
{
  data.bsdf_diffuse = float3(0.0f);
  data.bsdf_glossy  = float3(0.0f);
  data.pdf          = 0.0f;

  if(lobe == LOBE_DIFFUSE_REFLECTION)
  {
    brdf_diffuse_eval(data, mat, tint);
  }
  else if(lobe == LOBE_DIFFUSE_TRANSMISSION)
  {
    // Per KHR_materials_diffuse_transmission, the BTDF color is `diffuseTransmissionColorFactor`,
    // independent of `baseColor`. Pass it directly (bypassing the iridescence-modified `tint`,
    // which only applies to the dielectric base / specular layers).
    brdf_diffuse_transmission_eval(data, mat, mat.diffuseTransmissionColorFactor);
  }
  else if(lobe == LOBE_SPECULAR_REFLECTION || lobe == LOBE_METAL_REFLECTION)
  {
    const BsdfLobeFresnel f = bsdfLobeFresnel(mat, lobe, dot(data.k1, mat.N), tint);
    brdf_ggx_smith_eval(data, mat, lobe, f.tint, f.fresnel, f.eta);
  }
  else if(lobe == LOBE_SPECULAR_TRANSMISSION)
  {
    const BsdfLobeFresnel f = bsdfLobeFresnel(mat, lobe, dot(data.k1, mat.N), tint);
    btdf_ggx_smith_eval(data, mat, f.tint, f.reflected);
  }
  else if(lobe == LOBE_CLEARCOAT_REFLECTION)
  {
    const BsdfLobeFresnel f = bsdfLobeFresnel(mat, lobe, dot(data.k1, mat.N), tint);
    brdf_ggx_smith_eval(data, bsdfClearcoatMaterial(mat), LOBE_CLEARCOAT_REFLECTION, f.tint, f.fresnel, f.eta);
  }
  else if(lobe == LOBE_SHEEN_REFLECTION)
  {
    brdf_sheen_eval(data, mat);
  }
}

/*-------------------------------------------------------------------------------------------------
# Function `bsdfEvaluateCore`
> Single-path BSDF evaluation (forward or retro view direction in `data.k1`).
> Returns the stochastically selected lobe index.
-------------------------------------------------------------------------------------------------*/
int bsdfEvaluateCore(NVSHADERS_INOUT_TYPE(BsdfEvaluateData) data, PbrMaterial mat)
{
  float3    tint = mat.baseColor;
  const int lobe = findLobe(mat, dot(data.k1, mat.N), data.xi.z, tint);
  bsdfEvaluateLobe(data, mat, lobe, tint);

  // Occlusion effect (glTF occlusion texture): the bsdf, never the pdf
  data.bsdf_diffuse *= mat.occlusion;
  data.bsdf_glossy *= mat.occlusion;

  return lobe;
}

/*-------------------------------------------------------------------------------------------------
# Function `bsdfEvaluateRetroreflectionBlend`
> Blends a Minimal Retroreflective Microfacet Model evaluation into an already evaluated forward BSDF.
-------------------------------------------------------------------------------------------------*/
void bsdfEvaluateRetroreflectionBlend(NVSHADERS_INOUT_TYPE(BsdfEvaluateData) data, PbrMaterial mat, float blend)
{
  // Portsmouth et al. 2026's Minimal Retroreflective Microfacet Model
  // (https://jcgt.org/published/0015/01/04/) wraps an existing microfacet BSDF
  // by replacing the view direction V with reflect(-V, N).
  BsdfEvaluateData dRetro;
  dRetro.k1 = reflect(-data.k1, mat.N);
  dRetro.k2 = data.k2;
  dRetro.xi = data.xi;
  bsdfEvaluateCore(dRetro, mat);

  data.bsdf_diffuse = lerp(data.bsdf_diffuse, dRetro.bsdf_diffuse, blend);
  data.bsdf_glossy  = lerp(data.bsdf_glossy, dRetro.bsdf_glossy, blend);
  data.pdf          = lerp(data.pdf, dRetro.pdf, blend);
}

/*-------------------------------------------------------------------------------------------------
# Function `bsdfEvaluate`
> Evaluates the full BSDF model for the given material and set of directions.

You must provide `BsdfEvaluateData`'s `k1`, `k2`, and `xi` parameters.
(Evaluation is stochastic because this code randomly samples lobes depending on
`xi`; this is valid to do in a Monte Carlo path tracer.)

The diffuse lobe evaluation and the sum of the specular lobe evaluations
(including the cosine term from the rendering equation) will be returned in
`data.bsdf_diffuse` and `data.bsdf_glossy`. Additionally, the probability that
the sampling code will return this direction will be returned in `data.pdf`.
-------------------------------------------------------------------------------------------------*/
void bsdfEvaluate(NVSHADERS_INOUT_TYPE(BsdfEvaluateData) data, PbrMaterial mat)
{
  // Evaluate forward directly into the output; no dForward copy.
  const int   lobe  = bsdfEvaluateCore(data, mat);
  const float blend = mat.retroreflection;
  // Portsmouth et al. Sec. 5 / MaterialX: MRM applies to the base specular lobes only; keep
  // regular BTDF, clearcoat, and sheen unaffected.
  if(blend > 0.0f && isRetroreflectionLobe(lobe))
  {
    bsdfEvaluateRetroreflectionBlend(data, mat, blend);
  }
}

// One lobe with the MRM blend on the base lobes (diffuse, specular, metal): value and pdf are
// lerp(at k1, at reflect(-k1, N), b). reflect(-k1, N) has the same N.k1, so the weights and tint are
// the forward ones.
void bsdfEvaluateLobeBlended(NVSHADERS_INOUT_TYPE(BsdfEvaluateData) data, PbrMaterial mat, int lobe, float3 tint)
{
  bsdfEvaluateLobe(data, mat, lobe, tint);
  const float blend = mat.retroreflection;
  if(blend > 0.0f && isRetroreflectionLobe(lobe))
  {
    BsdfEvaluateData r;
    r.k1 = reflect(-data.k1, mat.N);
    r.k2 = data.k2;
    r.xi = data.xi;
    bsdfEvaluateLobe(r, mat, lobe, tint);
    data.bsdf_diffuse = lerp(data.bsdf_diffuse, r.bsdf_diffuse, blend);
    data.bsdf_glossy  = lerp(data.bsdf_glossy, r.bsdf_glossy, blend);
    data.pdf          = lerp(data.pdf, r.pdf, blend);
  }
}

// True when value and pdf are finite numbers.
bool bsdfLobeFinite(BsdfEvaluateData d)
{
  return !(isnan(d.pdf) || isinf(d.pdf) || any(isnan(d.bsdf_diffuse)) || any(isinf(d.bsdf_diffuse))
           || any(isnan(d.bsdf_glossy)) || any(isinf(d.bsdf_glossy)));
}

// Adds s * lobe(at.k1, at.k2) to acc. A lobe with s <= 0, a non-finite value or pdf, or pdf <= 0 adds
// nothing -- on both MIS sides alike, since both call this. Returns false only when `lobe` is
// `sampledLobe` and was dropped: that BSDF sample has no light-sample partner.
bool bsdfAccumulateLobe(NVSHADERS_INOUT_TYPE(BsdfEvaluateData) acc, BsdfEvaluateData at, PbrMaterial mat, int lobe, float s, float3 tint, int sampledLobe)
{
  if(!(s > 0.0f))  // zero-weight lobes cost one compare
    return lobe != sampledLobe;
  BsdfEvaluateData d;
  d.k1 = at.k1;
  d.k2 = at.k2;
  d.xi = at.xi;
  bsdfEvaluateLobeBlended(d, mat, lobe, tint);
  if(!bsdfLobeFinite(d) || !(d.pdf > 0.0f))
    return lobe != sampledLobe;
  acc.bsdf_diffuse += s * d.bsdf_diffuse;
  acc.bsdf_glossy += s * d.bsdf_glossy;
  acc.pdf += s * d.pdf;
  return true;
}

/*-------------------------------------------------------------------------------------------------
# Function `bsdfEvaluateSumWeighted`
> The whole BSDF at (k1, k2) for lobe weights `s` and `tint` that computeLobeWeights gave at k1:
> bsdf_diffuse/bsdf_glossy = occlusion * sum_j s_j g_j (the mean of bsdfEvaluate over xi.z) and
> pdf = q = sum_j s_j p_j. xi.z only picks the dispersion wavelength. Returns false if `sampledLobe`
> (>= 0) was dropped. Lobes are listed with constant indices so the weight array stays in registers.
-------------------------------------------------------------------------------------------------*/
bool bsdfEvaluateSumWeighted(NVSHADERS_INOUT_TYPE(BsdfEvaluateData) data, PbrMaterial mat, ARRAY_TYPE(float, LOBE_COUNT, s), float3 tint, int sampledLobe)
{
  BsdfEvaluateData acc;
  acc.k1           = data.k1;
  acc.k2           = data.k2;
  acc.xi           = data.xi;
  acc.bsdf_diffuse = float3(0.0f);
  acc.bsdf_glossy  = float3(0.0f);
  acc.pdf          = 0.0f;
  bool covered     = true;
  covered = bsdfAccumulateLobe(acc, data, mat, LOBE_CLEARCOAT_REFLECTION, s[LOBE_CLEARCOAT_REFLECTION], tint, sampledLobe) && covered;
  covered = bsdfAccumulateLobe(acc, data, mat, LOBE_SHEEN_REFLECTION, s[LOBE_SHEEN_REFLECTION], tint, sampledLobe) && covered;
  covered = bsdfAccumulateLobe(acc, data, mat, LOBE_METAL_REFLECTION, s[LOBE_METAL_REFLECTION], tint, sampledLobe) && covered;
  covered = bsdfAccumulateLobe(acc, data, mat, LOBE_SPECULAR_REFLECTION, s[LOBE_SPECULAR_REFLECTION], tint, sampledLobe) && covered;
  covered = bsdfAccumulateLobe(acc, data, mat, LOBE_SPECULAR_TRANSMISSION, s[LOBE_SPECULAR_TRANSMISSION], tint, sampledLobe) && covered;
  covered = bsdfAccumulateLobe(acc, data, mat, LOBE_DIFFUSE_TRANSMISSION, s[LOBE_DIFFUSE_TRANSMISSION], tint, sampledLobe) && covered;
  covered = bsdfAccumulateLobe(acc, data, mat, LOBE_DIFFUSE_REFLECTION, s[LOBE_DIFFUSE_REFLECTION], tint, sampledLobe) && covered;
  // Occlusion effect (glTF occlusion texture): the bsdf, never the pdf
  data.bsdf_diffuse = acc.bsdf_diffuse * mat.occlusion;
  data.bsdf_glossy  = acc.bsdf_glossy * mat.occlusion;
  data.pdf          = acc.pdf;
  return covered;
}

// bsdfEvaluateSumWeighted with the weights computed here, at data.k1.
bool bsdfEvaluateSum(NVSHADERS_INOUT_TYPE(BsdfEvaluateData) data, PbrMaterial mat, int sampledLobe)
{
  float3 tint = mat.baseColor;
  ARRAY_TYPE(float, LOBE_COUNT, s) = computeLobeWeights(mat, dot(data.k1, mat.N), tint);
  return bsdfEvaluateSumWeighted(data, mat, s, tint, sampledLobe);
}

/*-------------------------------------------------------------------------------------------------
# Function `bsdfEvaluateAll`
> Next-event estimation's evaluation of every lobe (see bsdfEvaluateSumWeighted). Its pdf is the lobe
> mixture's: for MIS, pair it with bsdfSampleMixture (or bsdfSampleCore(..., true)) on the BSDF side,
> never with bsdfSample.
-------------------------------------------------------------------------------------------------*/
void bsdfEvaluateAll(NVSHADERS_INOUT_TYPE(BsdfEvaluateData) data, PbrMaterial mat)
{
  bsdfEvaluateSum(data, mat, -1);
}


/*-------------------------------------------------------------------------------------------------
# Function `bsdfSampleLobe`
> Samples a pre-selected BSDF lobe (forward or retro view direction in `data.k1`).
-------------------------------------------------------------------------------------------------*/
void bsdfSampleLobe(NVSHADERS_INOUT_TYPE(BsdfSampleData) data, PbrMaterial mat, int lobe, float3 tint)
{
  data.pdf           = 0;
  data.bsdf_over_pdf = float3(0.0F);
  data.event_type    = BSDF_EVENT_ABSORB;

  if(lobe == LOBE_DIFFUSE_REFLECTION)
  {
    brdf_diffuse_sample(data, mat, tint);
  }
  else if(lobe == LOBE_DIFFUSE_TRANSMISSION)
  {
    // Per KHR_materials_diffuse_transmission, the BTDF color is `diffuseTransmissionColorFactor`,
    // independent of `baseColor`. Pass it directly; see brdf_diffuse_transmission_eval dispatch.
    brdf_diffuse_transmission_sample(data, mat, mat.diffuseTransmissionColorFactor);
  }
  else if(lobe == LOBE_SPECULAR_REFLECTION || lobe == LOBE_METAL_REFLECTION)
  {
    const BsdfLobeFresnel f = bsdfLobeFresnel(mat, lobe, dot(data.k1, mat.N), tint);
    brdf_ggx_smith_sample(data, mat, lobe, f.tint, f.fresnel, f.eta);
  }
  else if(lobe == LOBE_SPECULAR_TRANSMISSION)
  {
    const BsdfLobeFresnel f = bsdfLobeFresnel(mat, lobe, dot(data.k1, mat.N), tint);
    btdf_ggx_smith_sample(data, mat, f.tint, f.reflected);
  }
  else if(lobe == LOBE_CLEARCOAT_REFLECTION)
  {
    const BsdfLobeFresnel f = bsdfLobeFresnel(mat, lobe, dot(data.k1, mat.N), tint);
    brdf_ggx_smith_sample(data, bsdfClearcoatMaterial(mat), LOBE_CLEARCOAT_REFLECTION, f.tint, f.fresnel, f.eta);
  }
  else if(lobe == LOBE_SHEEN_REFLECTION)
  {
    // Sheen is using the state.sheenColor and state.sheenInvRoughness values directly.
    // Only brdf_sheen_sample needs a third random sample for the v-cavities flip. Put this as argument.
    brdf_sheen_sample(data, mat);
  }

  // Avoid internal reflection
  if(data.pdf <= 0.00001F || any(isnan(data.bsdf_over_pdf)))
  {
    data.event_type = BSDF_EVENT_ABSORB;
  }
  if((isnan(data.pdf) || isinf(data.pdf)) && data.event_type != BSDF_EVENT_ABSORB)
  {
    // Treat as a perfectly specular bounce; change GLOSSY to IMPULSE
    data.event_type = (data.event_type & (~BSDF_EVENT_GLOSSY)) | BSDF_EVENT_IMPULSE;
    data.pdf        = DIRAC;
  }
}

/*-------------------------------------------------------------------------------------------------
# Function `bsdfSampleRetroreflectionPath`
> Selects the forward or retroreflective MRM sampling path and returns its selection probability.
> Requires `blend > 0`.
-------------------------------------------------------------------------------------------------*/
float bsdfSampleRetroreflectionPath(NVSHADERS_INOUT_TYPE(BsdfSampleData) data, PbrMaterial mat, float blend)
{
  // Use xi.z for path selection, then decorrelate it before bsdfSampleCore uses it
  // for lobe selection and dispersion wavelength sampling.
  bool useRetro = data.xi.z < blend;
  data.xi.z     = rerandomize(data.xi.z);

  if(useRetro)
  {
    data.k1 = reflect(-data.k1, mat.N);
  }

  return useRetro ? blend : (1.0f - blend);
}

/*-------------------------------------------------------------------------------------------------
# Function `bsdfSampleCore`
> bsdfSample's body; returns the sampled lobe. Restores k1 and xi afterwards (the MRM path mirrors k1
> and both MRM and the lobe choice re-hash xi.z). With `mixturePdf`, `pdf` is the lobe mixture's q(k2)
> from bsdfEvaluateSumWeighted with this vertex's weights, the pdf next-event estimation's
> bsdfEvaluateAll weighs against, or DIRAC when no light sample is this direction's partner (an
> impulse, or a direction where the sampled lobe's own evaluation adds nothing). Direction, event and
> bsdf_over_pdf never depend on `mixturePdf`.
-------------------------------------------------------------------------------------------------*/
int bsdfSampleCore(NVSHADERS_INOUT_TYPE(BsdfSampleData) data, PbrMaterial mat, bool mixturePdf)
{
  // Select lobe with the forward view direction, then optionally apply MRM only on
  // retroreflection-eligible lobes (see isRetroreflectionLobe: diffuse/specular/metal
  // reflection, not transmission, clearcoat, or sheen).
  const float3 k1   = data.k1;
  const float3 xi   = data.xi;
  float3       tint = mat.baseColor;
  ARRAY_TYPE(float, LOBE_COUNT, s) = computeLobeWeights(mat, dot(k1, mat.N), tint);
  const int lobe = lobeFromWeights(s, xi.z);

  const float blend    = mat.retroreflection;
  const bool  hasRetro = blend > 0.0f && isRetroreflectionLobe(lobe);
  if(hasRetro)
  {
    data.xi.z = rerandomize(data.xi.z);
    bsdfSampleRetroreflectionPath(data, mat, blend);  // picks a path with probability = its blend weight; may mirror k1
  }
  bsdfSampleLobe(data, mat, lobe, tint);
  data.k1 = k1;
  data.xi = xi;
  if(data.event_type == BSDF_EVENT_ABSORB || data.pdf == DIRAC)
    return lobe;

  BsdfEvaluateData e;
  e.k1 = k1;
  e.k2 = data.k2;
  e.xi = xi;  // the original xi.z: a sampled transmission drew its wavelength from it (never the MRM path)
  if(mixturePdf)
  {
    const bool covered = bsdfEvaluateSumWeighted(e, mat, s, tint, lobe);
    data.pdf           = (covered && e.pdf > 0.0f) ? e.pdf : DIRAC;
  }
  else if(hasRetro)
  {
    // MRM: each path is picked as often as it weighs in the blend, so the throughput is the path's own
    // bsdf/pdf, undivided. The direction's density is the lobe's blended pdf, as bsdfEvaluate gives it.
    bsdfEvaluateLobeBlended(e, mat, lobe, tint);
    data.pdf = (bsdfLobeFinite(e) && e.pdf > 0.0f) ? e.pdf : DIRAC;  // not covered: no light-sample partner
  }
  return lobe;
}

/*-------------------------------------------------------------------------------------------------
# Function `bsdfSample`

> Samples the full BSDF model for the given material and input direction.

You must provide `BsdfSampleData`'s `k1` and `xi` parameters. This function
will set the other parameters of `BsdfSampleData`.

There are two things you should check after calling this function:
* Is `data.event_type` equal to `BSDF_EVENT_ABSORB`? If so, the sampler sampled
  an output direction that would be absorbed by the material (e.g. it chose a
  reflective lobe but sampled a vector below the surface).
  The light path ends here.
* Is `data.pdf` equal to `DIRAC`? If so, this sampled a perfectly specular lobe.
  If you're using Multiple Importance Sampling weights, you should compute them
  as if `data.pdf` was infinity.

`data.pdf` is the density of the sampled lobe alone, the one `bsdfEvaluate`
returns for that lobe. To weigh against `bsdfEvaluateAll` instead, use
`bsdfSampleMixture`.
-------------------------------------------------------------------------------------------------*/
void bsdfSample(NVSHADERS_INOUT_TYPE(BsdfSampleData) data, PbrMaterial mat)
{
  bsdfSampleCore(data, mat, false);
}

/*-------------------------------------------------------------------------------------------------
# Function `bsdfSampleMixture`
> bsdfSample with the pdf MIS needs when next-event estimation uses bsdfEvaluateAll: the lobe
> mixture's q at the sampled direction, or `DIRAC` where no light sample is its partner.
-------------------------------------------------------------------------------------------------*/
void bsdfSampleMixture(NVSHADERS_INOUT_TYPE(BsdfSampleData) data, PbrMaterial mat)
{
  bsdfSampleCore(data, mat, true);
}

/*-------------------------------------------------------------------------------------------------
# Simple BSDF Model
The functions below are used to evaluate and sample the BSDF for a simple PBR material,
without any additional lobes like clearcoat, sheen, etc. and without the need of random numbers.
This is based on the metallic/roughness BRDF in Appendix B of the glTF specification.
-------------------------------------------------------------------------------------------------*/

// Returns the probability that bsdfSampleSimple samples a glossy lobe.
float bsdfSimpleGlossyProbability(float NdotV, float metallic)
{
  return lerp(schlickFresnel(0.04F, 1.0F, NdotV), 1.0F, metallic);
}

/*-------------------------------------------------------------------------------------------------
# Function `bsdfEvaluateSimple`
> Evaluates the simple BSDF model using the given material and input and output directions.

You must provide `data.k1` and `data.k2`, but do not need to provide `data.xi`.
-------------------------------------------------------------------------------------------------*/
void bsdfEvaluateSimple(NVSHADERS_INOUT_TYPE(BsdfEvaluateData) data, PbrMaterial mat)
{
  // Specular reflection
  float3 H     = normalize(data.k1 + data.k2);
  float  NdotV = clampedDot(mat.N, data.k1);
  float  NdotL = clampedDot(mat.N, data.k2);
  float  VdotH = clampedDot(data.k1, H);
  float  NdotH = clampedDot(mat.N, H);

  if(NdotV == 0.0f || NdotL == 0.0f || VdotH == 0.0f || NdotH == 0.0f)
  {
    data.bsdf_diffuse = data.bsdf_glossy = float3(0.0f);
    data.pdf                             = 0.0f;
    return;
  }

  // We combine the metallic and specular lobes into a single glossy lobe.
  // The metallic weight is     metallic *    fresnel(f0 = baseColor)
  // The specular weight is (1-metallic) *    fresnel(f0 = c_min_reflectance)
  // The diffuse weight is  (1-metallic) * (1-fresnel(f0 = c_min_reflectance)) * baseColor

  // Fresnel terms
  float  c_min_reflectance = 0.04F;
  float3 f0                = lerp(float3(c_min_reflectance), mat.baseColor, mat.metallic);
  float3 fGlossy           = schlickFresnel(f0, float3(1.0F), VdotH);  // Metallic + specular
  float  fDiffuse          = schlickFresnel(1.0F - c_min_reflectance, 0.0F, VdotH) * (1.0F - mat.metallic);

  // Specular GGX
  float3 localH  = float3(dot(mat.T, H), dot(mat.B, H), NdotH);
  float  d       = hvd_ggx_eval(1.0f / mat.roughness, localH);
  float3 localK1 = float3(dot(mat.T, data.k1), dot(mat.B, data.k1), NdotV);
  float3 localK2 = float3(dot(mat.T, data.k2), dot(mat.B, data.k2), NdotL);
  float  G1 = 0.0f, G2 = 0.0f;
  ggx_smith_shadow_mask(G1, G2, localK1, localK2, mat.roughness);

  float diffusePdf  = M_1_PI * NdotL;
  float specularPdf = G1 * d * 0.25f / (NdotV * NdotH);
  data.pdf          = lerp(diffusePdf, specularPdf, bsdfSimpleGlossyProbability(NdotV, mat.metallic));

  data.bsdf_diffuse = mat.baseColor * fDiffuse * diffusePdf;  // Lambertian
  data.bsdf_glossy  = fGlossy * G2 * specularPdf;             // GGX-Smith
}

/*-------------------------------------------------------------------------------------------------
# Function `bsdfSampleSimple`
> Evaluates the simple BSDF model using the given material and input and output directions.

You must provide `data.k1` and `data.xi`. For one sample of pure reflection
(e.g. vk_mini_samples/ray_trace), use `data.xi == float2(0,0)`.

After calling this function, you should check if `data.event_type` is
`BSDF_EVENT_ABSORB`. If so, the sampling code sampled a direction below the
surface, and the light path ends here (it should be treated as a reflectance of 0).

This code cannot currently return a PDF of `DIRAC`, but that might change in
the future.
-------------------------------------------------------------------------------------------------*/
void bsdfSampleSimple(NVSHADERS_INOUT_TYPE(BsdfSampleData) data, PbrMaterial mat)
{
  float3 tint        = mat.baseColor;
  data.bsdf_over_pdf = float3(0.0F);

  float nk1 = clampedDot(mat.N, data.k1);
  if(data.xi.z <= bsdfSimpleGlossyProbability(nk1, mat.metallic))
  {
    // Glossy GGX
    data.event_type = BSDF_EVENT_GLOSSY_REFLECTION;
    // Transform to local space
    float3 localK1    = float3(dot(mat.T, data.k1), dot(mat.B, data.k1), nk1);
    float3 halfVector = hvd_ggx_sample_vndf(localK1, mat.roughness, data.xi.xy);
    // Transform from local space
    halfVector = mat.T * halfVector.x + mat.B * halfVector.y + mat.N * halfVector.z;
    data.k2    = reflect(-data.k1, halfVector);
  }
  else
  {
    // Diffuse
    data.event_type = BSDF_EVENT_DIFFUSE_REFLECTION;
    float3 localDir = cosineSampleHemisphere(data.xi.x, data.xi.y);
    data.k2         = mat.T * localDir.x + mat.B * localDir.y + mat.N * localDir.z;
  }

  BsdfEvaluateData evalData;
  evalData.k1 = data.k1;
  evalData.k2 = data.k2;
  bsdfEvaluateSimple(evalData, mat);
  data.pdf          = evalData.pdf;
  float3 bsdf_total = evalData.bsdf_diffuse + evalData.bsdf_glossy;
  if(data.pdf <= 0.00001F || any(isnan(bsdf_total)))
  {
    data.bsdf_over_pdf = float3(0.0f);
    data.event_type    = BSDF_EVENT_ABSORB;
  }
  else
  {
    data.bsdf_over_pdf = bsdf_total / data.pdf;
  }
}

/*-------------------------------------------------------------------------------------------------
# Function `bsdfSimpleAverageReflectance`
> Returns the approximate average reflectance of the Simple BSDF -- that is,
> `average_over_k2(f(k1, k2))` -- if GGX didn't lose energy.

This is useful for things like the variance reduction algorithm in
Tomasz Stachowiak's *Stochastic Screen-Space Reflections*; see also
Ray-Tracing Gems 1, chapter 32, *Accurate Real-Time Specular Reflections
with Radiance Caching*.
-------------------------------------------------------------------------------------------------*/
float3 bsdfSimpleAverageReflectance(float3 k1, PbrMaterial mat)
{
  float  NdotV             = clampedDot(mat.N, k1);
  float  c_min_reflectance = 0.04F;
  float3 f0                = lerp(float3(c_min_reflectance), mat.baseColor, mat.metallic);
  // This is approximate because
  // average_over_k2(fresnel(f0, 1.0f, VdotH)) != fresnel(f0, 1.0f, NdotV).
  float3 bsdf_glossy_average = schlickFresnel(f0, float3(1.0F), NdotV);
  float3 bsdf_diffuse_average = mat.baseColor * schlickFresnel(1.0F - c_min_reflectance, 0.0F, NdotV) * (1.0F - mat.metallic);
  return bsdf_glossy_average + bsdf_diffuse_average;
}


#endif  // NVVKHL_BSDF_FUNCTIONS_H
