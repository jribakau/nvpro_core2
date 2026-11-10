/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */


#ifndef TONEMAMP_FUNCTIONS_H
#define TONEMAMP_FUNCTIONS_H 1

#include "slang_types.h"
#include "tonemap_io.h.slang"
#include "functions.h.slang"

NAMESPACE_SHADERIO_BEGIN()

// Converts a color from linear RGB to sRGB.
inline float3 toSrgb(float3 rgb)
{
  float3 low  = rgb * 12.92f;
  float3 high = fma(pow(rgb, float3(1.0F / 2.4F)), float3(1.055F), float3(-0.055F));
  return lerp(low, high, float3(greaterThan(rgb, float3(0.0031308F))));
}

// Converts a color from sRGB to linear RGB.
inline float3 toLinear(float3 srgb)
{
  float3 low  = srgb / 12.92F;
  float3 high = pow((srgb + float3(0.055F)) / 1.055F, float3(2.4F));
  return lerp(low, high, float3(greaterThan(srgb, float3(0.04045F))));
}

// Converts from logarithmic photographer's EV100 ("exposure in stops") to
// linear scene luminance in candela-seconds/m^2.
inline float ev100Luminance(float ev100)
{
  // The constant factor doesn't really matter; the auto-exposure code only needs
  // a logarithmic scale, and using EV100 here gives values photographers are familiar with.
  // See https://en.wikipedia.org/wiki/Exposure_value#Relationship_of_EV_to_lighting_conditions
  // for the equation source, and historical background as to why this is called EV100.
  return exp2(ev100) / float(100.0f / 12.5f);
}

// Converts from linear scene luminance to photographer's EV100 ("exposure in stops").
// See ev100Luminance().
inline float luminanceEv100(float luminance)
{
  return log2(luminance * float(100.0f / 12.5f));
}

// Returns the luminance of a linear color in the BT.709 standard.
inline float bt709Luminance(float3 linearColor)
{
  return dot(linearColor, float3(0.2126, 0.7152, 0.0722));
}


/*-------------------------------------------------------------------------------------------------
# Function `tonemapFilmic`
> Filmic tonemapping operator by Jim Hejl and Richard Burgess-Dawson,
> approximating the Digital Fusion Cineon mode, but more saturated and with
> darker midtones. sRGB correction is built in.

http://filmicworlds.com/blog/filmic-tonemapping-operators/
-------------------------------------------------------------------------------------------------*/
inline float3 tonemapFilmic(float3 color)
{
  float3 temp = max(float3(0.0F), color - float3(0.004F));
  float3 result = (temp * (float3(6.2F) * temp + float3(0.5F))) / (temp * (float3(6.2F) * temp + float3(1.7F)) + float3(0.06F));
  return result;
}

/*-------------------------------------------------------------------------------------------------
# Function `tonemapUncharted`
> Tone mapping operator from Uncharted 2 by John Hable. sRGB correction is built in.

See: http://filmicworlds.com/blog/filmic-tonemapping-operators/
-------------------------------------------------------------------------------------------------*/

inline float3 tonemapUncharted2Impl(float3 color)
{
  const float a = 0.15F;
  const float b = 0.50F;
  const float c = 0.10F;
  const float d = 0.20F;
  const float e = 0.02F;
  const float f = 0.30F;
  return ((color * (a * color + c * b) + d * e) / (color * (a * color + b) + d * f)) - e / f;
}

inline float3 tonemapUncharted2(float3 color)
{
  const float W             = 11.2F;
  const float exposure_bias = 2.0F;
  color                     = tonemapUncharted2Impl(color * exposure_bias);
  float3 white_scale        = float3(1.0F) / tonemapUncharted2Impl(float3(W));
  // We apply pow() here instead of calling toSrgb to match the
  // original implementation.
  return pow(color * white_scale, float3(1.0F / 2.2F));
}

/*-------------------------------------------------------------------------------------------------
# Function `tonemapACES`
> An approximation by Stephen Hill to the Academy Color Encoding System's
> filmic curve for displaying HDR images on LDR output devices.

From https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl,
via https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
-------------------------------------------------------------------------------------------------*/
inline float3 tonemapACES(float3 color)
{
  // Input transform
  const float3x3 ACESInputMat = float3x3(0.59719F, 0.07600F, 0.02840F,   // Row 1
                                         0.35458F, 0.90834F, 0.13383F,   // Row 2
                                         0.04823F, 0.01566F, 0.83777F);  // Row 3
  color                       = mul(color, ACESInputMat);

  // RRT and ODT fit
  float3 a = color * (color + float3(0.0245786F)) - float3(0.000090537F);
  float3 b = color * (float3(0.983729F) * color + float3(0.4329510F)) + float3(0.238081F);
  color    = a / b;  // Always OK because of the large constant term in b's polynomial
  // Output transform
  const float3x3 ACESOutputMat = float3x3(1.60475F, -0.10208F, -0.00327F,  //
                                          -0.53108F, 1.10813F, -0.07276F,  //
                                          -0.07367F, -0.00605F, 1.07602F);
  color                        = mul(color, ACESOutputMat);
  return toSrgb(color);
}

/*-------------------------------------------------------------------------------------------------
# Function `tonemapAgX`
> Benjamin Wrensch's approximation to the AgX tone mapping curve by Troy Sobotka.

From https://iolite-engine.com/blog_posts/minimal_agx_implementation
-------------------------------------------------------------------------------------------------*/
inline float3 tonemapAgX(float3 color)
{
  // Input transform
  const float3x3 agx_mat = float3x3(0.842479062253094F, 0.0423282422610123F, 0.0423756549057051F,  //
                                    0.0784335999999992F, 0.878468636469772F, 0.0784336F,           //
                                    0.0792237451477643F, 0.0791661274605434F, 0.879142973793104F);
  color                  = mul(color, agx_mat);

  // Log2 space encoding
  const float min_ev = -12.47393f;
  const float max_ev = 4.026069f;
  color              = clamp(log2(color), min_ev, max_ev);
  color              = (color - min_ev) / (max_ev - min_ev);

  // Apply 6th-order sigmoid function approximation
  float3 v = fma(float3(15.5F), color, float3(-40.14F));
  v        = fma(color, v, float3(31.96F));
  v        = fma(color, v, float3(-6.868F));
  v        = fma(color, v, float3(0.4298F));
  v        = fma(color, v, float3(0.1191F));
  v        = fma(color, v, float3(-0.0023F));

  // Output transform
  const float3x3 agx_mat_inv = float3x3(1.19687900512017F, -0.0528968517574562F, -0.0529716355144438F,  //
                                        -0.0980208811401368F, 1.15190312990417F, -0.0980434501171241F,  //
                                        -0.0990297440797205F, -0.0989611768448433F, 1.15107367264116F);
  v                          = mul(v, agx_mat_inv);

  // Skip the pow(..., float3(2.2)), because we want sRGB output here.
  return v;
}

/*-------------------------------------------------------------------------------------------------
# Function `tonemapKhronosPBR`
> The Khronos PBR neutral tone mapper.

Adapted from https://github.com/KhronosGroup/ToneMapping/blob/main/PBR_Neutral/pbrNeutral.glsl
-------------------------------------------------------------------------------------------------*/
inline float3 tonemapKhronosPBR(float3 color)
{
  const float startCompression = 0.8F - 0.04F;
  const float desaturation     = 0.15F;

#ifdef __cplusplus
  float x    = glm::min(color.x, glm::min(color.y, color.z));
  float peak = glm::max(color.x, glm::max(color.y, color.z));
#else
  float x    = min(color.x, min(color.y, color.z));
  float peak = max(color.x, max(color.y, color.z));
#endif

  float offset = x < 0.08F ? x * (-6.25F * x + 1.F) : 0.04F;
  color -= offset;

  if(peak >= startCompression)
  {
    const float d       = 1.F - startCompression;
    float       newPeak = 1.F - d * d / (peak + d - startCompression);
    color *= newPeak / peak;

    float g = 1.F - 1.F / (desaturation * (peak - newPeak) + 1.F);
    color   = lerp(color, float3(newPeak), float3(g));
  }
  return toSrgb(color);
}

/*-------------------------------------------------------------------------------------------------
# Function `tonemapShaped`
> A filmic curve with two handles, worked in stops: `u` stops over mid grey in (0.18), `Y` stops over
> mid grey out. Mid grey comes out at 0.14, with a slope of 1.3 stops out a stop in, as Khronos PBR
> Neutral gives it, so switching to it from Neutral keeps a picture's exposure. Above mid grey a
> rational quadratic Bezier from there, along that slope, to white at 16 (SHAPED_WHITE), which it
> reaches flat: its middle point's weight is the shoulder (`shoulder` 0: 4, highlights bright and
> rolling off late, about as Neutral's; 1: a quarter, rolling off early and gently). Below mid grey the
> same slope, steepened past 1.5 stops down by up to 1.5 more stops a stop (`toe` 0 open shadows, 1
> crushed). Each handle moves its own side of mid grey only, and the curve rises for every value of
> both and is white by 16, which the firefly clamp counts on (fv_lighting.cpp kFireflyClamp).
>
> Applied to the brightest channel with the hue kept, then eased to white as that channel is
> compressed (Neutral's way), so a bright red stays red on its way to white rather than turning orange.
-------------------------------------------------------------------------------------------------*/
static const float SHAPED_MID_IN       = 0.18F;
static const float SHAPED_MID_OUT      = 0.14F;
static const float SHAPED_SLOPE        = 1.3F;   // stops out a stop in, at mid grey
static const float SHAPED_WHITE        = 16.0F;  // the value that comes out white
static const float SHAPED_TOE_START    = 1.5F;   // stops under mid grey where the toe bends
static const float SHAPED_TOE_KNEE     = 0.5F;   // the bend's half-width, stops
static const float SHAPED_TOE_MAX      = 1.5F;   // the steepening at a toe of 1, stops out a stop in
static const float SHAPED_DESATURATION = 0.15F;  // Neutral's: how fast a compressed colour goes white

inline float shapedMax(float a, float b)
{
  return a > b ? a : b;
}
inline float shapedMin(float a, float b)
{
  return a < b ? a : b;
}

// The Bezier's middle weight for a shoulder of 0 to 1: 4 (hard) to 1/4 (soft), 1 halfway
inline float shapedShoulderWeight(float shoulder)
{
  return exp2(2.0F - 4.0F * shapedMin(shapedMax(shoulder, 0.0F), 1.0F));
}

// Stops out for `u` stops in, from mid grey to white (0 <= u <= log2(SHAPED_WHITE / SHAPED_MID_IN)):
// the Bezier P0 (0, 0), P1 (yw / SHAPED_SLOPE, yw), P2 (uw, yw), P1 weighed `weight`, solved for its
// parameter at u (the root in [0, 1] of a quadratic, in the form that stays stable as u goes to 0)
inline float shapedShoulderStops(float u, float weight)
{
  const float yw   = log2(1.0F / SHAPED_MID_OUT);
  const float uw   = log2(SHAPED_WHITE / SHAPED_MID_IN);
  const float x1   = yw / SHAPED_SLOPE;
  const float w    = weight;
  const float a    = u * (2.0F - 2.0F * w) + 2.0F * w * x1 - uw;
  const float b    = u * (2.0F * w - 2.0F) - 2.0F * w * x1;
  const float disc = sqrt(shapedMax(b * b - 4.0F * a * u, 0.0F));
  const float t    = shapedMin(shapedMax(2.0F * u / shapedMax(disc - b, 1e-12F), 0.0F), 1.0F);
  const float d    = (1.0F - t) * (1.0F - t) + 2.0F * w * t * (1.0F - t) + t * t;
  return (2.0F * w * t * (1.0F - t) + t * t) * yw / d;
}

// Stops out for `u` stops in under mid grey (u <= 0): the mid slope, steepened past the toe's start
inline float shapedToeStops(float u, float toe)
{
  const float v    = -u - SHAPED_TOE_START;
  const float h    = SHAPED_TOE_KNEE;
  const float bent = v <= -h ? 0.0F : (v >= h ? v : (v + h) * (v + h) / (4.0F * h));
  return SHAPED_SLOPE * u - SHAPED_TOE_MAX * shapedMin(shapedMax(toe, 0.0F), 1.0F) * bent;
}

// The curve on one value: linear in, linear display out (before the sRGB encoding), 0 to 1
inline float shapedCurve(float x, float toe, float shoulder)
{
  if(!(x > 0.0F))
    return 0.0F;
  const float u  = log2(x / SHAPED_MID_IN);
  const float uw = log2(SHAPED_WHITE / SHAPED_MID_IN);
  if(u >= uw)
    return 1.0F;
  const float y = u >= 0.0F ? shapedShoulderStops(u, shapedShoulderWeight(shoulder)) : shapedToeStops(u, toe);
  return shapedMin(SHAPED_MID_OUT * exp2(y), 1.0F);
}

// The curve on a colour, linear display out: the brightest channel through it, the hue kept, then
// eased to white by how much that channel was compressed
inline float3 tonemapShapedLinear(float3 color, float toe, float shoulder)
{
  color            = max(color, float3(0.0F));
  const float peak = shapedMax(color.x, shapedMax(color.y, color.z));
  if(!(peak > 0.0F))
    return float3(0.0F);
  const float newPeak = shapedCurve(peak, toe, shoulder);
  if(peak > 3.0e38F)  // an infinite pixel: white
    return float3(1.0F);
  color *= newPeak / peak;
  const float g = 1.0F - 1.0F / (SHAPED_DESATURATION * shapedMax(peak - newPeak, 0.0F) + 1.0F);
  return lerp(color, float3(newPeak), float3(g));
}

inline float3 tonemapShaped(float3 color, float toe, float shoulder)
{
  return toSrgb(tonemapShapedLinear(color, toe, shoulder));
}

// Computes the CIE 1960 chromaticity coordinates of a light source given
// temperature and tint.
//
// ANSI C78.377-2008 defines light colors in terms of a correlated color
// temperature `t` of a blackbody, plus a tint shift `duv`, which is the
// distance from the "Planckian locus" of blackbody chromaticities in the
// CIE 1960 color space; negative values are more magenta.
//
// Note that the D65 Illuminant D white point used for sRGB cooresponds to
// `t == 6506.11144` and `duv = 0.00325895312`, i.e. daylight is very slightly
// green compared to a blackbody, which matches the standard.
//
// This parameterization of white balance/tint is rather physically based, but
// might not be the best for artists -- in your own apps, you may wish to
// provide RGB white balance parameters instead.
inline float2 cie1960LightColor(float t, float duv)
{
  // The approximation for the locus we use here is from Michael P.Krystek,
  // "An algorithm to calculate correlated colour temperature"(1985),
  // as it appears in https: // en.wikipedia.org/wiki/Planckian_locus#Approximation .
  // We compute the derivative analytically.
  // As a result of this approximation, D65 is at 6506 rather than the usual
  // 6504 K.
  float  rcpUDenom = 1.0F / ((7.08145163e-7F * t + 8.42420235e-4F) * t + 1.F);
  float  rcpVDenom = 1.0F / ((1.61456053e-7F * t - 2.89741816e-5F) * t + 1.F);
  float2 uv        = float2(((1.28641212e-7F * t + 1.54118254e-4F) * t + 0.860117757F) * rcpUDenom,  //
                            ((4.20481691e-8F * t + 4.22806245e-5F) * t + 0.317398726F) * rcpVDenom);

  // For the derivative, note that we can re - use the denominator, since
  // d/dt f(t) / g(t) = (f'(t)g(t) - f(t)g'(t)) / g(t)^2:
  float2 derivative =
      float2(((-7.68136056380576e-13F * t - 9.60894034459919e-7F) * t - 0.000570462348979613F) * rcpUDenom * rcpUDenom,  //
             ((-8.04477403759601e-12F * t - 1.8395552854377e-8F) * t + 5.14769928267326e-5F) * rcpVDenom * rcpVDenom);
  derivative = normalize(derivative);  // Normalize it

  // Apply the duv offset
  uv += duv * float2(derivative.y, -derivative.x);
  return uv;
}

// Converts from CIE 1960 (u, v) chromaticity to CIE 1931 XYZ, normalizing
// to a brightness of Y=1.
inline float3 cie1931XyzFromCie1960UV(float2 uv)
{
  // From https://en.wikipedia.org/wiki/CIE_1960_color_space#Relation_to_CIE_XYZ
  float2 xy = uv * float2(3.f, 2.f) / (2.f * uv.x - 8.f * uv.y + 4.f);
  // From https://en.wikipedia.org/wiki/CIE_1931_color_space#CIE_xyY_color_space :
  // For low color temperatures and high Duv, the light color might get a
  // negative z = 1 - x - y value, which is totally impossible.
  // Scale the xy value back in to prevent this:
  xy /= max(1.f, xy.x + xy.y);
  // Convert to (x, y, 1-x-y) then divide by y
  float rcpY = 1.f / xy.y;
  return float3(xy.x * rcpY, 1.f, (1.f - xy.x) * rcpY - 1.f);
}

/*-------------------------------------------------------------------------------------------------
# Advanced Color Grading Functions
-------------------------------------------------------------------------------------------------*/

// Saturation: scales colors away from the gray of the same luminance, in linear light. `color` is
// sRGB-encoded, as the SDR grading takes it, and so is the result. BT.709's weights are for linear
// values: on encoded ones they make saturated colours too dark a gray (pure red 0.21 instead of
// 0.50, https://30fps.net/pages/better-srgb-to-greyscale/), and a scale there changes the
// luminance as well. Here every saturation keeps it, up to where a channel would go below 0.
// sat > 1 increases saturation, sat = 1 applies no change, sat = 0 converts to grayscale.
inline float3 adjustSaturation(float3 color, float sat)
{
  if(sat == 1.0F)
    return color;
  const float3 lin = toLinear(color);
  const float  lum = luminance(lin);
  return toSrgb(max(lerp(float3(lum), lin, float3(sat)), float3(0.0F)));
}

// Vibrance: Boosts saturation of desaturated colors while preserving already-saturated colors.
// Unlike saturation which affects all colors equally, vibrance is selective.
inline float3 adjustVibrance(float3 color, float vibrance)
{
  float maxc = max(color.x, max(color.y, color.z));
  float minc = min(color.x, min(color.y, color.z));
  float sat  = maxc - minc;
  return lerp(adjustSaturation(color, 1.0F + vibrance), color, float3(saturate(sat)));
}

// Shadow/Midtone/Highlight Adjustment: Selectively adjusts different tonal ranges.
// s, m, h are bias values typically in range [-1, 1]
inline float3 smhAdjust(float3 color, float shadowBias, float midtoneBias, float highlightBias)
{
  float lum = luminance(color);
  
  // Compute weights for each tonal range using smooth transitions
  float shadow    = smoothstep(0.0F, 0.5F, 1.0F - lum);
  float highlight = smoothstep(0.0F, 0.7F, lum);
  float midtone   = 1.0F - shadow - highlight;
  
  // Apply weighted adjustments
  return color * (1.0F + shadow * shadowBias + midtone * midtoneBias + highlight * highlightBias);
}

// Split Toning: Applies different color tints to shadows and highlights.
// balance shifts the transition point between cool and warm tones.
inline float3 splitToning(float3 color, float3 coolColor, float3 warmColor, float balance)
{
  float lum = luminance(color);
  float t = smoothstep(0.0F, 1.0F, lum + balance);
  return color * lerp(coolColor, warmColor, float3(t));
}

/*-------------------------------------------------------------------------------------------------
# Looks
> Creative grading on the sRGB-encoded image, after the tone curve, as a colourist's tools work on
> display values. Every one is the identity at its default (tonemap_io.h.slang).
-------------------------------------------------------------------------------------------------*/

// Whether all three channels are `x`: the GLSL-free spelling of all(v == x), which C++'s glm
// compares as a whole vector
inline bool allEqual(float3 v, float x)
{
  return v.x == x && v.y == x && v.z == x;
}

// Black and white through a colour filter: the image's linear light weighted by `weights` (a red
// filter's brighten skin and darken a blue sky), `amount` of it over the colour.
inline float3 monoAdjust(float3 color, float3 weights, float amount)
{
  const float y = max(dot(toLinear(color), weights), 0.0F);
  return lerp(color, toSrgb(float3(y)), float3(amount));
}

// Lift, gamma and gain a channel: lift is added to the shadows and fades to nothing at white, gain
// scales everything (white most), gamma then bends the midtones (above 1 lighter).
inline float3 liftGammaGain(float3 color, float3 lift, float3 gamma, float3 gain)
{
  const float3 c = gain * (color + lift * (float3(1.0F) - color));
  return pow(max(c, float3(0.0F)), float3(1.0F) / gamma);
}

// Each channel to `levels` steps, `amount` of it over the image.
inline float3 posterizeAdjust(float3 color, float levels, float amount)
{
  const float  steps = levels - 1.0F;
  const float3 p     = round(saturate(color) * float3(steps)) / float3(steps);
  return lerp(color, p, float3(amount));
}

// A cell of the grain, for the frame's seed: [0, 1)
inline float grainHash(int2 cell, uint32_t seed)
{
  uint32_t h = uint32_t(cell.x) * 0x8DA6B343u + uint32_t(cell.y) * 0xD8163841u + seed * 0xCB1AB31Fu;
  h ^= h >> 16;
  h *= 0x7FEB352Du;
  h ^= h >> 15;
  h *= 0x846CA68Bu;
  h ^= h >> 16;
  return float(h >> 8) * (1.0F / 16777216.0F);
}

// Film grain, the same in every channel: smooth value noise on cells 1.3 pixels across at 1080
// lines, larger on a larger image so a picture's grain is the same at every size (and off the pixel
// grid, so the cells don't show), strongest in the midtones as a negative's is. `amount` 1 is heavy:
// a mid grey moves by up to about 0.2.
inline float3 filmGrain(float3 color, float amount, float2 pixel, float2 imageSize, uint32_t seed)
{
  const float  cellSize = 1.3F * max(imageSize.y / 1080.0F, 1.0F);
  const float2 p        = (pixel + float2(0.5F)) / cellSize;
  const float2 i        = floor(p);
  const float2 f        = p - i;
  const float2 u        = f * f * (float2(3.0F) - 2.0F * f);
  const int2   c        = int2(i);
  const float  n00      = grainHash(c, seed);
  const float  n10      = grainHash(c + int2(1, 0), seed);
  const float  n01      = grainHash(c + int2(0, 1), seed);
  const float  n11      = grainHash(c + int2(1, 1), seed);
  const float  n        = (n00 + (n10 - n00) * u.x) + ((n01 + (n11 - n01) * u.x) - (n00 + (n10 - n00) * u.x)) * u.y;
  const float  y        = luminance(saturate(color));
  const float  midtones = 0.35F + 2.6F * y * (1.0F - y);  // 0.35 at black and at white, 1 at mid grey
  return color + float3((n * 2.0F - 1.0F) * 0.2F * amount * midtones);
}

// A small sensor's response in place of a tone curve's shoulder: straight up to white, where each
// channel clips on its own (a bright warm light goes yellow, then white, with a hard edge), exposed so
// that middle grey lands where the curve has it. `curveGrey` is the curve's value for 0.18, linear.
// sRGB-encoded, as the curves' results.
inline float3 sensorClipCurve(float3 color, float curveGrey)
{
  return toSrgb(saturate(color * (curveGrey / 0.18F)));
}

// A cell of sensorNoise: smooth value noise in [0, 1) at `p`, in cells
inline float sensorNoiseCell(float2 p, uint32_t seed)
{
  const float2 i   = floor(p);
  const float2 f   = p - i;
  const float2 u   = f * f * (float2(3.0F) - 2.0F * f);
  const int2   c   = int2(i);
  const float  n00 = grainHash(c, seed);
  const float  n10 = grainHash(c + int2(1, 0), seed);
  const float  n01 = grainHash(c + int2(0, 1), seed);
  const float  n11 = grainHash(c + int2(1, 1), seed);
  const float  top = n00 + (n10 - n00) * u.x;
  const float  bot = n01 + (n11 - n01) * u.x;
  return top + (bot - top) * u.y;
}

// How much of sensorNoise a pixel of brightness `y` (0 to 1) takes: all of it at black, a quarter at white
inline float sensorNoiseWeight(float y)
{
  const float dark = 1.0F - saturate(y);
  return 0.25F + 0.75F * dark * dark;
}

// A small sensor's noise at high gain, unlike film's grain: strongest in the shadows, and coloured:
// a brightness noise on cells about a pixel across at 1080 lines, and a noise of each channel's own on
// cells twice that (the colour blotches), both larger on a larger image so a picture's noise is the
// same at every size. `amount` 1 is heavy: a near-black moves by up to about 0.17.
inline float3 sensorNoise(float3 color, float amount, float2 pixel, float2 imageSize, uint32_t seed)
{
  const float  scale  = max(imageSize.y / 1080.0F, 1.0F);
  const float2 fine   = (pixel + float2(0.5F)) / (1.1F * scale);
  const float2 coarse = (pixel + float2(0.5F)) / (2.3F * scale);
  const float  light  = sensorNoiseCell(fine, seed + 0x51ED270Bu) * 2.0F - 1.0F;
  const float3 tint   = float3(sensorNoiseCell(coarse, seed + 0x2545F491u), sensorNoiseCell(coarse, seed + 0x9E3779B9u),
                               sensorNoiseCell(coarse, seed + 0x7F4A7C15u))
                      * 2.0F - float3(1.0F);
  const float  weight = sensorNoiseWeight(luminance(saturate(color)));
  return color + (float3(0.6F * light) + 0.5F * tint) * (0.15F * amount * weight);
}

// Returns a matrix that corrects a light source with the given Kelvin
// temperature and Duv tint to RGB (1,1,1), then multiplies by exposure.
// Use it like mul(inputMatrix, color).
inline float3x3 getColorCorrectionMatrix(float exposure, float temperature, float tint)
{
  // Suesstrunk, Holm, and Finlayson's paper "Chromatic Adaptation
  // Performance of Different RGB Sensors" (2001,
  // https://web.archive.org/web/20061018020916/http://infoscience.epfl.ch/getfile.py?mode=best&recid=34049 )
  // found that the CMCCAT2000 space performed the best at mapping colors
  // between scenes with different illuminations, based on data from 16
  // data sets.
  //
  // Following their approach, we'll convert colors and the light source
  // color to CMCCAT2000, divide by the whitepoint and apply exposure, and
  // then convert back to linear RGB.
  // Since these are all linear operations, it's the same as multiplying by
  // a single 3x3 matrix.
  //
  // Here are the matrices we'll need:
  const float3x3 kCmccat2000FromCie1931Xyz = float3x3(0.7982F, 0.3389F, -0.1371F,              // Row 1
                                                      -0.5918F, 1.5512F, 0.0406F,              // Row 2
                                                      0.0008F, 0.0239F, 0.9753F);              // Row 3
  const float3x3 kCmccat2000FromRgb        = float3x3(0.39858179F, 0.51147528F, 0.03823013F,   //
                                                      0.08651038F, 0.90263008F, 0.04376704F,   //
                                                      0.02423435F, 0.13363512F, 0.92889263F);  //
  const float3x3 kRgbFromCmccat2000        = float3x3(2.86209568F, -1.61563711F, -0.04166966,  //
                                                      -0.27259146F, 1.26953249F, -0.04859816,  //
                                                      -0.0354543F, -0.14049009F, 1.0846294);

  // Whitepoint to CMCCAT2000
  float3 whitepointXYZ        = cie1931XyzFromCie1960UV(cie1960LightColor(temperature, tint));
  float3 whitepointCmccat2000 = mul(kCmccat2000FromCie1931Xyz, whitepointXYZ);
  // Componentwise multiplier.
  // Note that even for D65, whitepointCmccat2000 isn't [1,1,1], so we need to
  // apply a normalization factor. This is mul(kCmccat2000FromRgb, float3(1,1,1)):
  float3   middleFactor = exposure * (float3(0.948287308f, 1.03290749f, 1.08676255f) / whitepointCmccat2000);
  float3x3 right        = float3x3(middleFactor[0] * kCmccat2000FromRgb[0],  //
                                   middleFactor[1] * kCmccat2000FromRgb[1],  //
                                   middleFactor[2] * kCmccat2000FromRgb[2]);
  // Fold all the matrices together
  float3x3 result = mul(kRgbFromCmccat2000, right);
  // Note that for the default exposure, temperature, and tint we now have the
  // identity matrix to within 2e-7.
  return result;
}

// Applies the given tone mapper and color grading settings to a given color.
// Requires the pixel index so it can dither,
// and the screen size so that it can apply vignetting.
/*-------------------------------------------------------------------------------------------------
# Function `falseColorExposure`
> The false-colour exposure view's colour for a pixel, sRGB-encoded as applyTonemap's result:
> `luminance` is the exposed scene's (after exposure and white balance, what the curve takes), `shown`
> the image's graded colour there. Whole stops around middle grey (0.18) in nine bands of inferno,
> each a stop wide and centred on its stop (falseColorBand): -4 and below dark violet, 0 the middle
> band's red, +4 and above pale yellow. Where the image clips (a channel at its top) green, where it
> crushes (every channel under half an 8-bit step) blue, whatever the stop. The app's legend
> (src/components/DisplayMenu.svelte) lists the same colours.
-------------------------------------------------------------------------------------------------*/
inline int falseColorBand(float luminance)
{
  const float stops = log2(max(luminance, 1e-12F) / 0.18F);
  return int(clamp(floor(stops + 0.5F), -4.0F, 4.0F)) + 4;
}

inline float3 falseColorExposure(float luminance, float3 shown)
{
  const float top = max(shown.x, max(shown.y, shown.z));
  if(top >= 254.5F / 255.F)
    return float3(0.F, 255.F, 77.F) / 255.F;  // clipped
  if(top < 0.5F / 255.F)
    return float3(26.F, 115.F, 255.F) / 255.F;  // crushed
  switch(falseColorBand(luminance))
  {
    case 0:
      return float3(22.F, 11.F, 57.F) / 255.F;
    case 1:
      return float3(66.F, 10.F, 104.F) / 255.F;
    case 2:
      return float3(106.F, 23.F, 110.F) / 255.F;
    case 3:
      return float3(147.F, 38.F, 103.F) / 255.F;
    case 4:
      return float3(188.F, 55.F, 84.F) / 255.F;  // middle grey
    case 5:
      return float3(221.F, 81.F, 58.F) / 255.F;
    case 6:
      return float3(243.F, 120.F, 25.F) / 255.F;
    case 7:
      return float3(252.F, 165.F, 10.F) / 255.F;
    default:
      return float3(246.F, 215.F, 70.F) / 255.F;
  }
}

// The tone curve `tm` names, for a colour with exposure and white balance applied: sRGB-encoded
inline float3 tonemapCurve(TonemapperData tm, float3 color)
{
  float3 c = color;
  switch(tm.method)
  {
    case TONEMAP_METHOD(eFilmic):
      c = tonemapFilmic(color);
      break;
    case TONEMAP_METHOD(eUncharted2):
      c = tonemapUncharted2(color);
      break;
    case TONEMAP_METHOD(eClip):
      c = toSrgb(color);
      break;
    case TONEMAP_METHOD(eACES):
      c = tonemapACES(color);
      break;
    case TONEMAP_METHOD(eAgX):
      c = tonemapAgX(color);
      break;
    case TONEMAP_METHOD(eKhronosPBR):
      c = tonemapKhronosPBR(color);
      break;
    case TONEMAP_METHOD(eShaped):
      c = tonemapShaped(color, tm.shapedToe, tm.shapedShoulder);
      break;
  }
  return c;
}

inline float3 applyTonemap(TonemapperData tm, float3 color, float2 pixel, float2 imageSize)
{
  // Exposure and white balance.
  // These are combined into a matrix and applied together before tonemapping
  // because they're correcting for scene lighting and camera parameters.
  color = mul(tm.inputMatrix, color);
  const float exposed = luminance(color);  // the false-colour view's

  // Tonemap
  float3 c = tonemapCurve(tm, color);
  // a small sensor's highlights in place of the curve's, middle grey where the curve has it
  if(tm.sensorClip > 0.0F)
  {
    const float grey = toLinear(saturate(tonemapCurve(tm, float3(0.18F)))).y;
    c                = lerp(c, sensorClipCurve(color, grey), float3(tm.sensorClip));
  }

  // SDR color grading.
  
  // contrast and clamp (just in case)
  c = clamp(lerp(float3(0.5F), c, float3(tm.contrast)), float3(0.F), float3(1.F));
  
  // brightness
  c = pow(c, float3(1.0F / tm.brightness));
  
  // saturation
  c = adjustSaturation(c, tm.saturation);

  // black and white, through a colour filter
  if(tm.mono > 0.0F)
  {
    c = monoAdjust(c, tm.monoMix, tm.mono);
  }

  // Advanced color grading
  // vibrance (selective saturation boost for desaturated colors)
  if(tm.vibrance != 0.0F)
  {
    c = adjustVibrance(c, tm.vibrance);
  }
  // shadow/midtone/highlight adjustments
  if(tm.shadowBias != 0.0F || tm.midtoneBias != 0.0F || tm.highlightBias != 0.0F)
  {
    c = smhAdjust(c, tm.shadowBias, tm.midtoneBias, tm.highlightBias);
  }
  // lift, gamma and gain
  if(!allEqual(tm.lift, 0.0F) || !allEqual(tm.gamma, 1.0F) || !allEqual(tm.gain, 1.0F))
  {
    c = liftGammaGain(c, tm.lift, tm.gamma, tm.gain);
  }
  // split toning (cool shadows, warm highlights)
  if(tm.coolColor.x != 1.0F || tm.coolColor.y != 1.0F || tm.coolColor.z != 1.0F ||
     tm.warmColor.x != 1.0F || tm.warmColor.y != 1.0F || tm.warmColor.z != 1.0F)
  {
    c = splitToning(c, tm.coolColor, tm.warmColor, tm.splitBalance);
  }
  // posterising
  if(tm.posterize >= 2.0F && tm.posterizeMix > 0.0F)
  {
    c = posterizeAdjust(c, tm.posterize, tm.posterizeMix);
  }

  // vignette
  float2 center_uv = (pixel / imageSize) * 2.0F - 1.0F;
  c *= 1.0F - dot(center_uv, center_uv) * tm.vignette;

  // The exposure view: the stops of what the image shows, undithered and without grain
  if(tm.falseColor != 0)
    return falseColorExposure(exposed, c);

  // film grain
  if(tm.grain > 0.0F)
  {
    c = filmGrain(c, tm.grain, pixel, imageSize, tm.grainSeed);
  }
  // a small sensor's noise
  if(tm.sensorNoise > 0.0F)
  {
    c = sensorNoise(c, tm.sensorNoise, pixel, imageSize, tm.grainSeed);
  }

  // Dither assuming 8-bit output.
  if(tm.dither == 1)
  {
    const float levelsMinus1 = 255.F;
    // Based on Fardan Niqbal's survey at https://www.shadertoy.com/view/cl2GRm,
    // we use R2 noise in [0, 1]:
    float noise = fract(dot(pixel, float2(0.245122331F, 0.430159704F)));
    // routed through a triangular wave to [-.5,.5]:
    noise = .5F - 2.F * abs(noise - .5F);
    // shaped so its probability distribution function is a triangle over [-1, 1]:
    const float trinoise = sign(noise) * (1.F - sqrt(1.F - 2.F * abs(noise)));
    // We use triangular noise for most of the range (so that the amount of
    // noise appears uniform) but switch to uniform at the ends:
    const bool3 lowEnd  = greaterThan(float3(.5F / levelsMinus1), c);
    const bool3 highEnd = greaterThan(c, float3(1.F - .5F / levelsMinus1));
#if defined(GL_core_profile)  // GLSL has no || for boolean vectors
    const bool3 useUniform = bool3(lowEnd.x || highEnd.x, lowEnd.y || highEnd.y, lowEnd.z || highEnd.z);
#else
    const bool3 useUniform = lowEnd || highEnd;
#endif
    // Note that because we add noise in sRGB space, we gain an average of
    // 4 * 10^-6 luminance (roughly 0.279/L^2 where L is the number of levels).
    // http://www.thetenthplanet.de/archives/5367 does energy-preserving dither,
    // but with a box instead of a triangular kernel, so the amount of noise
    // has a banding pattern over the image.
    c += lerp(float3(trinoise), float3(noise), float3(useUniform)) / levelsMinus1;
  }

  return c;
}

NAMESPACE_SHADERIO_END()

#endif  // TONEMAMP_FUNCTIONS_H
