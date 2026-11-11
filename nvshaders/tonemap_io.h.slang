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


#ifndef TONEMAMP_SHADERIO_H
#define TONEMAMP_SHADERIO_H 1

#include "slang_types.h"

NAMESPACE_SHADERIO_BEGIN()

#define TONEMAP_WORKGROUP_SIZE 16


#if defined(GL_core_profile)  // GLSL
// GLSL has no enums, so the tone map methods become macros. They carry the enum name as
// prefix to keep the global macro namespace clean; TONEMAP_METHOD(eFilmic) resolves to
// the macro here and to ToneMapMethod::eFilmic everywhere else.
#define TONEMAP_METHOD(name) ToneMapMethod_##name
#define ToneMapMethod_eFilmic 0
#define ToneMapMethod_eUncharted2 1
#define ToneMapMethod_eClip 2
#define ToneMapMethod_eACES 3
#define ToneMapMethod_eAgX 4
#define ToneMapMethod_eKhronosPBR 5
#define ToneMapMethod_eShaped 6
#define ToneMapMethod_eCount 7
#else
#define TONEMAP_METHOD(name) ToneMapMethod::name

enum ToneMapMethod
{
  eFilmic = 0,
  eUncharted2,
  eClip,
  eACES,
  eAgX,
  eKhronosPBR,
  eShaped,  // tonemapShaped: a filmic curve with a toe and a shoulder to set
  eCount
};


// Bindings only relevant to `tonemapper.slang`
enum TonemapBinding
{
  eImageInput = 0,
  eImageOutput,
  eHistogramInputOutput,
  eLuminanceInputOutput,
};
#endif


// Tonemapper settings.
struct TonemapperData
{
  int isActive SLANG_DEFAULT(1);
  int method   SLANG_DEFAULT(TONEMAP_METHOD(eFilmic));

  // Input color correction
  float exposure SLANG_DEFAULT(1.0f);
  float temperature SLANG_DEFAULT(6506.11144f);  // Color temperature in degrees Kelvin. The default produces sRGB white for our approximation
  float tint SLANG_DEFAULT(3.25895312e-3f);  // Tint adjustment as an ANSI C78.377-2008 Duv value. The default produces sRGB white for our approximation
  float3x3 inputMatrix SLANG_DEFAULT(float3x3(0));  // Will be computed from the above

  // Post effects
  float contrast   SLANG_DEFAULT(1.0f);
  float brightness SLANG_DEFAULT(1.0f);
  float saturation SLANG_DEFAULT(1.0f);
  float vignette   SLANG_DEFAULT(0.0f);

  // Advanced color grading
  float vibrance      SLANG_DEFAULT(0.0f);  // Boosts desaturated colors while preserving saturated ones
  float shadowBias    SLANG_DEFAULT(0.0f);  // -1 to 1: Adjust shadow tones
  float midtoneBias   SLANG_DEFAULT(0.0f);  // -1 to 1: Adjust midtone brightness
  float highlightBias SLANG_DEFAULT(0.0f);  // -1 to 1: Adjust highlight tones
  float3 coolColor    SLANG_DEFAULT(float3(1.0f, 1.0f, 1.0f));  // Tint for shadows in split toning
  float3 warmColor    SLANG_DEFAULT(float3(1.0f, 1.0f, 1.0f));  // Tint for highlights in split toning
  float splitBalance  SLANG_DEFAULT(0.0f);  // -0.5 to 0.5: Balance between cool and warm tones

  // Looks (applyTonemap, after Saturation): black and white through a colour filter, lift, gamma and
  // gain a channel, posterising and film grain. The defaults leave the image as it is.
  float3   monoMix      SLANG_DEFAULT(float3(0.2126F, 0.7152F, 0.0722F));  // the black-and-white image's weights of linear R, G, B
  float    mono         SLANG_DEFAULT(0.0F);                               // 0 colour to 1 black and white
  float3   lift         SLANG_DEFAULT(float3(0.0F));                       // added to the shadows, fading to nothing at white
  float3   gamma        SLANG_DEFAULT(float3(1.0F));                       // above 1 lightens the midtones
  float3   gain         SLANG_DEFAULT(float3(1.0F));                       // scales everything, white most
  float    posterize    SLANG_DEFAULT(0.0F);                               // levels a channel; under 2 none
  float    posterizeMix SLANG_DEFAULT(1.0F);                               // how much of the posterised image
  float    grain        SLANG_DEFAULT(0.0F);                               // film grain's strength, 0 none
  uint32_t grainSeed    SLANG_DEFAULT(0);                                  // which grain: the frame's, so it moves as film's does

  // Auto-Exposure
  int      autoExposure         SLANG_DEFAULT(0);      // 0: off, 1: on
  float    autoExposureSpeed    SLANG_DEFAULT(5.0F);   // Speed of auto-exposure adjustment
  float    evMinValue           SLANG_DEFAULT(-5.0F);  // Minimum EV value for auto-exposure
  float    evMaxValue           SLANG_DEFAULT(10.0F);  // Maximum EV value for auto-exposure
  uint32_t enableCenterMetering SLANG_DEFAULT(0);      // 0 = off, 1 = on
  float    centerMeteringSize   SLANG_DEFAULT(0.5F);   // Size of the center metering area (0.0 to 1.0, where 1.0 is the full screen)
  uint32_t averageMode          SLANG_DEFAULT(1);      // 0 = Mean, 1 = Median
  // Dither
  int dither SLANG_DEFAULT(1);  // 0: no dither, 1: dither
  // 1: the false-colour exposure view (falseColorExposure) in place of the image
  int falseColor SLANG_DEFAULT(0);
  // The Shaped curve's handles (tonemapShaped), 0 to 1 each: how deep the shadows go, and how early
  // and gently the highlights roll off into white
  float shapedToe      SLANG_DEFAULT(0.5F);
  float shapedShoulder SLANG_DEFAULT(0.5F);
  // A small sensor's picture (applyTonemap), 0 to 1 each and 0 none: `sensorClip`, how much of the
  // curve gives way to a response that is straight up to white and clips there a channel at a time
  // (sensorClipCurve); `sensorNoise`, noise that is strongest in the shadows and coloured (sensorNoise),
  // moving with grainSeed
  float sensorClip  SLANG_DEFAULT(0.0F);
  float sensorNoise SLANG_DEFAULT(0.0F);
  // Khronos PBR Neutral's and Shaped's highlights to white, 0 to 1 (highlightDesaturation): how fast a
  // colour the curve compresses goes white. 0 keeps its colour to the top, 0.5 Khronos's own rate, 1
  // burns a lamp or a neon tube out white as a sensor does. The other curves have their own and ignore it.
  float highlightsToWhite SLANG_DEFAULT(0.5F);
};


#define EXPOSURE_HISTOGRAM_SIZE 256

NAMESPACE_SHADERIO_END()


#endif  // TONEMAMP_SHADERIO_H
