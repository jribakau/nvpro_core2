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

#ifndef DH_HDR_H
#define DH_HDR_H 1

#include "slang_types.h"

NAMESPACE_SHADERIO_BEGIN()

#define HDR_WORKGROUP_SIZE 32


// Common structures used for HDR environment maps.

// Environment acceleration structure - computed in hdr_env
struct EnvAccel
{
  uint  alias;
  float q;
};

// The environment's distribution for light samples a BSDF sample is weighed against (MIS
// compensation, Karlik et al. 2019, Unreal's PathTracingSkylightMISCompensation): each cell in
// proportion to how far its radiance rises above the environment's mean, so light samples go to
// what BSDF sampling finds poorly and leave the even rest to it. One per sampling-grid cell: the
// alias method's entry, as EnvAccel, and the solid-angle pdf of a direction in this cell.
//
// The same distribution as cumulative tables, for a sampler whose points are stratified (Sobol): the
// alias method sends neighbouring points to unrelated cells, inverting a CDF keeps them neighbours.
// `cdf` is the row's conditional CDF at the cell's right edge; the grid's `width` x `height` cells
// are followed by `height` entries whose `cdf` is the row marginal at each row's end (the rest of
// those entries unused). environmentSampleCompensatedStratified inverts them.
struct EnvCompensatedCell
{
  uint  alias;
  float q;
  float pdf;
  float cdf;
};

struct HdrPushBlock
{
  float4x4 mvp;
  float2   size;
  float    roughness;
  uint     numSamples;
};


struct HdrDomePushConstant
{
  float4x4 mvp;
  float4   multColor;
  float    blur;
};

enum EnvBindings
{
  eHdr                   = 0,
  eImpSamples            = 1,
  eImpSamplesCompensated = 2,  // EnvCompensatedCell per sampling-grid cell
};

enum EnvDomeBindings
{
  eHdrBrdf     = 0,
  eHdrDiffuse  = 1,
  eHdrSpecular = 2
};

enum EnvDomeDraw
{
  eHdrImage = 0
};

// The GPU-image path samples from a *grid* that may be coarser than the image: each grid cell
// covers a block of image texels, and the alias table has one entry per cell. A smooth source (a
// sky with its sun sampled separately) loses nothing to a coarse grid, and the alias build costs
// what the grid costs rather than what the image costs. The grid must divide the image exactly, so
// every texel belongs to one cell; a grid equal to the image is the per-texel case. The producer
// supplies the per-cell importance -- see nvvk::HdrIbl::updateFromGpuImage.

// env_write_pdf.slang: writes into the lat-long image's alpha the PDF of the grid cell each texel
// falls in. With a grid equal to the image this is the per-texel formula of `createEnvironmentAccel`
// (nvvk/hdr_ibl.cpp), so the file-load and GPU-image paths agree for the sampling shader.
enum EnvWritePdfBindings
{
  eEnvWritePdfImage = 0,
  eEnvWritePdfCells = 1,  // StructuredBuffer<float>, the per-cell importance, read
};

struct EnvWritePdfPushConstant
{
  uint2 imageSize;    // width, height of the lat-long image
  uint2 gridSize;     // width, height of the sampling grid; divides imageSize
  float invIntegral;  // 1 / integral of the emitted radiance (energy normalization)
};

NAMESPACE_SHADERIO_END()

#endif  // DH_HDR_H
