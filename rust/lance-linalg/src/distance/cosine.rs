// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Cosine distance
//!
//! <https://en.wikipedia.org/wiki/Cosine_similarity>
//!
//! `bf16, f16, f32, f64` types are supported.

use std::sync::Arc;

use arrow_array::{
    Array, FixedSizeListArray, Float32Array,
    cast::AsArray,
    types::{Float16Type, Float32Type, Float64Type, Int8Type},
};
use arrow_schema::DataType;
use half::{bf16, f16};
use lance_arrow::{ArrowFloatType, FixedSizeListArrayExt, FloatArray};
#[allow(unused_imports)]
use lance_core::utils::cpu::{SIMD_SUPPORT, SimdSupport};

use super::{Dot, norm_l2::norm_l2};
use super::{Normalize, dot::dot};
#[allow(unused_imports)]
use crate::simd::{
    FloatSimd, SIMD,
    f32::{f32x8, f32x16},
};
use crate::{Error, Result};

/// Cosine Distance
pub trait Cosine: Dot + Normalize {
    /// Cosine distance between two vectors.
    #[inline]
    fn cosine(x: &[Self], other: &[Self]) -> f32 {
        let x_norm = norm_l2(x);
        Self::cosine_fast(x, x_norm, other)
    }

    /// Fast cosine function, that assumes that the norm of the first vector is already known.
    #[inline]
    fn cosine_fast(x: &[Self], x_norm: f32, y: &[Self]) -> f32 {
        cosine_scalar(x, x_norm, y)
    }

    /// Cosine between two vectors, with the L2 norms of both vectors already known.
    #[inline]
    fn cosine_with_norms(x: &[Self], x_norm: f32, y_norm: f32, y: &[Self]) -> f32 {
        cosine_scalar_fast(x, x_norm, y, y_norm)
    }

    fn cosine_batch<'a>(
        x: &'a [Self],
        batch: &'a [Self],
        dimension: usize,
    ) -> Box<dyn Iterator<Item = f32> + 'a> {
        let x_norm = norm_l2(x);

        Box::new(
            batch
                .chunks_exact(dimension)
                .map(move |y| Self::cosine_fast(x, x_norm, y)),
        )
    }
}

impl Cosine for u8 {
    #[inline]
    fn cosine(x: &[Self], other: &[Self]) -> f32 {
        super::cosine_u8::cosine_u8(x, other)
    }
}

#[cfg(feature = "fp16kernels")]
mod bf16_kernel {
    use half::bf16;

    // These are the `cosine_bf16` function in bf16.c. Our build.rs script compiles
    // a version of this file for each SIMD level with different suffixes.
    unsafe extern "C" {
        #[cfg(target_arch = "aarch64")]
        pub fn cosine_bf16_neon(x: *const bf16, x_norm: f32, y: *const bf16, dimension: u32)
        -> f32;
        #[cfg(all(kernel_support = "avx512", target_arch = "x86_64"))]
        pub fn cosine_bf16_avx512(
            x: *const bf16,
            x_norm: f32,
            y: *const bf16,
            dimension: u32,
        ) -> f32;
        #[cfg(target_arch = "x86_64")]
        pub fn cosine_bf16_avx2(x: *const bf16, x_norm: f32, y: *const bf16, dimension: u32)
        -> f32;
        #[cfg(target_arch = "loongarch64")]
        pub fn cosine_bf16_lsx(x: *const bf16, x_norm: f32, y: *const bf16, dimension: u32) -> f32;
        #[cfg(target_arch = "loongarch64")]
        pub fn cosine_bf16_lasx(x: *const bf16, x_norm: f32, y: *const bf16, dimension: u32)
        -> f32;
    }
}

impl Cosine for bf16 {
    fn cosine_fast(x: &[Self], x_norm: f32, y: &[Self]) -> f32 {
        match *SIMD_SUPPORT {
            #[cfg(all(feature = "fp16kernels", target_arch = "aarch64"))]
            SimdSupport::Neon => unsafe {
                bf16_kernel::cosine_bf16_neon(x.as_ptr(), x_norm, y.as_ptr(), y.len() as u32)
            },
            #[cfg(all(
                feature = "fp16kernels",
                kernel_support = "avx512",
                target_arch = "x86_64"
            ))]
            SimdSupport::Avx512FP16 => unsafe {
                bf16_kernel::cosine_bf16_avx512(x.as_ptr(), x_norm, y.as_ptr(), y.len() as u32)
            },
            #[cfg(all(feature = "fp16kernels", target_arch = "x86_64"))]
            SimdSupport::Avx2 | SimdSupport::Avx512 => unsafe {
                bf16_kernel::cosine_bf16_avx2(x.as_ptr(), x_norm, y.as_ptr(), y.len() as u32)
            },
            #[cfg(all(feature = "fp16kernels", target_arch = "loongarch64"))]
            SimdSupport::Lasx => unsafe {
                bf16_kernel::cosine_bf16_lasx(x.as_ptr(), x_norm, y.as_ptr(), y.len() as u32)
            },
            #[cfg(all(feature = "fp16kernels", target_arch = "loongarch64"))]
            SimdSupport::Lsx => unsafe {
                bf16_kernel::cosine_bf16_lsx(x.as_ptr(), x_norm, y.as_ptr(), y.len() as u32)
            },
            _ => cosine_scalar(x, x_norm, y),
        }
    }
}

#[cfg(feature = "fp16kernels")]
mod kernel {
    use super::*;

    // These are the `cosine_f16` function in f16.c. Our build.rs script compiles
    // a version of this file for each SIMD level with different suffixes.
    unsafe extern "C" {
        #[cfg(target_arch = "aarch64")]
        pub fn cosine_f16_neon(x: *const f16, x_norm: f32, y: *const f16, dimension: u32) -> f32;
        #[cfg(all(kernel_support = "avx512", target_arch = "x86_64"))]
        pub fn cosine_f16_avx512(x: *const f16, x_norm: f32, y: *const f16, dimension: u32) -> f32;
        #[cfg(target_arch = "x86_64")]
        pub fn cosine_f16_avx2(x: *const f16, x_norm: f32, y: *const f16, dimension: u32) -> f32;
        #[cfg(target_arch = "loongarch64")]
        pub fn cosine_f16_lsx(x: *const f16, x_norm: f32, y: *const f16, dimension: u32) -> f32;
        #[cfg(target_arch = "loongarch64")]
        pub fn cosine_f16_lasx(x: *const f16, x_norm: f32, y: *const f16, dimension: u32) -> f32;
    }
}

impl Cosine for f16 {
    fn cosine_fast(x: &[Self], x_norm: f32, y: &[Self]) -> f32 {
        match *SIMD_SUPPORT {
            #[cfg(all(feature = "fp16kernels", target_arch = "aarch64"))]
            SimdSupport::Neon => unsafe {
                kernel::cosine_f16_neon(x.as_ptr(), x_norm, y.as_ptr(), y.len() as u32)
            },
            #[cfg(all(
                feature = "fp16kernels",
                kernel_support = "avx512",
                target_arch = "x86_64"
            ))]
            SimdSupport::Avx512FP16 => unsafe {
                kernel::cosine_f16_avx512(x.as_ptr(), x_norm, y.as_ptr(), y.len() as u32)
            },
            #[cfg(all(feature = "fp16kernels", target_arch = "x86_64"))]
            SimdSupport::Avx2 => unsafe {
                kernel::cosine_f16_avx2(x.as_ptr(), x_norm, y.as_ptr(), y.len() as u32)
            },
            #[cfg(all(feature = "fp16kernels", target_arch = "loongarch64"))]
            SimdSupport::Lasx => unsafe {
                kernel::cosine_f16_lasx(x.as_ptr(), x_norm, y.as_ptr(), y.len() as u32)
            },
            #[cfg(all(feature = "fp16kernels", target_arch = "loongarch64"))]
            SimdSupport::Lsx => unsafe {
                kernel::cosine_f16_lsx(x.as_ptr(), x_norm, y.as_ptr(), y.len() as u32)
            },
            _ => cosine_scalar(x, x_norm, y),
        }
    }
}

/// f32 single-vector cosine helpers used by `cosine_batch` for fixed
/// dimensions 8 and 16.
///
/// These were previously a single generic `cosine_once<S, N>` but the
/// monomorphizations have to dispatch on `SIMD_SUPPORT` for the SIMD path
/// to stay correct under any compile baseline. Splitting them into two
/// concrete entry points keeps the dispatch site flat and lets each width
/// route to a `#[target_feature]` AVX2 inner function.
mod f32 {
    use super::*;

    #[inline]
    pub(super) fn cosine_once_8(x: &[f32], x_norm: f32, y: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            match *SIMD_SUPPORT {
                SimdSupport::Avx2 | SimdSupport::Avx512 | SimdSupport::Avx512FP16 => unsafe {
                    cosine_once_x86::cosine_once_8_avx2(x, x_norm, y)
                },
                _ => cosine_once_8_scalar(x, x_norm, y),
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            cosine_once_8_other(x, x_norm, y)
        }
    }

    #[inline]
    pub(super) fn cosine_once_16(x: &[f32], x_norm: f32, y: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            match *SIMD_SUPPORT {
                SimdSupport::Avx2 | SimdSupport::Avx512 | SimdSupport::Avx512FP16 => unsafe {
                    cosine_once_x86::cosine_once_16_avx2(x, x_norm, y)
                },
                _ => cosine_once_16_scalar(x, x_norm, y),
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            cosine_once_16_other(x, x_norm, y)
        }
    }

    /// Portable scalar `cosine_once` for length-8 vectors. Matches the SIMD
    /// path modulo summation order.
    #[cfg(target_arch = "x86_64")]
    #[inline]
    pub(super) fn cosine_once_8_scalar(x: &[f32], x_norm: f32, y: &[f32]) -> f32 {
        let mut xy = 0.0f32;
        let mut y2 = 0.0f32;
        for i in 0..8 {
            xy += x[i] * y[i];
            y2 += y[i] * y[i];
        }
        1.0 - xy / x_norm / y2.sqrt()
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    pub(super) fn cosine_once_16_scalar(x: &[f32], x_norm: f32, y: &[f32]) -> f32 {
        let mut xy = 0.0f32;
        let mut y2 = 0.0f32;
        for i in 0..16 {
            xy += x[i] * y[i];
            y2 += y[i] * y[i];
        }
        1.0 - xy / x_norm / y2.sqrt()
    }

    #[cfg(target_arch = "x86_64")]
    mod cosine_once_x86 {
        use super::{f32x8, f32x16};
        use crate::simd::SIMD;

        #[target_feature(enable = "avx,avx2,fma")]
        pub unsafe fn cosine_once_8_avx2(x: &[f32], x_norm: f32, y: &[f32]) -> f32 {
            let xv = f32x8::load_unaligned(x.as_ptr());
            let yv = f32x8::load_unaligned(y.as_ptr());
            let y2 = yv * yv;
            let xy = xv * yv;
            1.0 - xy.reduce_sum() / x_norm / y2.reduce_sum().sqrt()
        }

        #[target_feature(enable = "avx,avx2,fma")]
        pub unsafe fn cosine_once_16_avx2(x: &[f32], x_norm: f32, y: &[f32]) -> f32 {
            let xv = f32x16::load_unaligned(x.as_ptr());
            let yv = f32x16::load_unaligned(y.as_ptr());
            let y2 = yv * yv;
            let xy = xv * yv;
            1.0 - xy.reduce_sum() / x_norm / y2.reduce_sum().sqrt()
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    #[inline]
    fn cosine_once_8_other(x: &[f32], x_norm: f32, y: &[f32]) -> f32 {
        let xv = unsafe { f32x8::load_unaligned(x.as_ptr()) };
        let yv = unsafe { f32x8::load_unaligned(y.as_ptr()) };
        let y2 = yv * yv;
        let xy = xv * yv;
        1.0 - xy.reduce_sum() / x_norm / y2.reduce_sum().sqrt()
    }

    #[cfg(not(target_arch = "x86_64"))]
    #[inline]
    fn cosine_once_16_other(x: &[f32], x_norm: f32, y: &[f32]) -> f32 {
        let xv = unsafe { f32x16::load_unaligned(x.as_ptr()) };
        let yv = unsafe { f32x16::load_unaligned(y.as_ptr()) };
        let y2 = yv * yv;
        let xy = xv * yv;
        1.0 - xy.reduce_sum() / x_norm / y2.reduce_sum().sqrt()
    }
}

impl Cosine for f32 {
    #[inline]
    fn cosine_fast(x: &[Self], x_norm: Self, other: &[Self]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            match *SIMD_SUPPORT {
                SimdSupport::Avx2 | SimdSupport::Avx512 | SimdSupport::Avx512FP16 => unsafe {
                    f32_x86::cosine_fast_avx2(x, x_norm, other)
                },
                _ => cosine_scalar(x, x_norm, other),
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            cosine_fast_f32_simd_other(x, x_norm, other)
        }
    }

    #[inline]
    fn cosine_with_norms(x: &[Self], x_norm: Self, y_norm: Self, y: &[Self]) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            match *SIMD_SUPPORT {
                SimdSupport::Avx2 | SimdSupport::Avx512 | SimdSupport::Avx512FP16 => unsafe {
                    f32_x86::cosine_with_norms_avx2(x, x_norm, y_norm, y)
                },
                _ => cosine_scalar_fast(x, x_norm, y, y_norm),
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            cosine_with_norms_f32_simd_other(x, x_norm, y_norm, y)
        }
    }

    fn cosine_batch<'a>(
        x: &'a [Self],
        batch: &'a [Self],
        dimension: usize,
    ) -> Box<dyn Iterator<Item = f32> + 'a> {
        let x_norm = norm_l2(x);

        match dimension {
            8 => Box::new(
                batch
                    .chunks_exact(dimension)
                    .map(move |y| f32::cosine_once_8(x, x_norm, y)),
            ),
            16 => Box::new(
                batch
                    .chunks_exact(dimension)
                    .map(move |y| f32::cosine_once_16(x, x_norm, y)),
            ),
            _ => Box::new(
                batch
                    .chunks_exact(dimension)
                    .map(move |y| Self::cosine_fast(x, x_norm, y)),
            ),
        }
    }
}

/// AVX2 + FMA implementations of the f32 cosine kernels.
///
/// Both functions carry `#[target_feature(enable = "avx,avx2,fma")]` so the
/// SIMD primitives in `crate::simd::f32` (which use raw AVX intrinsics) inline
/// correctly even when the compile baseline does not have AVX2 enabled.
#[cfg(target_arch = "x86_64")]
mod f32_x86 {
    use super::{dot, f32x8, f32x16, norm_l2};
    use crate::simd::{FloatSimd, SIMD};

    #[target_feature(enable = "avx,avx2,fma")]
    pub unsafe fn cosine_fast_avx2(x: &[f32], x_norm: f32, other: &[f32]) -> f32 {
        let dim = x.len();
        let unrolled_len = dim / 16 * 16;
        let mut y_norm16 = f32x16::zeros();
        let mut xy16 = f32x16::zeros();
        for i in (0..unrolled_len).step_by(16) {
            let xv = f32x16::load_unaligned(x.as_ptr().add(i));
            let yv = f32x16::load_unaligned(other.as_ptr().add(i));
            xy16.multiply_add(xv, yv);
            y_norm16.multiply_add(yv, yv);
        }
        let aligned_len = dim / 8 * 8;
        let mut y_norm8 = f32x8::zeros();
        let mut xy8 = f32x8::zeros();
        for i in (unrolled_len..aligned_len).step_by(8) {
            let xv = f32x8::load_unaligned(x.as_ptr().add(i));
            let yv = f32x8::load_unaligned(other.as_ptr().add(i));
            xy8.multiply_add(xv, yv);
            y_norm8.multiply_add(yv, yv);
        }
        let y_norm =
            y_norm16.reduce_sum() + y_norm8.reduce_sum() + norm_l2(&other[aligned_len..]).powi(2);
        let xy =
            xy16.reduce_sum() + xy8.reduce_sum() + dot(&x[aligned_len..], &other[aligned_len..]);
        1.0 - xy / x_norm / y_norm.sqrt()
    }

    #[target_feature(enable = "avx,avx2,fma")]
    pub unsafe fn cosine_with_norms_avx2(x: &[f32], x_norm: f32, y_norm: f32, y: &[f32]) -> f32 {
        let dim = x.len();
        let unrolled_len = dim / 16 * 16;
        let mut xy16 = f32x16::zeros();
        for i in (0..unrolled_len).step_by(16) {
            let xv = f32x16::load_unaligned(x.as_ptr().add(i));
            let yv = f32x16::load_unaligned(y.as_ptr().add(i));
            xy16.multiply_add(xv, yv);
        }
        let aligned_len = dim / 8 * 8;
        let mut xy8 = f32x8::zeros();
        for i in (unrolled_len..aligned_len).step_by(8) {
            let xv = f32x8::load_unaligned(x.as_ptr().add(i));
            let yv = f32x8::load_unaligned(y.as_ptr().add(i));
            xy8.multiply_add(xv, yv);
        }
        let xy = xy16.reduce_sum() + xy8.reduce_sum() + dot(&x[aligned_len..], &y[aligned_len..]);
        1.0 - xy / x_norm / y_norm
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
fn cosine_fast_f32_simd_other(x: &[f32], x_norm: f32, other: &[f32]) -> f32 {
    let dim = x.len();
    let unrolled_len = dim / 16 * 16;
    let mut y_norm16 = f32x16::zeros();
    let mut xy16 = f32x16::zeros();
    for i in (0..unrolled_len).step_by(16) {
        unsafe {
            let xv = f32x16::load_unaligned(x.as_ptr().add(i));
            let yv = f32x16::load_unaligned(other.as_ptr().add(i));
            xy16.multiply_add(xv, yv);
            y_norm16.multiply_add(yv, yv);
        }
    }
    let aligned_len = dim / 8 * 8;
    let mut y_norm8 = f32x8::zeros();
    let mut xy8 = f32x8::zeros();
    for i in (unrolled_len..aligned_len).step_by(8) {
        unsafe {
            let xv = f32x8::load_unaligned(x.as_ptr().add(i));
            let yv = f32x8::load_unaligned(other.as_ptr().add(i));
            xy8.multiply_add(xv, yv);
            y_norm8.multiply_add(yv, yv);
        }
    }
    let y_norm =
        y_norm16.reduce_sum() + y_norm8.reduce_sum() + norm_l2(&other[aligned_len..]).powi(2);
    let xy = xy16.reduce_sum() + xy8.reduce_sum() + dot(&x[aligned_len..], &other[aligned_len..]);
    1.0 - xy / x_norm / y_norm.sqrt()
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
fn cosine_with_norms_f32_simd_other(x: &[f32], x_norm: f32, y_norm: f32, y: &[f32]) -> f32 {
    let dim = x.len();
    let unrolled_len = dim / 16 * 16;
    let mut xy16 = f32x16::zeros();
    for i in (0..unrolled_len).step_by(16) {
        unsafe {
            let xv = f32x16::load_unaligned(x.as_ptr().add(i));
            let yv = f32x16::load_unaligned(y.as_ptr().add(i));
            xy16.multiply_add(xv, yv);
        }
    }
    let aligned_len = dim / 8 * 8;
    let mut xy8 = f32x8::zeros();
    for i in (unrolled_len..aligned_len).step_by(8) {
        unsafe {
            let xv = f32x8::load_unaligned(x.as_ptr().add(i));
            let yv = f32x8::load_unaligned(y.as_ptr().add(i));
            xy8.multiply_add(xv, yv);
        }
    }
    let xy = xy16.reduce_sum() + xy8.reduce_sum() + dot(&x[aligned_len..], &y[aligned_len..]);
    1.0 - xy / x_norm / y_norm
}

impl Cosine for f64 {
    #[inline]
    fn cosine_fast(x: &[Self], x_norm: f32, y: &[Self]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            match *SIMD_SUPPORT {
                SimdSupport::Avx2 | SimdSupport::Avx512 | SimdSupport::Avx512FP16 => unsafe {
                    f64_x86::cosine_fast_avx2(x, x_norm, y)
                },
                _ => cosine_scalar(x, x_norm, y),
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            cosine_fast_f64_simd_other(x, x_norm, y)
        }
    }
}

/// AVX2 + FMA implementation of the f64 cosine_fast kernel.
///
/// Carries `#[target_feature(enable = "avx,avx2,fma")]` so the SIMD primitives
/// in `crate::simd::f64` (which use raw AVX intrinsics) inline correctly even
/// when the compile baseline does not have AVX2 enabled.
#[cfg(target_arch = "x86_64")]
mod f64_x86 {
    use crate::simd::f64::{f64x4, f64x8};
    use crate::simd::{FloatSimd, SIMD};

    #[target_feature(enable = "avx,avx2,fma")]
    pub unsafe fn cosine_fast_avx2(x: &[f64], x_norm: f32, y: &[f64]) -> f32 {
        let dim = x.len();
        let unrolled_len = dim / 8 * 8;
        let mut y_norm8 = f64x8::zeros();
        let mut xy8 = f64x8::zeros();
        for i in (0..unrolled_len).step_by(8) {
            let xv = f64x8::load_unaligned(x.as_ptr().add(i));
            let yv = f64x8::load_unaligned(y.as_ptr().add(i));
            xy8.multiply_add(xv, yv);
            y_norm8.multiply_add(yv, yv);
        }
        let aligned_len = dim / 4 * 4;
        let mut y_norm4 = f64x4::zeros();
        let mut xy4 = f64x4::zeros();
        for i in (unrolled_len..aligned_len).step_by(4) {
            let xv = f64x4::load_unaligned(x.as_ptr().add(i));
            let yv = f64x4::load_unaligned(y.as_ptr().add(i));
            xy4.multiply_add(xv, yv);
            y_norm4.multiply_add(yv, yv);
        }
        let tail_y_norm: f64 = y[aligned_len..].iter().map(|&v| v * v).sum();
        let tail_xy: f64 = x[aligned_len..]
            .iter()
            .zip(y[aligned_len..].iter())
            .map(|(&a, &b)| a * b)
            .sum();

        let y_norm_sq = (y_norm8.reduce_sum() + y_norm4.reduce_sum() + tail_y_norm) as f32;
        let xy = (xy8.reduce_sum() + xy4.reduce_sum() + tail_xy) as f32;
        1.0 - xy / x_norm / y_norm_sq.sqrt()
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
fn cosine_fast_f64_simd_other(x: &[f64], x_norm: f32, y: &[f64]) -> f32 {
    use crate::simd::f64::{f64x4, f64x8};
    use crate::simd::{FloatSimd, SIMD};

    let dim = x.len();
    let unrolled_len = dim / 8 * 8;
    let mut y_norm8 = f64x8::zeros();
    let mut xy8 = f64x8::zeros();
    for i in (0..unrolled_len).step_by(8) {
        unsafe {
            let xv = f64x8::load_unaligned(x.as_ptr().add(i));
            let yv = f64x8::load_unaligned(y.as_ptr().add(i));
            xy8.multiply_add(xv, yv);
            y_norm8.multiply_add(yv, yv);
        }
    }
    let aligned_len = dim / 4 * 4;
    let mut y_norm4 = f64x4::zeros();
    let mut xy4 = f64x4::zeros();
    for i in (unrolled_len..aligned_len).step_by(4) {
        unsafe {
            let xv = f64x4::load_unaligned(x.as_ptr().add(i));
            let yv = f64x4::load_unaligned(y.as_ptr().add(i));
            xy4.multiply_add(xv, yv);
            y_norm4.multiply_add(yv, yv);
        }
    }
    let tail_y_norm: f64 = y[aligned_len..].iter().map(|&v| v * v).sum();
    let tail_xy: f64 = x[aligned_len..]
        .iter()
        .zip(y[aligned_len..].iter())
        .map(|(&a, &b)| a * b)
        .sum();

    let y_norm_sq = (y_norm8.reduce_sum() + y_norm4.reduce_sum() + tail_y_norm) as f32;
    let xy = (xy8.reduce_sum() + xy4.reduce_sum() + tail_xy) as f32;
    1.0 - xy / x_norm / y_norm_sq.sqrt()
}

/// Fallback non-SIMD implementation
#[inline]
fn cosine_scalar<T: Dot>(x: &[T], x_norm: f32, y: &[T]) -> f32 {
    let y_sq = dot(y, y);
    let xy = dot(x, y);
    // 1 - xy / (sqrt(x_sq) * sqrt(y_sq))
    1.0 - xy / (x_norm * y_sq.sqrt())
}

#[inline]
pub(crate) fn cosine_scalar_fast<T: Dot>(x: &[T], x_norm: f32, y: &[T], y_norm: f32) -> f32 {
    let xy = dot(x, y);
    // 1 - xy / (sqrt(x_sq) * sqrt(y_sq))
    // use f64 for overflow protection.
    1.0 - (xy / (x_norm * y_norm))
}

/// Cosine distance function between two vectors.
pub fn cosine_distance<T: Cosine>(from: &[T], to: &[T]) -> f32 {
    T::cosine(from, to)
}

/// Cosine Distance
///
/// <https://en.wikipedia.org/wiki/Cosine_similarity>
///
/// Parameters
/// -----------
///
/// - *from*: the vector to compute distance from.
/// - *to*: the batch of vectors to compute distance to.
/// - *dimension*: the dimension of the vector.
///
/// Returns
/// -------
/// An iterator of pair-wise cosine distance between from vector to each vector in the batch.
///
pub fn cosine_distance_batch<'a, T: Cosine>(
    from: &'a [T],
    batch: &'a [T],
    dimension: usize,
) -> Box<dyn Iterator<Item = f32> + 'a> {
    T::cosine_batch(from, batch, dimension)
}

fn do_cosine_distance_arrow_batch<T: ArrowFloatType>(
    from: &T::ArrayType,
    to: &FixedSizeListArray,
) -> Result<Arc<Float32Array>>
where
    T::Native: Cosine,
{
    let dimension = to.value_length() as usize;
    debug_assert_eq!(from.len(), dimension);

    // TODO: if we detect there is a run of nulls, should we skip those?
    let to_values =
        to.values()
            .as_any()
            .downcast_ref::<T::ArrayType>()
            .ok_or(Error::InvalidArgumentError(format!(
                "Unsupported data type {:?}",
                to.values().data_type()
            )))?;
    let dists = cosine_distance_batch(from.as_slice(), to_values.as_slice(), dimension);

    Ok(Arc::new(Float32Array::new(
        dists.collect(),
        to.nulls().cloned(),
    )))
}

/// Compute Cosine distance between a vector and a batch of vectors.
///
/// Null buffer of `to` is propagated to the returned array.
///
/// Parameters
///
/// - `from`: the vector to compute distance from.
/// - `to`: a list of vectors to compute distance to.
///
/// # Panics
///
/// Panics if the length of `from` is not equal to the dimension (value length) of `to`.
pub fn cosine_distance_arrow_batch(
    from: &dyn Array,
    to: &FixedSizeListArray,
) -> Result<Arc<Float32Array>> {
    match *from.data_type() {
        DataType::Float16 => do_cosine_distance_arrow_batch::<Float16Type>(from.as_primitive(), to),
        DataType::Float32 => do_cosine_distance_arrow_batch::<Float32Type>(from.as_primitive(), to),
        DataType::Float64 => do_cosine_distance_arrow_batch::<Float64Type>(from.as_primitive(), to),
        DataType::Int8 => do_cosine_distance_arrow_batch::<Float32Type>(
            &from
                .as_primitive::<Int8Type>()
                .into_iter()
                .map(|x| x.unwrap() as f32)
                .collect(),
            &to.convert_to_floating_point()?,
        ),
        _ => Err(Error::InvalidArgumentError(format!(
            "Unsupported data type {:?}",
            from.data_type()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test_utils::{
        arbitrary_bf16, arbitrary_f16, arbitrary_f32, arbitrary_f64, arbitrary_vector_pair,
    };
    use approx::assert_relative_eq;
    use num_traits::AsPrimitive;
    use proptest::prelude::*;

    fn cosine_dist_brute_force(x: &[f32], y: &[f32]) -> f32 {
        let xy = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| xi * yi)
            .sum::<f32>();
        let x_sq = x.iter().map(|&xi| xi * xi).sum::<f32>().sqrt();
        let y_sq = y.iter().map(|&yi| yi * yi).sum::<f32>().sqrt();
        1.0 - xy / x_sq / y_sq
    }

    #[test]
    fn test_cosine() {
        let x: Float32Array = (1..9).map(|v| v as f32).collect();
        let y: Float32Array = (100..108).map(|v| v as f32).collect();
        let d = cosine_distance_batch(x.values(), y.values(), 8).collect::<Vec<_>>();
        // from scipy.spatial.distance.cosine
        assert_relative_eq!(d[0], 1.0 - 0.900_957);

        let x = Float32Array::from_iter_values([3.0, 45.0, 7.0, 2.0, 5.0, 20.0, 13.0, 12.0]);
        let y = Float32Array::from_iter_values([2.0, 54.0, 13.0, 15.0, 22.0, 34.0, 50.0, 1.0]);
        let d = cosine_distance_batch(x.values(), y.values(), 8).collect::<Vec<_>>();
        // from sklearn.metrics.pairwise import cosine_similarity
        assert_relative_eq!(d[0], 1.0 - 0.873_580_63);
    }

    #[test]
    fn test_cosine_large() {
        let total = 1024;
        let x = (0..total).map(|v| v as f32).collect::<Vec<_>>();
        let y = (1024..1024 + total).map(|v| v as f32).collect::<Vec<_>>();
        let d = cosine_distance_batch(&x, &y, total).collect::<Vec<_>>();
        assert_relative_eq!(d[0], cosine_dist_brute_force(&x, &y));
    }

    #[test]
    fn test_cosine_not_aligned() {
        let x: Float32Array = vec![16_f32, 32_f32].into();
        let y: Float32Array = vec![1_f32, 2_f32, 4_f32, 8_f32].into();
        let d = cosine_distance_batch(x.values(), y.values(), 2).collect::<Vec<_>>();
        assert_relative_eq!(d[0], 0.0);
        assert_relative_eq!(d[0], 0.0);
    }

    /// Reference implementation of cosine distance, plus error propagation.
    ///
    /// Pass `rel_err` to provide the allowed relative error in the dot product
    /// results. This function will then compute the expected absolute error.
    fn cosine_ref(x: &[f64], y: &[f64], rel_err: f64) -> (f32, f32) {
        let xy = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| xi * yi)
            .sum::<f64>();
        let x_sq = x.iter().map(|&xi| xi * xi).sum::<f64>().sqrt();
        let y_sq = y.iter().map(|&yi| yi * yi).sum::<f64>().sqrt();
        let expected = (1.0 - xy / x_sq / y_sq) as f32;

        let factor = 1.0 + rel_err;
        let low = (1.0 - (xy * factor) / (x_sq / factor) / (y_sq / factor)) as f32;
        let high = (1.0 - (xy / factor) / (x_sq * factor) / (y_sq * factor)) as f32;
        let low = (expected - low).abs();
        let high = (expected - high).abs();
        let error = low.max(high);

        (expected, error)
    }

    fn do_cosine_test<T: Cosine + AsPrimitive<f64>>(
        x: &[T],
        y: &[T],
    ) -> std::result::Result<(), TestCaseError> {
        let x_f64 = x.iter().map(|&v| v.as_()).collect::<Vec<_>>();
        let y_f64 = y.iter().map(|&v| v.as_()).collect::<Vec<_>>();

        let (expected, max_error) = cosine_ref(&x_f64, &y_f64, 1e-6);
        let result = T::cosine(x, y);

        prop_assert!(approx::relative_eq!(result, expected, epsilon = max_error));
        Ok(())
    }

    proptest::proptest! {
        #[test]
        fn test_cosine_f16((x, y) in arbitrary_vector_pair(arbitrary_f16, 4..4048)) {
            // Cosine requires non-zero vectors
            prop_assume!(norm_l2(&x) > 1e-6);
            prop_assume!(norm_l2(&y) > 1e-6);
            do_cosine_test(&x, &y)?;
        }

        #[test]
        fn test_cosine_bf16((x, y) in arbitrary_vector_pair(arbitrary_bf16, 4..4048)){
            prop_assume!(norm_l2(&x) > 1e-6);
            prop_assume!(norm_l2(&y) > 1e-6);
            do_cosine_test(&x, &y)?;
        }

        #[test]
        fn test_cosine_f32((x, y) in arbitrary_vector_pair(arbitrary_f32, 4..4048)){
            prop_assume!(norm_l2(&x) > 1e-10);
            prop_assume!(norm_l2(&y) > 1e-10);
            do_cosine_test(&x, &y)?;
        }

        #[test]
        fn test_cosine_f64((x, y) in arbitrary_vector_pair(arbitrary_f64, 4..4048)){
            prop_assume!(norm_l2(&x) > 1e-20);
            prop_assume!(norm_l2(&y) > 1e-20);
            do_cosine_test(&x, &y)?;
        }

        /// Cross-backend parity for the f32 cosine_fast kernel. The scalar
        /// fallback (`cosine_scalar`) routes through `dot::<f32>::dot`, while
        /// the SIMD path uses `f32x16` / `f32x8` primitives. They must agree
        /// within numerical tolerance so the runtime fallback is safe to take
        /// once the compile baseline is lowered.
        #[cfg(target_arch = "x86_64")]
        #[test]
        fn test_cosine_fast_f32_scalar_simd_parity(
            (x, y) in arbitrary_vector_pair(arbitrary_f32, 4..4048)
        ) {
            prop_assume!(norm_l2(&x) > 1e-10);
            prop_assume!(norm_l2(&y) > 1e-10);
            let x_norm = norm_l2(&x);
            let scalar = cosine_scalar(&x, x_norm, &y);
            let simd = <f32 as Cosine>::cosine_fast(&x, x_norm, &y);
            prop_assert!(approx::relative_eq!(scalar, simd, max_relative = 1e-5));
        }

        /// Cross-backend parity for the f32 cosine_with_norms kernel. The
        /// scalar fallback (`cosine_scalar_fast`) and the SIMD path must
        /// agree within numerical tolerance.
        #[cfg(target_arch = "x86_64")]
        #[test]
        fn test_cosine_with_norms_f32_scalar_simd_parity(
            (x, y) in arbitrary_vector_pair(arbitrary_f32, 4..4048)
        ) {
            prop_assume!(norm_l2(&x) > 1e-10);
            prop_assume!(norm_l2(&y) > 1e-10);
            let x_norm = norm_l2(&x);
            let y_norm = norm_l2(&y);
            let scalar = cosine_scalar_fast(&x, x_norm, &y, y_norm);
            let simd = <f32 as Cosine>::cosine_with_norms(&x, x_norm, y_norm, &y);
            prop_assert!(approx::relative_eq!(scalar, simd, max_relative = 1e-5));
        }

        /// Cross-backend parity for the f64 cosine_fast kernel. The scalar
        /// fallback (`cosine_scalar`) routes through `dot::<f64>::dot` (which
        /// itself dispatches), while the SIMD path uses `f64x4` / `f64x8`
        /// primitives. They must agree within numerical tolerance.
        #[cfg(target_arch = "x86_64")]
        #[test]
        fn test_cosine_fast_f64_scalar_simd_parity(
            (x, y) in arbitrary_vector_pair(arbitrary_f64, 4..4048)
        ) {
            prop_assume!(norm_l2(&x) > 1e-20);
            prop_assume!(norm_l2(&y) > 1e-20);
            let x_norm = norm_l2(&x);
            let scalar = cosine_scalar(&x, x_norm, &y);
            let simd = <f64 as Cosine>::cosine_fast(&x, x_norm, &y);
            prop_assert!(approx::relative_eq!(scalar, simd, max_relative = 1e-5));
        }

        /// Cross-backend parity for the despecialized cosine_once_8 kernel.
        /// Replaces the previous generic `cosine_once<f32x8, 8>` invocation.
        #[cfg(target_arch = "x86_64")]
        #[test]
        fn test_cosine_once_8_scalar_simd_parity(
            x in proptest::array::uniform8(arbitrary_f32()),
            y in proptest::array::uniform8(arbitrary_f32()),
        ) {
            let xs: Vec<f32> = x.to_vec();
            let ys: Vec<f32> = y.to_vec();
            prop_assume!(norm_l2(&xs) > 1e-10);
            prop_assume!(norm_l2(&ys) > 1e-10);
            let x_norm = norm_l2(&xs);
            let scalar = super::f32::cosine_once_8_scalar(&xs, x_norm, &ys);
            let simd = super::f32::cosine_once_8(&xs, x_norm, &ys);
            prop_assert!(approx::relative_eq!(scalar, simd, max_relative = 1e-5));
        }

        /// Cross-backend parity for the despecialized cosine_once_16 kernel.
        /// Replaces the previous generic `cosine_once<f32x16, 16>` invocation.
        #[cfg(target_arch = "x86_64")]
        #[test]
        fn test_cosine_once_16_scalar_simd_parity(
            x in proptest::array::uniform16(arbitrary_f32()),
            y in proptest::array::uniform16(arbitrary_f32()),
        ) {
            let xs: Vec<f32> = x.to_vec();
            let ys: Vec<f32> = y.to_vec();
            prop_assume!(norm_l2(&xs) > 1e-10);
            prop_assume!(norm_l2(&ys) > 1e-10);
            let x_norm = norm_l2(&xs);
            let scalar = super::f32::cosine_once_16_scalar(&xs, x_norm, &ys);
            let simd = super::f32::cosine_once_16(&xs, x_norm, &ys);
            prop_assert!(approx::relative_eq!(scalar, simd, max_relative = 1e-5));
        }
    }
}
