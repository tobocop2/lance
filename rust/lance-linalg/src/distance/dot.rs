// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Dot product.

use std::iter::Sum;
use std::ops::AddAssign;
use std::sync::Arc;

use crate::Error;
use arrow_array::types::{Float16Type, Float64Type, Int8Type};
use arrow_array::{Array, FixedSizeListArray, Float32Array, cast::AsArray, types::Float32Type};
use arrow_schema::DataType;
use half::{bf16, f16};
use lance_arrow::{ArrowFloatType, FixedSizeListArrayExt, FloatArray};
use lance_core::assume_eq;
#[allow(unused_imports)]
use lance_core::utils::cpu::{SIMD_SUPPORT, SimdSupport};
use num_traits::{AsPrimitive, Num, real::Real};

use crate::Result;

/// Default implementation of dot product.
///
// The following code has been tuned for auto-vectorization.
// Please make sure run `cargo bench --bench dot` with and without AVX-512 before any change.
// Tested `target-features`: avx512f,avx512vl,f16c
#[inline]
fn dot_scalar<
    T: AsPrimitive<Output>,
    Output: Real + Sum + AddAssign + 'static,
    const LANES: usize,
>(
    from: &[T],
    to: &[T],
) -> Output {
    let x_chunks = to.chunks_exact(LANES);
    let y_chunks = from.chunks_exact(LANES);
    let sum = if x_chunks.remainder().is_empty() {
        Output::zero()
    } else {
        x_chunks
            .remainder()
            .iter()
            .zip(y_chunks.remainder().iter())
            .map(|(&x, &y)| x.as_() * y.as_())
            .sum::<Output>()
    };
    // Use known size to allow LLVM to kick in auto-vectorization.
    let mut sums = [Output::zero(); LANES];
    for (x, y) in x_chunks.zip(y_chunks) {
        for i in 0..LANES {
            sums[i] += x[i].as_() * y[i].as_();
        }
    }
    sum + sums.iter().copied().sum::<Output>()
}

/// Dot product.
#[inline]
pub fn dot<T: Dot>(from: &[T], to: &[T]) -> f32 {
    T::dot(from, to)
}

/// Negative [Dot] distance.
#[inline]
pub fn dot_distance<T: Dot>(from: &[T], to: &[T]) -> f32 {
    1.0 - T::dot(from, to)
}

/// Dot product
pub trait Dot: Num {
    /// Dot product.
    fn dot(x: &[Self], y: &[Self]) -> f32;
}

#[cfg(feature = "fp16kernels")]
mod bf16_kernel {
    use half::bf16;

    // These are the `dot_bf16` function in bf16.c. Our build.rs script compiles
    // a version of this file for each SIMD level with different suffixes.
    unsafe extern "C" {
        #[cfg(target_arch = "aarch64")]
        pub fn dot_bf16_neon(ptr1: *const bf16, ptr2: *const bf16, len: u32) -> f32;
        #[cfg(all(kernel_support = "avx512", target_arch = "x86_64"))]
        pub fn dot_bf16_avx512(ptr1: *const bf16, ptr2: *const bf16, len: u32) -> f32;
        #[cfg(target_arch = "x86_64")]
        pub fn dot_bf16_avx2(ptr1: *const bf16, ptr2: *const bf16, len: u32) -> f32;
        #[cfg(target_arch = "loongarch64")]
        pub fn dot_bf16_lsx(ptr1: *const bf16, ptr2: *const bf16, len: u32) -> f32;
        #[cfg(target_arch = "loongarch64")]
        pub fn dot_bf16_lasx(ptr1: *const bf16, ptr2: *const bf16, len: u32) -> f32;
    }
}

impl Dot for bf16 {
    #[inline]
    fn dot(x: &[Self], y: &[Self]) -> f32 {
        match *SIMD_SUPPORT {
            #[cfg(all(feature = "fp16kernels", target_arch = "aarch64"))]
            SimdSupport::Neon => unsafe {
                bf16_kernel::dot_bf16_neon(x.as_ptr(), y.as_ptr(), x.len() as u32)
            },
            #[cfg(all(
                feature = "fp16kernels",
                kernel_support = "avx512",
                target_arch = "x86_64"
            ))]
            SimdSupport::Avx512FP16 => unsafe {
                bf16_kernel::dot_bf16_avx512(x.as_ptr(), y.as_ptr(), x.len() as u32)
            },
            #[cfg(all(feature = "fp16kernels", target_arch = "x86_64"))]
            SimdSupport::Avx2 | SimdSupport::Avx512 => unsafe {
                bf16_kernel::dot_bf16_avx2(x.as_ptr(), y.as_ptr(), x.len() as u32)
            },
            #[cfg(all(feature = "fp16kernels", target_arch = "loongarch64"))]
            SimdSupport::Lasx => unsafe {
                bf16_kernel::dot_bf16_lasx(x.as_ptr(), y.as_ptr(), x.len() as u32)
            },
            #[cfg(all(feature = "fp16kernels", target_arch = "loongarch64"))]
            SimdSupport::Lsx => unsafe {
                bf16_kernel::dot_bf16_lsx(x.as_ptr(), y.as_ptr(), x.len() as u32)
            },
            _ => dot_scalar::<Self, f32, 32>(x, y),
        }
    }
}

#[cfg(feature = "fp16kernels")]
mod kernel {
    use super::*;

    // These are the `dot_f16` function in f16.c. Our build.rs script compiles
    // a version of this file for each SIMD level with different suffixes.
    unsafe extern "C" {
        #[cfg(target_arch = "aarch64")]
        pub fn dot_f16_neon(ptr1: *const f16, ptr2: *const f16, len: u32) -> f32;
        #[cfg(all(kernel_support = "avx512", target_arch = "x86_64"))]
        pub fn dot_f16_avx512(ptr1: *const f16, ptr2: *const f16, len: u32) -> f32;
        #[cfg(target_arch = "x86_64")]
        pub fn dot_f16_avx2(ptr1: *const f16, ptr2: *const f16, len: u32) -> f32;
        #[cfg(target_arch = "loongarch64")]
        pub fn dot_f16_lsx(ptr1: *const f16, ptr2: *const f16, len: u32) -> f32;
        #[cfg(target_arch = "loongarch64")]
        pub fn dot_f16_lasx(ptr1: *const f16, ptr2: *const f16, len: u32) -> f32;
    }
}

impl Dot for f16 {
    #[inline]
    fn dot(x: &[Self], y: &[Self]) -> f32 {
        match *SIMD_SUPPORT {
            #[cfg(all(feature = "fp16kernels", target_arch = "aarch64"))]
            SimdSupport::Neon => unsafe {
                kernel::dot_f16_neon(x.as_ptr(), y.as_ptr(), x.len() as u32)
            },
            #[cfg(all(
                feature = "fp16kernels",
                kernel_support = "avx512",
                target_arch = "x86_64"
            ))]
            SimdSupport::Avx512FP16 => unsafe {
                kernel::dot_f16_avx512(x.as_ptr(), y.as_ptr(), x.len() as u32)
            },
            #[cfg(all(feature = "fp16kernels", target_arch = "x86_64"))]
            SimdSupport::Avx2 => unsafe {
                kernel::dot_f16_avx2(x.as_ptr(), y.as_ptr(), x.len() as u32)
            },
            #[cfg(all(feature = "fp16kernels", target_arch = "loongarch64"))]
            SimdSupport::Lasx => unsafe {
                kernel::dot_f16_lasx(x.as_ptr(), y.as_ptr(), x.len() as u32)
            },
            #[cfg(all(feature = "fp16kernels", target_arch = "loongarch64"))]
            SimdSupport::Lsx => unsafe {
                kernel::dot_f16_lsx(x.as_ptr(), y.as_ptr(), x.len() as u32)
            },
            _ => dot_scalar::<Self, f32, 32>(x, y),
        }
    }
}

impl Dot for f32 {
    #[inline]
    fn dot(x: &[Self], y: &[Self]) -> f32 {
        dot_scalar::<Self, Self, 16>(x, y)
    }
}

impl Dot for f64 {
    #[inline]
    fn dot(x: &[Self], y: &[Self]) -> f32 {
        dot_f64_simd(x, y)
    }
}

/// Dot product for f64. Runtime-dispatched to the best available backend.
///
/// On x86_64, dispatches via `SIMD_SUPPORT` to either an AVX2-targeted
/// kernel (which carries `#[target_feature(enable = "avx,avx2,fma")]` so it
/// stays correct under any compile baseline) or a portable scalar fallback.
/// On aarch64 and loongarch64, the SIMD primitives in `crate::simd::f64` are
/// unconditionally backed by NEON / LSX-LASX respectively, so no runtime gate
/// is required.
#[inline]
fn dot_f64_simd(x: &[f64], y: &[f64]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        match *SIMD_SUPPORT {
            SimdSupport::Avx2 | SimdSupport::Avx512 | SimdSupport::Avx512FP16 => unsafe {
                x86::dot_f64_avx2(x, y)
            },
            _ => dot_f64_scalar(x, y),
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        dot_f64_simd_other(x, y)
    }
}

/// Portable scalar dot product for f64. Used as the x86_64 fallback when no
/// AVX2 is detected, and exposed for cross-backend parity testing.
#[cfg(target_arch = "x86_64")]
#[inline]
fn dot_f64_scalar(x: &[f64], y: &[f64]) -> f32 {
    x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum::<f64>() as f32
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use crate::simd::f64::{f64x4, f64x8};
    use crate::simd::{FloatSimd, SIMD};

    /// AVX2 + FMA dot product for f64. Two-level unrolling: f64x8 main loop,
    /// f64x4 remainder, scalar tail.
    ///
    /// Caller must ensure the host supports AVX2 + FMA (gated by the
    /// `SIMD_SUPPORT` match in `super::dot_f64_simd`).
    #[target_feature(enable = "avx,avx2,fma")]
    pub unsafe fn dot_f64_avx2(x: &[f64], y: &[f64]) -> f32 {
        let dim = x.len();
        let unrolled_len = dim / 8 * 8;

        let mut acc8 = f64x8::zeros();
        for i in (0..unrolled_len).step_by(8) {
            let a = f64x8::load_unaligned(x.as_ptr().add(i));
            let b = f64x8::load_unaligned(y.as_ptr().add(i));
            acc8.multiply_add(a, b);
        }

        let aligned_len = dim / 4 * 4;
        let mut acc4 = f64x4::zeros();
        for i in (unrolled_len..aligned_len).step_by(4) {
            let a = f64x4::load_unaligned(x.as_ptr().add(i));
            let b = f64x4::load_unaligned(y.as_ptr().add(i));
            acc4.multiply_add(a, b);
        }

        let tail: f64 = x[aligned_len..]
            .iter()
            .zip(y[aligned_len..].iter())
            .map(|(&a, &b)| a * b)
            .sum();

        (acc8.reduce_sum() + acc4.reduce_sum() + tail) as f32
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
fn dot_f64_simd_other(x: &[f64], y: &[f64]) -> f32 {
    use crate::simd::f64::{f64x4, f64x8};
    use crate::simd::{FloatSimd, SIMD};

    let dim = x.len();
    let unrolled_len = dim / 8 * 8;

    let mut acc8 = f64x8::zeros();
    for i in (0..unrolled_len).step_by(8) {
        unsafe {
            let a = f64x8::load_unaligned(x.as_ptr().add(i));
            let b = f64x8::load_unaligned(y.as_ptr().add(i));
            acc8.multiply_add(a, b);
        }
    }

    let aligned_len = dim / 4 * 4;
    let mut acc4 = f64x4::zeros();
    for i in (unrolled_len..aligned_len).step_by(4) {
        unsafe {
            let a = f64x4::load_unaligned(x.as_ptr().add(i));
            let b = f64x4::load_unaligned(y.as_ptr().add(i));
            acc4.multiply_add(a, b);
        }
    }

    let tail: f64 = x[aligned_len..]
        .iter()
        .zip(y[aligned_len..].iter())
        .map(|(&a, &b)| a * b)
        .sum();

    (acc8.reduce_sum() + acc4.reduce_sum() + tail) as f32
}

impl Dot for u8 {
    #[inline]
    fn dot(x: &[Self], y: &[Self]) -> f32 {
        super::dot_u8::dot_u8(x, y) as f32
    }
}

/// Negative dot product, to present the relative order of dot distance.
pub fn dot_distance_batch<'a, T: Dot>(
    from: &'a [T],
    to: &'a [T],
    dimension: usize,
) -> Box<dyn Iterator<Item = f32> + 'a> {
    assume_eq!(from.len(), dimension);
    assume_eq!(to.len() % dimension, 0);
    Box::new(to.chunks_exact(dimension).map(|v| dot_distance(from, v)))
}

fn do_dot_distance_arrow_batch<T: ArrowFloatType>(
    from: &T::ArrayType,
    to: &FixedSizeListArray,
) -> Result<Arc<Float32Array>>
where
    T::Native: Dot,
{
    let dimension = to.value_length() as usize;
    debug_assert_eq!(from.len(), dimension);

    // TODO: if we detect there is a run of nulls, should we skip those?
    let to_values =
        to.values()
            .as_any()
            .downcast_ref::<T::ArrayType>()
            .ok_or(Error::InvalidArgumentError(format!(
                "Invalid type: expect {:?} got {:?}",
                from.data_type(),
                to.value_type()
            )))?;

    let dists = to_values
        .as_slice()
        .chunks_exact(dimension)
        .map(|v| dot_distance(from.as_slice(), v));

    Ok(Arc::new(Float32Array::new(
        dists.collect(),
        to.nulls().cloned(),
    )))
}

/// Compute negative dot product distance between a vector and a batch of vectors.
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
pub fn dot_distance_arrow_batch(
    from: &dyn Array,
    to: &FixedSizeListArray,
) -> Result<Arc<Float32Array>> {
    let dimension = to.value_length() as usize;
    debug_assert_eq!(from.len(), dimension);

    match *from.data_type() {
        DataType::Float16 => do_dot_distance_arrow_batch::<Float16Type>(from.as_primitive(), to),
        DataType::Float32 => do_dot_distance_arrow_batch::<Float32Type>(from.as_primitive(), to),
        DataType::Float64 => do_dot_distance_arrow_batch::<Float64Type>(from.as_primitive(), to),
        DataType::Int8 => do_dot_distance_arrow_batch::<Float32Type>(
            &from
                .as_primitive::<Int8Type>()
                .into_iter()
                .map(|x| x.unwrap() as f32)
                .collect(),
            &to.convert_to_floating_point()?,
        ),
        _ => Err(Error::InvalidArgumentError(format!(
            "Unsupported data type: {:?}",
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
    use num_traits::{Float, FromPrimitive};
    use proptest::prelude::*;

    #[test]
    fn test_dot() {
        let x: Vec<f32> = (0..20).map(|v| v as f32).collect();
        let y: Vec<f32> = (100..120).map(|v| v as f32).collect();

        assert_eq!(f32::dot(&x, &y), dot(&x, &y));

        let x: Vec<f32> = (0..512).map(|v| v as f32).collect();
        let y: Vec<f32> = (100..612).map(|v| v as f32).collect();

        assert_eq!(f32::dot(&x, &y), dot(&x, &y));

        let x: Vec<f16> = (0..20).map(|v| f16::from_i32(v).unwrap()).collect();
        let y: Vec<f16> = (100..120).map(|v| f16::from_i32(v).unwrap()).collect();
        assert_eq!(f16::dot(&x, &y), dot(&x, &y));

        let x: Vec<f64> = (20..40).map(|v| f64::from_i32(v).unwrap()).collect();
        let y: Vec<f64> = (120..140).map(|v| f64::from_i32(v).unwrap()).collect();
        assert_eq!(f64::dot(&x, &y), dot(&x, &y));
    }

    /// Reference implementation of dot product.
    fn dot_scalar_ref(x: &[f64], y: &[f64]) -> f32 {
        x.iter().zip(y.iter()).map(|(&x, &y)| x * y).sum::<f64>() as f32
    }

    /// Error bound for vector dot product
    /// http://ftp.demec.ufpr.br/CFD/bibliografia/Higham_2002_Accuracy%20and%20Stability%20of%20Numerical%20Algorithms.pdf
    /// Chapter 3 (page 61) equation 3.5
    /// A float point calculation error is bounded by:
    /// (kє/(1-kє)) Sum_i(|x_i||y_i|) if kє < 1
    /// We are currently using a SIMD version of naive product and summation.
    /// Therefore, k = 2n-1 (n multiplications, n-1 additions).
    /// For f16 and bf16, kє can be >=1.
    /// When that happens, we will use a simpler estimation method:
    /// Imagine that each `x_i` can vary by `є * |x_i|`, similarly for `y_i`.
    /// (Basically, it's accurate to ±(1 + є) * |x_i|).
    /// Error for `sum(x, y)` is `є_x + є_y`.
    /// Error for multiple is `є_x * x + є_y * y + є_x * є_y`,
    /// which simplifies to `є_x * x + є_y * y`
    /// See: https://www.geol.lsu.edu/jlorenzo/geophysics/uncertainties/Uncertaintiespart2.html
    /// The multiplication of `x_i` and `y_i` can vary by `є|x_i||y_i| + є|y_i||x_i|`.
    /// This simplifies to `2є|x_i||y_i|`.
    /// So the error for the sum of all the multiplications is `2є Sum_i(|x_i||y_i|)`.
    fn max_error<T: Float + AsPrimitive<f64>>(x: &[f64], y: &[f64]) -> f32 {
        let dot = x
            .iter()
            .cloned()
            .zip(y.iter().cloned())
            .map(|(x, y)| x.abs() * y.abs())
            .sum::<f64>();
        let k = ((2 * x.len()) - 1) as f64;
        let k_epsilon = k * T::epsilon().as_();

        if k_epsilon < 1.0 {
            (k_epsilon * dot) as f32
        } else {
            (2.0 * T::epsilon().as_() * dot) as f32
        }
    }

    fn do_dot_test<T: Dot + AsPrimitive<f64> + Float>(
        x: &[T],
        y: &[T],
    ) -> std::result::Result<(), TestCaseError> {
        let f64_x = x.iter().map(|&v| v.as_()).collect::<Vec<f64>>();
        let f64_y = y.iter().map(|&v| v.as_()).collect::<Vec<f64>>();

        let expected = dot_scalar_ref(&f64_x, &f64_y);
        let result = dot(x, y);

        let max_error = max_error::<T>(&f64_x, &f64_y);

        prop_assert!(approx::relative_eq!(expected, result, epsilon = max_error));
        Ok(())
    }

    proptest::proptest! {
        #[test]
        fn test_dot_f16((x, y) in arbitrary_vector_pair(arbitrary_f16, 4..4048)) {
            do_dot_test(&x, &y)?;
        }

        #[test]
        fn test_dot_bf16((x, y) in arbitrary_vector_pair(arbitrary_bf16, 4..4048)){
            do_dot_test(&x, &y)?;
        }

        #[test]
        fn test_dot_f32((x, y) in arbitrary_vector_pair(arbitrary_f32, 4..4048)){
            do_dot_test(&x, &y)?;
        }

        #[test]
        fn test_dot_f64((x, y) in arbitrary_vector_pair(arbitrary_f64, 4..4048)){
            do_dot_test(&x, &y)?;
        }

        /// Cross-backend parity: scalar fallback must match the dispatched
        /// SIMD path within numerical tolerance. Exercises `dot_f64_scalar`
        /// directly so future baseline-lowering work has CI coverage of the
        /// fallback path before any global RUSTFLAGS change.
        #[cfg(target_arch = "x86_64")]
        #[test]
        fn test_dot_f64_scalar_simd_parity(
            (x, y) in arbitrary_vector_pair(arbitrary_f64, 4..4048)
        ) {
            let scalar = dot_f64_scalar(&x, &y);
            let simd = dot_f64_simd(&x, &y);
            prop_assert!(approx::relative_eq!(scalar, simd, max_relative = 1e-6));
        }

        /// Explicit scalar-vs-AVX2 parity. The sibling
        /// `test_dot_f64_scalar_simd_parity` test trivially passes on a
        /// non-AVX2 host (e.g. the qemu-Nehalem CI runner) because the SIMD
        /// dispatcher takes the scalar fallback there — both sides of the
        /// assertion call `dot_f64_scalar`. This test forces the AVX2 inner
        /// function on hosts that actually have AVX2 so the AVX2 codegen
        /// gets exercised in CI.
        #[cfg(target_arch = "x86_64")]
        #[test]
        fn test_dot_f64_scalar_vs_avx2_parity(
            (x, y) in arbitrary_vector_pair(arbitrary_f64, 4..4048)
        ) {
            if !std::is_x86_feature_detected!("avx2") {
                return Ok(());
            }
            let scalar = dot_f64_scalar(&x, &y);
            let avx2 = unsafe { x86::dot_f64_avx2(&x, &y) };
            prop_assert!(approx::relative_eq!(scalar, avx2, max_relative = 1e-6));
        }
    }
}
