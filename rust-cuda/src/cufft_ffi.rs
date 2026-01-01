use num_complex::Complex32;
use std::fmt;
use std::ptr;

// cuFFT handle type
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cufftHandle(i32);

// cuFFT result codes
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum cufftResult {
    CUFFT_SUCCESS = 0,
    CUFFT_INVALID_PLAN = 1,
    CUFFT_ALLOC_FAILED = 2,
    CUFFT_INVALID_TYPE = 3,
    CUFFT_INVALID_VALUE = 4,
    CUFFT_INTERNAL_ERROR = 5,
    CUFFT_EXEC_FAILED = 6,
    CUFFT_SETUP_FAILED = 7,
    CUFFT_INVALID_SIZE = 8,
    CUFFT_UNALIGNED_DATA = 9,
}

impl fmt::Display for cufftResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            cufftResult::CUFFT_SUCCESS => write!(f, "CUFFT_SUCCESS"),
            cufftResult::CUFFT_INVALID_PLAN => write!(f, "CUFFT_INVALID_PLAN"),
            cufftResult::CUFFT_ALLOC_FAILED => write!(f, "CUFFT_ALLOC_FAILED"),
            cufftResult::CUFFT_INVALID_TYPE => write!(f, "CUFFT_INVALID_TYPE"),
            cufftResult::CUFFT_INVALID_VALUE => write!(f, "CUFFT_INVALID_VALUE"),
            cufftResult::CUFFT_INTERNAL_ERROR => write!(f, "CUFFT_INTERNAL_ERROR"),
            cufftResult::CUFFT_EXEC_FAILED => write!(f, "CUFFT_EXEC_FAILED"),
            cufftResult::CUFFT_SETUP_FAILED => write!(f, "CUFFT_SETUP_FAILED"),
            cufftResult::CUFFT_INVALID_SIZE => write!(f, "CUFFT_INVALID_SIZE"),
            cufftResult::CUFFT_UNALIGNED_DATA => write!(f, "CUFFT_UNALIGNED_DATA"),
        }
    }
}

impl std::error::Error for cufftResult {}

// cuFFT transform types
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum cufftType {
    CUFFT_C2C = 0x29, // Complex-to-Complex
}

// cuFFT transform directions
pub const CUFFT_FORWARD: i32 = -1;
pub const CUFFT_INVERSE: i32 = 1;

// FFI declarations
extern "C" {
    pub fn cufftPlanMany(
        plan: *mut cufftHandle,
        rank: i32,
        n: *const i32,
        inembed: *const i32,
        istride: i32,
        idist: i32,
        onembed: *const i32,
        ostride: i32,
        odist: i32,
        type_: cufftType,
        batch: i32,
    ) -> cufftResult;

    pub fn cufftExecC2C(
        plan: cufftHandle,
        idata: *mut Complex32,
        odata: *mut Complex32,
        direction: i32,
    ) -> cufftResult;

    pub fn cufftDestroy(plan: cufftHandle) -> cufftResult;
}

// Safe wrapper for cuFFT plan
pub struct CufftPlan {
    handle: cufftHandle,
}

impl CufftPlan {
    /// Create a new batch 1D FFT plan
    /// Matches the parameters from cuda/src/batch_fft.cu:56-68
    pub fn new_batch_1d(length: usize, batch: usize) -> Result<Self, cufftResult> {
        let mut handle = cufftHandle(0);
        let n = [length as i32];

        unsafe {
            let result = cufftPlanMany(
                &mut handle,
                1,                      // rank (1D)
                n.as_ptr(),            // dimensions of each FFT
                ptr::null(),           // inembed (NULL = same as n)
                1,                     // istride (elements are contiguous)
                length as i32,         // idist (distance between FFTs)
                ptr::null(),           // onembed (NULL = same as n)
                1,                     // ostride
                length as i32,         // odist
                cufftType::CUFFT_C2C, // type (complex-to-complex)
                batch as i32,          // batch size
            );

            if result != cufftResult::CUFFT_SUCCESS {
                return Err(result);
            }
        }

        Ok(Self { handle })
    }

    /// Execute forward FFT in-place
    pub fn execute_forward(&self, data: *mut Complex32) -> Result<(), String> {
        unsafe {
            let result = cufftExecC2C(self.handle, data, data, CUFFT_FORWARD);
            if result != cufftResult::CUFFT_SUCCESS {
                return Err(format!("cuFFT execution failed: {}", result));
            }
        }

        // Synchronize device to ensure completion
        // Matches cuda/src/batch_fft.cu:74
        crate::cuda_ffi::cuda_device_synchronize()
            .map_err(|e| format!("CUDA synchronization failed: {}", e))?;

        Ok(())
    }
}

impl Drop for CufftPlan {
    fn drop(&mut self) {
        unsafe {
            let result = cufftDestroy(self.handle);
            if result != cufftResult::CUFFT_SUCCESS {
                eprintln!("Warning: cufftDestroy failed: {:?}", result);
            }
        }
    }
}
