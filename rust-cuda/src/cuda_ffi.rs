use std::ffi::c_void;
use std::fmt;
use std::ptr;

// CUDA error types
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum cudaError_t {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    // Add more as needed
}

impl fmt::Display for cudaError_t {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            cudaError_t::cudaSuccess => write!(f, "cudaSuccess"),
            cudaError_t::cudaErrorInvalidValue => write!(f, "cudaErrorInvalidValue"),
            cudaError_t::cudaErrorMemoryAllocation => write!(f, "cudaErrorMemoryAllocation"),
            _ => write!(f, "cudaError({})", *self as i32),
        }
    }
}

impl std::error::Error for cudaError_t {}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum cudaMemcpyKind {
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
}

// Opaque handle for CUDA events
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cudaEvent_t(*mut c_void);

unsafe impl Send for cudaEvent_t {}
unsafe impl Sync for cudaEvent_t {}

// FFI declarations
extern "C" {
    pub fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> cudaError_t;
    pub fn cudaFree(devPtr: *mut c_void) -> cudaError_t;
    pub fn cudaMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t;
    pub fn cudaDeviceSynchronize() -> cudaError_t;
    pub fn cudaEventCreate(event: *mut cudaEvent_t) -> cudaError_t;
    pub fn cudaEventRecord(event: cudaEvent_t, stream: *mut c_void) -> cudaError_t;
    pub fn cudaEventSynchronize(event: cudaEvent_t) -> cudaError_t;
    pub fn cudaEventElapsedTime(
        ms: *mut f32,
        start: cudaEvent_t,
        end: cudaEvent_t,
    ) -> cudaError_t;
    pub fn cudaEventDestroy(event: cudaEvent_t) -> cudaError_t;
}

// Safe wrapper for CUDA device buffer
pub struct CudaBuffer<T> {
    ptr: *mut T,
    len: usize,
}

impl<T> CudaBuffer<T> {
    pub fn alloc(len: usize) -> Result<Self, cudaError_t> {
        let mut ptr: *mut c_void = ptr::null_mut();
        let size = len * std::mem::size_of::<T>();

        unsafe {
            let result = cudaMalloc(&mut ptr, size);
            if result != cudaError_t::cudaSuccess {
                return Err(result);
            }
        }

        Ok(Self {
            ptr: ptr as *mut T,
            len,
        })
    }

    pub fn copy_from_host(&mut self, host_data: &[T]) -> Result<(), cudaError_t> {
        assert_eq!(host_data.len(), self.len, "Size mismatch");

        unsafe {
            let result = cudaMemcpy(
                self.ptr as *mut c_void,
                host_data.as_ptr() as *const c_void,
                self.len * std::mem::size_of::<T>(),
                cudaMemcpyKind::cudaMemcpyHostToDevice,
            );

            if result != cudaError_t::cudaSuccess {
                return Err(result);
            }
        }

        Ok(())
    }

    pub fn as_ptr(&self) -> *mut T {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

impl<T> Drop for CudaBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            let result = cudaFree(self.ptr as *mut c_void);
            if result != cudaError_t::cudaSuccess {
                eprintln!("Warning: cudaFree failed: {:?}", result);
            }
        }
    }
}

// Safe wrapper for CUDA events
pub struct CudaEvent {
    event: cudaEvent_t,
}

impl CudaEvent {
    pub fn create() -> Result<Self, cudaError_t> {
        let mut event = cudaEvent_t(ptr::null_mut());

        unsafe {
            let result = cudaEventCreate(&mut event);
            if result != cudaError_t::cudaSuccess {
                return Err(result);
            }
        }

        Ok(Self { event })
    }

    pub fn record(&self) -> Result<(), cudaError_t> {
        unsafe {
            let result = cudaEventRecord(self.event, ptr::null_mut());
            if result != cudaError_t::cudaSuccess {
                return Err(result);
            }
        }
        Ok(())
    }

    pub fn synchronize(&self) -> Result<(), cudaError_t> {
        unsafe {
            let result = cudaEventSynchronize(self.event);
            if result != cudaError_t::cudaSuccess {
                return Err(result);
            }
        }
        Ok(())
    }

    pub fn elapsed_time(&self, end: &CudaEvent) -> Result<f32, cudaError_t> {
        let mut ms: f32 = 0.0;

        unsafe {
            let result = cudaEventElapsedTime(&mut ms, self.event, end.event);
            if result != cudaError_t::cudaSuccess {
                return Err(result);
            }
        }

        Ok(ms)
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        unsafe {
            let result = cudaEventDestroy(self.event);
            if result != cudaError_t::cudaSuccess {
                eprintln!("Warning: cudaEventDestroy failed: {:?}", result);
            }
        }
    }
}

// Helper function for device synchronization
pub fn cuda_device_synchronize() -> Result<(), cudaError_t> {
    unsafe {
        let result = cudaDeviceSynchronize();
        if result != cudaError_t::cudaSuccess {
            return Err(result);
        }
    }
    Ok(())
}
