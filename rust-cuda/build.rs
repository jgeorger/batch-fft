use std::env;

fn main() {
    // Link to cuFFT library
    println!("cargo:rustc-link-lib=cufft");

    // Link to CUDA runtime
    println!("cargo:rustc-link-lib=cudart");

    // Add CUDA library search paths
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    } else {
        // Default CUDA paths
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
    }

    // Re-run if CUDA path changes
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
}
