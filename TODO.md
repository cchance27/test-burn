# `metallic` Module Improvement Plan

** 1. Remaining Todo Items**
-   [ ] **1.1. Other kernel optimizations and primitives** 
    -  Look into other optimizations for our kernels like, Kahan Summation, Hierarchical Reductions and Warp-Level Primitives 
-   [ ] **1.2. Add Comprehensive Documentation.**
    -   Add `#[doc = "..."]` comments to all public structs, traits, and functions.
    -   Explain the purpose, arguments, and safety considerations for each item.
    -   Provide examples where appropriate.

** 2. Offloading**
-   [ ] **2.1** Add automatic offloading based on GPU memory
-   [ ] **2.2** Optimize memory usage for inference