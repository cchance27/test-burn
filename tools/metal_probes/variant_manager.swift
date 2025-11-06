import Foundation

struct VariantManager {
    static let backendDisplayOrder: [MatmulShapeSpec.Backend] = [.mlx, .m1Optimized, .m1OptimizedV2, .m1OptimizedV3, .m1OptimizedV4, .m1OptimizedV5, .m1OptimizedV6, .m1OptimizedV7, .mps, .gemv, .gemmTiled]
    
    static func loadVariants(matmulDir: URL) -> [MatmulShapeSpec.Backend: [KernelVariant]] {
        let fileManager = FileManager.default
        
        // Use enhanced file as the primary source
        let enhancedManifestURL = matmulDir.appendingPathComponent("variants_enhanced.json")
        let sourceURL = enhancedManifestURL
        
        guard fileManager.fileExists(atPath: sourceURL.path) else {
            print("Error: Enhanced variants manifest not found.")
            return [:]
        }

        do {
            let data = try Data(contentsOf: sourceURL)
            let decoder = JSONDecoder()
            let rawManifest = try decoder.decode([String: [KernelVariantConfig]].self, from: data)
            var mapping: [MatmulShapeSpec.Backend: [KernelVariant]] = [:]
            for (key, configs) in rawManifest {
                guard let backend = MatmulShapeSpec.Backend(rawValue: key.lowercased()) else {
                    print("Warning: Unknown backend in variants manifest: \(key)")
                    continue
                }
                let enabledConfigs = configs.filter { $0.enabled ?? true }
                if enabledConfigs.isEmpty {
                    continue
                }
                let variants = buildVariants(from: enabledConfigs, backend: backend)
                if !variants.isEmpty {
                    mapping[backend] = variants
                }
            }
            return mapping
        } catch {
            print("Error: Failed to load enhanced variants manifest (\(error)).")
            return [:]
        }
    }

    private static func buildVariants(from configs: [KernelVariantConfig], backend: MatmulShapeSpec.Backend) -> [KernelVariant] {
        var variants: [KernelVariant] = []
        var hasBaseline = false
        for config in configs {
            let libraryName = config.library
            let tileOverride: (Int, Int, Int, Int, Int)?
            if let bm = config.tileBM, let bn = config.tileBN, let bk = config.tileBK, let wm = config.warpM, let wn = config.warpN {
                tileOverride = (bm, bn, bk, wm, wn)
            } else {
                tileOverride = nil
            }
            let transposeBOverride = config.transposeBOverride
            variants.append(
                KernelVariant(
                    name: config.name,
                    library: libraryName,
                    isBaseline: config.baseline ?? false,
                    tileOverride: tileOverride,
                    transposeBOverride: transposeBOverride,
                    supports: config.supports
                )
            )
            if config.baseline == true {
                hasBaseline = true
            }
        }
        if !hasBaseline, let first = variants.first {
            variants[0] = KernelVariant(
                name: first.name,
                library: first.library,
                isBaseline: true,
                tileOverride: first.tileOverride,
                transposeBOverride: first.transposeBOverride,
                supports: first.supports
            )
        }
        return variants
    }
    
    static func variantSupports(backend: MatmulShapeSpec.Backend, variant: KernelVariant, spec: MatmulShapeSpec) -> (Bool, String?) {
        // Apply overrides (e.g., transposeB) before evaluating support constraints
        let specToCheck: MatmulShapeSpec
        if let overrideB = variant.transposeBOverride {
            specToCheck = MatmulShapeSpec(
                op: spec.op,
                backend: spec.backend,
                batch: spec.batch,
                m: spec.m,
                n: spec.n,
                k: spec.k,
                transposeA: spec.transposeA,
                transposeB: overrideB,
                stridedBatch: spec.stridedBatch,
                accumulate: spec.accumulate,
                alpha: spec.alpha,
                beta: spec.beta,
                bias: spec.bias
            )
        } else {
            specToCheck = spec
        }

        // If no support information is available, assume it's supported
        guard let supports = variant.supports else {
            return (true, nil)
        }
        
        // Check each dimension-specific support
        if !supports.transposeA && specToCheck.transposeA {
            return (false, "transposeA not supported")
        }
        if !supports.transposeB && specToCheck.transposeB {
            return (false, "transposeB not supported")
        }
        if let expectedA = supports.expectedTransposeA, specToCheck.transposeA != expectedA {
            return (false, "transposeA=\(specToCheck.transposeA) but expected \(expectedA)")
        }
        if let expectedB = supports.expectedTransposeB, specToCheck.transposeB != expectedB {
            return (false, "transposeB=\(specToCheck.transposeB) but expected \(expectedB)")
        }
        if !supports.batch && specToCheck.batch > 1 {
            return (false, "batch=\(specToCheck.batch) not supported")
        }
        if !supports.bias && specToCheck.bias {
            return (false, "bias not supported")
        }
        if !supports.accumulate && (specToCheck.accumulate || specToCheck.beta != 0.0) {
            return (false, "accumulate not supported")
        }
        
        // Check if it's a small dimension case
        if !supports.smallMN && (specToCheck.m <= 16 || specToCheck.n <= 16) {
            return (false, "smallMN not supported (m=\(specToCheck.m), n=\(specToCheck.n))")
        }
        if !supports.smallK && specToCheck.k < 64 {
            return (false, "smallK not supported (k=\(specToCheck.k))")
        }
        
        // Check specific N values for GEMV
        if backend == .gemv, 
           let supportedNValues = supports.supportedNValues, 
           !supportedNValues.contains(specToCheck.n) {
            return (false, "n=\(specToCheck.n) not in supportedNValues")
        }
        
        // Heuristic gating for specialized v4/v5 kernels by name tokens
        let vname = variant.name
        // Ultra-tiny: target small shapes only
        if vname.contains("ultra_tiny") {
            if !(specToCheck.n <= 2048 && specToCheck.k <= 2048 && specToCheck.m == 1) { 
                return (false, "ultra_tiny requires n<=2048, k<=2048, m=1 (got m=\(specToCheck.m), n=\(specToCheck.n), k=\(specToCheck.k))")
            }
        }
        // Fused-bias kernels must only run when bias is requested
        if vname.contains("fused_bias") && !specToCheck.bias {
            return (false, "fused_bias requires bias=true")
        }
        // LargeK smallN: require large K and small N
        if vname.contains("largek_smalln") {
            if !(specToCheck.k >= 2048 && specToCheck.n <= 2048 && specToCheck.m == 1) { 
                return (false, "largek_smalln requires k>=2048, n<=2048, m=1 (got m=\(specToCheck.m), n=\(specToCheck.n), k=\(specToCheck.k))")
            }
        }
        // Debug: allow dbg variants regardless of bn/tg mapping but keep largeK gating
        if vname.contains("dbg") {
            if !(specToCheck.k >= 2048 && specToCheck.n >= 32 && specToCheck.m == 1) { 
                return (false, "dbg requires k>=2048, n>=32, m=1 (got m=\(specToCheck.m), n=\(specToCheck.n), k=\(specToCheck.k))")
            }
        }
        // smalln kernels: only run when N is truly small (<=16)
        if vname.contains("smalln") && !vname.contains("largek_smalln") {
            if !(specToCheck.n <= 16 && specToCheck.m == 1) { 
                return (false, "smalln requires n<=16, m=1 (got m=\(specToCheck.m), n=\(specToCheck.n))")
            }
        }
        // bn256 large-N vec4: prefer N large
        if vname.contains("bn256") && vname.contains("tgread") {
            if !(spec.n >= 4096 && spec.m == 1) { 
                return (false, "bn256+tgread requires n>=4096, m=1 (got m=\(spec.m), n=\(spec.n))")
            }
        }

        return (true, nil)
    }
}
