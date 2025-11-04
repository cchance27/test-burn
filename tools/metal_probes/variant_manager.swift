import Foundation

struct VariantManager {
    static let backendDisplayOrder: [MatmulShapeSpec.Backend] = [.mlx, .m1Optimized, .m1OptimizedV2, .m1OptimizedV3, .mps, .gemv, .gemmTiled]
    
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
    
    static func variantSupports(backend: MatmulShapeSpec.Backend, variant: KernelVariant, spec: MatmulShapeSpec) -> Bool {
        // If no support information is available, assume it's supported
        guard let supports = variant.supports else {
            return true
        }
        
        // Check each dimension-specific support
        if !supports.transposeA && spec.transposeA {
            return false
        }
        if !supports.transposeB && spec.transposeB {
            return false
        }
        if let expectedA = supports.expectedTransposeA, spec.transposeA != expectedA {
            return false
        }
        if let expectedB = supports.expectedTransposeB, spec.transposeB != expectedB {
            return false
        }
        if !supports.batch && spec.batch > 1 {
            return false
        }
        if !supports.bias && spec.bias {
            return false
        }
        if !supports.accumulate && (spec.accumulate || spec.beta != 0.0) {
            return false
        }
        
        // Check if it's a small dimension case
        if !supports.smallMN && (spec.m <= 16 || spec.n <= 16) {
            return false
        }
        if !supports.smallK && spec.k < 64 {
            return false
        }
        
        // Check specific N values for GEMV
        if backend == .gemv, 
           let supportedNValues = supports.supportedNValues, 
           !supportedNValues.contains(spec.n) {
            return false
        }
        
        return true
    }
}
