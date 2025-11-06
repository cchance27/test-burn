import Foundation
import Metal
import MetalPerformanceShaders

extension MTLDevice {
    func makeBuffer<T>(array: [T]) -> MTLBuffer {
        precondition(!array.isEmpty, "Buffer array must not be empty")
        let length = array.count * MemoryLayout<T>.stride
        return array.withUnsafeBytes { rawBuffer in
            guard let buffer = makeBuffer(bytes: rawBuffer.baseAddress!, length: length, options: .storageModeShared) else {
                fatalError("Failed to allocate Metal buffer")
            }
            return buffer
        }
    }
}

final class MatmulHarness {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let libraries: [String: MTLLibrary]
    private let buildDir: URL
    private let matmulDir: URL
    private let iterations: Int
    private let warmup: Int
    private let variants: [MatmulShapeSpec.Backend: [KernelVariant]]
    private let enabledBackends: [MatmulShapeSpec.Backend]
    private let compareAllBackends: Bool
    private let pipelineCache = PipelineCache()
    private var tensorCache: [MatmulShapeSpec: MatmulTensors] = [:]
    private(set) var variantFailures: [VariantFailure] = []

    init(buildDir: URL, matmulDir: URL) throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw HarnessError.metalUnavailable
        }
        guard let queue = device.makeCommandQueue() else {
            throw HarnessError.commandQueueUnavailable
        }
        self.device = device
        self.commandQueue = queue
        self.buildDir = buildDir
        self.matmulDir = matmulDir
        self.iterations = Int(ProcessInfo.processInfo.environment["MATMUL_BENCH_ITERS"] ?? "") ?? 8
        self.warmup = Int(ProcessInfo.processInfo.environment["MATMUL_BENCH_WARMUP"] ?? "") ?? 2
        self.libraries = try MatmulHarness.loadLibraries(device: device, buildDir: buildDir)
        let variantsMapping = VariantManager.loadVariants(matmulDir: matmulDir)
        self.variants = variantsMapping

        let defaultBackends = VariantManager.backendDisplayOrder.filter { variantsMapping[$0] != nil }
        let backendEnv = ProcessInfo.processInfo.environment["MATMUL_BACKENDS"]
        if let env = backendEnv, !env.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            let requested = env
                .split(separator: ",")
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() }
                .compactMap { MatmulShapeSpec.Backend(rawValue: $0) }
                .filter { variantsMapping[$0] != nil }
            self.enabledBackends = requested.isEmpty ? defaultBackends : requested
        } else {
            self.enabledBackends = defaultBackends
        }

        let compareEnv = ProcessInfo.processInfo.environment["MATMUL_COMPARE_ALL"]
        self.compareAllBackends = MatmulHarness.parseEnvBool(compareEnv, defaultValue: true)

        try validateVariantLibraries()
    }

    private static func parseEnvBool(_ raw: String?, defaultValue: Bool) -> Bool {
        guard let raw = raw else { return defaultValue }
        let normalized = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if normalized.isEmpty { return defaultValue }
        switch normalized {
        case "0", "false", "no", "off":
            return false
        case "1", "true", "yes", "on":
            return true
        default:
            return defaultValue
        }
    }

    private func validateVariantLibraries() throws {
        var missing: [String] = []
        for (backend, variants) in variants {
            for variant in variants {
                guard let libraryName = variant.library else { continue }
                if libraries[libraryName] == nil {
                    missing.append("\(backend.rawValue)::\(variant.name) expects \(libraryName).metallib")
                }
            }
        }
        if !missing.isEmpty {
            throw HarnessError.libraryLoadFailed(missing.joined(separator: ", "))
        }
    }
    
    private func candidateBackends(for spec: MatmulShapeSpec) -> [MatmulShapeSpec.Backend] {
        if compareAllBackends {
            return enabledBackends
        }
        return enabledBackends.contains(spec.backend) ? [spec.backend] : []
    }

    private func runVariant(spec: MatmulShapeSpec, backend: MatmulShapeSpec.Backend, variant: KernelVariant, tensors: MatmulTensors) throws -> BenchmarkResult {
        // Use unified runner for all kernel backends, MPS as special case
        switch backend {
        case .mps:
            return try runMPS(spec: spec, variant: variant, tensors: tensors)
        default:
            // For all other backends, use the unified runner
            let runner = UnifiedBackendRunner(
                device: device,
                commandQueue: commandQueue,
                libraries: libraries,
                iterations: iterations,
                warmup: warmup,
                pipelineCache: self.pipelineCache
            )
            return try runner.runVariant(spec: spec, backend: backend, variant: variant, tensors: tensors)
        }
    }

    private func tensorCacheKey(for spec: MatmulShapeSpec) -> String {
        let tA_str = spec.transposeA ? "1" : "0"
        let tB_str = spec.transposeB ? "1" : "0"
        let bias_str = spec.bias ? "1" : "0"
        let alpha_str = String(spec.alpha).replacingOccurrences(of: ".", with: "_")
        let beta_str = String(spec.beta).replacingOccurrences(of: ".", with: "_")

        let op_str = spec.op.replacingOccurrences(of: "[^a-zA-Z0-9_]", with: "_", options: .regularExpression)
        let backend_str = spec.backend.rawValue.replacingOccurrences(of: "[^a-zA-Z0-9_]", with: "_", options: .regularExpression)

        return "spec_\(op_str)_\(backend_str)_m\(spec.m)_n\(spec.n)_k\(spec.k)_tA\(tA_str)_tB\(tB_str)_alpha\(alpha_str)_beta\(beta_str)_bias\(bias_str)_batch\(spec.batch).bin"
    }

    private func cacheDirectory() -> URL {
        let cacheDir = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(".cache")
        try? FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        return cacheDir
    }

    /// Returns the cache file URL for a variant's benchmark results.
    /// 
    /// The cache file naming strategy ties the cache to the specific version of the Metal kernel:
    /// - With hash: `cache_{variant_name}.{library_hash}.json`
    /// - Without hash (e.g., MPS): `cache_{variant_name}.json`
    ///
    /// When a .metal file changes, it gets a new hash in the compiled .metallib filename.
    /// The cache files automatically use this new hash, and old cache files with outdated
    /// hashes are cleaned up. This ensures:
    /// 1. Cache invalidation when kernels change
    /// 2. No accidental reuse of results from old kernel versions
    /// 3. No collision between caches for different kernel libraries
    private func resultsCacheFileURL(for variantName: String, libraryHash: String?) -> URL {
        let sanitized = variantName.replacingOccurrences(of: "/", with: "_")
        if let hash = libraryHash {
            return cacheDirectory().appendingPathComponent("cache_\(sanitized).\(hash).json")
        } else {
            return cacheDirectory().appendingPathComponent("cache_\(sanitized).json")
        }
    }

    /// Extracts the hash from a compiled metallib filename.
    ///
    /// The build script compiles Metal kernels with hash-based naming:
    /// `{library_name}.{sha256_hash}.metallib`
    ///
    /// This function finds the metallib file for a given library and extracts its hash,
    /// which is then used to name cache files appropriately.
    private func getLibraryHash(for libraryName: String?) -> String? {
        guard let libName = libraryName else { return nil }
        
        // Check the build directory for metallib files matching this library name
        let fileManager = FileManager.default
        guard let contents = try? fileManager.contentsOfDirectory(at: buildDir, includingPropertiesForKeys: nil) else {
            return nil
        }
        
        for file in contents where file.pathExtension == "metallib" {
            let fileName = file.deletingPathExtension().lastPathComponent
            // Format is: basename.hash.metallib
            let components = fileName.split(separator: ".")
            if components.count >= 2 {
                let baseName = String(components[0])
                if baseName == libName {
                    // The hash is everything after the first dot
                    let hash = components.dropFirst().joined(separator: ".")
                    return String(hash)
                }
            }
        }
        return nil
    }
    
    /// Cleans up outdated cache files for a specific variant.
    ///
    /// When a Metal kernel is modified, it gets a new hash. This function removes cache files
    /// that have the old hash for this specific variant, ensuring we don't use stale results.
    ///
    /// Important: This only removes caches for the SPECIFIC variant being processed, not for
    /// other variants that might use the same library. The cache filename includes both the
    /// variant name and the library hash, so each variant's caches are independent.
    ///
    /// Example:
    /// - Variant: `m1_optimized_v2/nt_bn128_col`
    /// - Current hash: `abc123...`
    /// - Removes: `cache_m1_optimized_v2_nt_bn128_col.xyz789....json`
    /// - Keeps: `cache_m1_optimized_v2_nt_bn128_col.abc123....json`
    /// - Keeps: `cache_m1_optimized_v2_other_variant.xyz789....json` (different variant)
    private func cleanupOldCaches(for variantName: String, currentHash: String?) {
        let fileManager = FileManager.default
        let cacheDir = cacheDirectory()
        let sanitized = variantName.replacingOccurrences(of: "/", with: "_")
        
        guard let contents = try? fileManager.contentsOfDirectory(at: cacheDir, includingPropertiesForKeys: nil) else {
            return
        }
        
        // Find all cache files for this specific variant
        let prefix = "cache_\(sanitized)"
        for file in contents where file.lastPathComponent.hasPrefix(prefix) && file.pathExtension == "json" {
            let fileName = file.deletingPathExtension().lastPathComponent
            
            // Check if this is an old cache file with a different hash
            if let hash = currentHash {
                // Expected format: cache_variantname.hash.json
                let expectedName = "cache_\(sanitized).\(hash)"
                
                // Only remove if:
                // 1. The file starts with our specific variant name prefix
                // 2. The file has a hash extension (contains a dot after sanitized name)
                // 3. The hash doesn't match our current hash
                if fileName.hasPrefix(prefix + ".") && fileName != expectedName {
                    // This is an old cache file for THIS variant, remove it
                    try? fileManager.removeItem(at: file)
                    print("Removed outdated cache: \(file.lastPathComponent)")
                }
            } else {
                // If current hash is nil (e.g., MPS backend without library), clean up any hash-suffixed caches
                // that might be leftover from when this variant had a library
                if fileName.hasPrefix(prefix + ".") && fileName != "cache_\(sanitized)" {
                    try? fileManager.removeItem(at: file)
                    print("Removed hash-suffixed cache for non-library variant: \(file.lastPathComponent)")
                }
            }
        }
    }

    private func loadCachedResults(for variantName: String, libraryHash: String?) -> [BenchmarkResult]? {
        let cacheURL = resultsCacheFileURL(for: variantName, libraryHash: libraryHash)
        guard FileManager.default.fileExists(atPath: cacheURL.path) else { return nil }
        do {
            let data = try Data(contentsOf: cacheURL)
            return try JSONDecoder().decode([BenchmarkResult].self, from: data)
        } catch {
            print("Warning: Could not load results cache for \(variantName): \(error)")
            return nil
        }
    }

    private func saveCachedResults(_ results: [BenchmarkResult], for variantName: String, libraryHash: String?) {
        let cacheURL = resultsCacheFileURL(for: variantName, libraryHash: libraryHash)
        do {
            let data = try JSONEncoder().encode(results)
            try data.write(to: cacheURL)
        } catch {
            print("Warning: Could not save results cache for \(variantName): \(error)")
        }
    }

    private func prepareTensorCache(specs: [MatmulShapeSpec]) throws {
        tensorCache.removeAll()
        let tensorCacheDir = cacheDirectory().appendingPathComponent("tensors")
        try? FileManager.default.createDirectory(at: tensorCacheDir, withIntermediateDirectories: true)

        let uniqueSpecs = Set(specs)
        print("Pre-computing or loading \(uniqueSpecs.count) unique CPU reference tensors...")

        for spec in uniqueSpecs {
            let cacheKey = tensorCacheKey(for: spec)
            let cacheFileURL = tensorCacheDir.appendingPathComponent(cacheKey)

            if FileManager.default.fileExists(atPath: cacheFileURL.path) {
                do {
                    let data = try Data(contentsOf: cacheFileURL)
                    if let cachedData = CPUReferenceCache(data: data) {
                        tensorCache[spec] = MatmulTensors(
                            a: device.makeBuffer(array: cachedData.aValues),
                            b: device.makeBuffer(array: cachedData.bValues),
                            bias: cachedData.biasValues.map { device.makeBuffer(array: $0) },
                            output: device.makeBuffer(array: cachedData.initialOutput),
                            cpuReferenceFloat: cachedData.cpuReferenceFloat,
                            initialOutput: cachedData.initialOutput,
                            aLayout: cachedData.aLayout,
                            bLayout: cachedData.bLayout
                        )
                        print("Loaded tensor from cache: \(cacheKey)")
                        continue // Skip to next spec
                    } else {
                        print("Warning: Could not decode tensor cache for spec key \(cacheKey). Regenerating.")
                        try? FileManager.default.removeItem(at: cacheFileURL)
                    }
                } catch {
                    print("Warning: Could not load tensor cache for spec key \(cacheKey): \(error). Regenerating.")
                    try? FileManager.default.removeItem(at: cacheFileURL)
                }
            }

            // If not loaded from cache, generate new data
            print("Generating new tensor: \(cacheKey)")
            let rawData = generateRawTensorData(spec: spec)
            tensorCache[spec] = MatmulTensors(
                a: device.makeBuffer(array: rawData.aValues),
                b: device.makeBuffer(array: rawData.bValues),
                bias: rawData.biasValues.map { device.makeBuffer(array: $0) },
                output: device.makeBuffer(array: rawData.initialOutput),
                cpuReferenceFloat: rawData.cpuReferenceFloat,
                initialOutput: rawData.initialOutput,
                aLayout: rawData.aLayout,
                bLayout: rawData.bLayout
            )

            do {
                let data = rawData.toData()
                try data.write(to: cacheFileURL)
            } catch {
                print("Warning: Could not save tensor cache for spec key \(cacheKey): \(error)")
            }
        }
        print("Tensor pre-computation and loading complete.")
    }

    func run(specs: [MatmulShapeSpec]) throws -> [BenchmarkResult] {
        variantFailures.removeAll()
        try prepareTensorCache(specs: specs)

        var allResults: [BenchmarkResult] = []
        var allCachedResults: [String: [BenchmarkResult]] = [:]
        var variantHashes: [String: String?] = [:]
        
        var variantSpecs: [String: [MatmulShapeSpec]] = [:]
        
        for spec in specs {
            let backends = candidateBackends(for: spec)
            if backends.isEmpty {
                variantFailures.append(VariantFailure(spec: spec, backend: spec.backend, variantName: "<none>", errorDescription: "no enabled backends"))
                continue
            }

            for backend in backends {
                guard let variantList = variants[backend], !variantList.isEmpty else { continue }
                for variant in variantList {
                    let key = "\(backend.rawValue)/\(variant.name)"
                    let (isSupported, reason) = VariantManager.variantSupports(backend: backend, variant: variant, spec: spec)
                    if isSupported {
                        variantSpecs[key, default: []].append(spec)
                    } else {
                        let errorDesc = reason ?? "unsupported"
                        variantFailures.append(VariantFailure(spec: spec, backend: backend, variantName: variant.name, errorDescription: errorDesc))
                        if variantSpecs[key] == nil { variantSpecs[key] = [] }
                    }
                }
            }
        }
        
        // Get library hashes and clean up old caches
        for (variantName, _) in variantSpecs {
            let parts = variantName.split(separator: "/")
            if parts.count >= 2 {
                let backendStr = String(parts[0])
                let varName = parts.dropFirst().joined(separator: "/")
                if let backend = MatmulShapeSpec.Backend(rawValue: backendStr),
                   let variantList = variants[backend],
                   let variant = variantList.first(where: { $0.name == varName }) {
                    let hash = getLibraryHash(for: variant.library)
                    variantHashes[variantName] = hash
                    cleanupOldCaches(for: variantName, currentHash: hash)
                }
            }
        }
        
        for variantName in variantSpecs.keys {
            let hash = variantHashes[variantName] ?? nil
            if let cached = loadCachedResults(for: variantName, libraryHash: hash) {
                allCachedResults[variantName] = cached
                allResults.append(contentsOf: cached)
            } else {
                allCachedResults[variantName] = []
            }
        }
        
        for (variantName, specsForVariant) in variantSpecs {
            print("Processing variant: \(variantName)")
            
            let cachedResults = allCachedResults[variantName] ?? []
            
            let uncachedSpecs = specsForVariant.filter { spec in
                !cachedResults.contains { result in
                    result.spec.m == spec.m && result.spec.n == spec.n && result.spec.k == spec.k &&
                    result.spec.transposeA == spec.transposeA && result.spec.transposeB == spec.transposeB &&
                    result.spec.batch == spec.batch && result.spec.bias == spec.bias && result.spec.op == spec.op
                }
            }
            
            if uncachedSpecs.isEmpty {
                if specsForVariant.isEmpty { print("  No supported specs for \(variantName).") }
                else { print("  All results for \(variantName) are cached. Skipping benchmark run.") }
                continue
            }
            
            print("  Running \(uncachedSpecs.count) uncached specs for \(variantName)...")
            
            var completedUncachedRuns = 0
            
            for spec in uncachedSpecs {
                guard let tensors = tensorCache[spec] else {
                    variantFailures.append(VariantFailure(spec: spec, backend: spec.backend, variantName: "<unknown>", errorDescription: "Could not find pre-computed tensors for spec."))
                    continue
                }

                for backend in candidateBackends(for: spec) {
                    guard let variantList = variants[backend] else { continue }
                    for variant in variantList where "\(backend.rawValue)/\(variant.name)" == variantName {
                        do {
                            print("    Running: \(variant.name) [\(backend.rawValue)] - M:\(spec.m) N:\(spec.n) K:\(spec.k) | \(completedUncachedRuns + 1)/\(uncachedSpecs.count)", terminator: "\r")
                            let result = try runVariant(spec: spec, backend: backend, variant: variant, tensors: tensors)
                            allResults.append(result)
                            completedUncachedRuns += 1
                        } catch {
                            variantFailures.append(VariantFailure(spec: spec, backend: backend, variantName: variant.name, errorDescription: String(describing: error)))
                            completedUncachedRuns += 1
                        }
                    }
                }
            }
            
            print("")
            
            let resultsForVariant = allResults.filter { "\($0.backend.rawValue)/\($0.variantName)" == variantName }
            let hash = variantHashes[variantName] ?? nil
            saveCachedResults(resultsForVariant, for: variantName, libraryHash: hash)
            
            print("  Saved results for \(variantName). Total cached: \(resultsForVariant.count)")
        }
        
        print("Benchmark completed. Processed \(allResults.count) total results with \(variantFailures.count) failures.")
        return allResults
    }

    private func runMPS(spec: MatmulShapeSpec, variant: KernelVariant, tensors: MatmulTensors) throws -> BenchmarkResult {
        guard MPSSupportsMTLDevice(device) else { throw HarnessError.metalUnavailable }

        let alpha = Double(spec.alpha)
        let beta = Double(spec.beta)

        let aStride = tensors.aLayout.cols * MemoryLayout<Float16>.stride
        let bStride = tensors.bLayout.cols * MemoryLayout<Float16>.stride
        let outStride = spec.n * MemoryLayout<Float16>.stride

        let aDescriptor: MPSMatrixDescriptor
        let bDescriptor: MPSMatrixDescriptor
        let outDescriptor: MPSMatrixDescriptor

        let batchCount = max(spec.batch, 1)
        let aMatrixBytes = tensors.aLayout.rows * aStride
        let bMatrixBytes = tensors.bLayout.rows * bStride
        let outMatrixBytes = spec.m * outStride

        if batchCount > 1 {
            aDescriptor = MPSMatrixDescriptor(rows: tensors.aLayout.rows, columns: tensors.aLayout.cols, matrices: batchCount, rowBytes: aStride, matrixBytes: aMatrixBytes, dataType: .float16)
            bDescriptor = MPSMatrixDescriptor(rows: tensors.bLayout.rows, columns: tensors.bLayout.cols, matrices: batchCount, rowBytes: bStride, matrixBytes: bMatrixBytes, dataType: .float16)
            outDescriptor = MPSMatrixDescriptor(rows: spec.m, columns: spec.n, matrices: batchCount, rowBytes: outStride, matrixBytes: outMatrixBytes, dataType: .float16)
        } else {
            aDescriptor = MPSMatrixDescriptor(rows: tensors.aLayout.rows, columns: tensors.aLayout.cols, rowBytes: aStride, dataType: .float16)
            bDescriptor = MPSMatrixDescriptor(rows: tensors.bLayout.rows, columns: tensors.bLayout.cols, rowBytes: bStride, dataType: .float16)
            outDescriptor = MPSMatrixDescriptor(rows: spec.m, columns: spec.n, rowBytes: outStride, dataType: .float16)
        }

        let op = MPSMatrixMultiplication(device: device, transposeLeft: spec.transposeA, transposeRight: spec.transposeB, resultRows: spec.m, resultColumns: spec.n, interiorColumns: spec.k, alpha: alpha, beta: beta)

        var gpuTimings: [Double] = []
        var cpuTimings: [Double] = []

        for iteration in 0..<(iterations + warmup) {
            restoreOutputBuffer(buffer: tensors.output, initial: tensors.initialOutput)

            guard let commandBuffer = commandQueue.makeCommandBuffer() else { throw HarnessError.commandQueueUnavailable }

            op.batchStart = 0
            op.batchSize = batchCount
            let aMatrix = MPSMatrix(buffer: tensors.a, offset: 0, descriptor: aDescriptor)
            let bMatrix = MPSMatrix(buffer: tensors.b, offset: 0, descriptor: bDescriptor)
            let resultMatrix = MPSMatrix(buffer: tensors.output, offset: 0, descriptor: outDescriptor)
            op.encode(commandBuffer: commandBuffer, leftMatrix: aMatrix, rightMatrix: bMatrix, resultMatrix: resultMatrix)
            
            let cpuStart = DispatchTime.now().uptimeNanoseconds
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()

            if iteration >= warmup {
                gpuTimings.append(measureGPUTime(commandBuffer: commandBuffer))
                let cpuElapsed = DispatchTime.now().uptimeNanoseconds &- cpuStart
                cpuTimings.append(Double(cpuElapsed) / 1_000_000.0)
            }
        }

        let validation = validateOutput(buffer: tensors.output, reference: tensors.cpuReferenceFloat, elementCount: tensors.cpuReferenceFloat.count)

        let safeMaxAbsError = validation.maxAbsError.isFinite ? validation.maxAbsError : .greatestFiniteMagnitude
        let safeMaxRelError = validation.maxRelError.isFinite ? validation.maxRelError : .greatestFiniteMagnitude

        return BenchmarkResult(spec: spec, backend: .mps, variantName: variant.name, library: nil, isBaseline: variant.isBaseline, gpuTimingsMs: gpuTimings, cpuTimingsMs: cpuTimings, maxAbsError: safeMaxAbsError, maxRelError: safeMaxRelError)
    }

    private static func loadLibraries(device: MTLDevice, buildDir: URL) throws -> [String: MTLLibrary] {
        let fileManager = FileManager.default
        guard let contents = try? fileManager.contentsOfDirectory(at: buildDir, includingPropertiesForKeys: nil) else {
            throw HarnessError.libraryLoadFailed(buildDir.path)
        }

        var libraries: [String: MTLLibrary] = [:]
        for file in contents where file.pathExtension == "metallib" {
            do {
                let library = try device.makeLibrary(URL: file)
                let fileName = file.deletingPathExtension().lastPathComponent
                if let baseName = fileName.split(separator: ".").first {
                    libraries[String(baseName)] = library
                }
            } catch {
                throw HarnessError.libraryLoadFailed(file.path)
            }
        }
        return libraries
    }
    
    private func generateRawTensorData(spec: MatmulShapeSpec) -> CPUReferenceCache {
        var rng = LCG(seed: 0x1234_5678_9ABC_DEF0)

        let aLayout = baseLayout(rows: spec.m, cols: spec.k, transposed: spec.transposeA)
        let bLayout = baseLayout(rows: spec.k, cols: spec.n, transposed: spec.transposeB)
        let aElements = spec.batch * aLayout.rows * aLayout.cols
        let bElements = spec.batch * bLayout.rows * bLayout.cols
        let outElements = spec.batch * spec.m * spec.n

        var aValues = [Float16](repeating: 0, count: aElements)
        var bValues = [Float16](repeating: 0, count: bElements)
        var biasValues: [Float16]? = spec.bias ? [Float16](repeating: 0, count: spec.n) : nil
        var initialOutput = [Float16](repeating: 0, count: outElements)

        for i in 0..<aElements { aValues[i] = Float16(rng.nextFloatSigned(scale: 1.0)) }
        for i in 0..<bElements { bValues[i] = Float16(rng.nextFloatSigned(scale: 1.0)) }
        if var bias = biasValues { for i in 0..<bias.count { bias[i] = Float16(rng.nextFloatSigned(scale: 1.0)) }; biasValues = bias }
        for i in 0..<initialOutput.count { initialOutput[i] = Float16(rng.nextFloatSigned(scale: 1.0)) }

        let cpuReference = cpuMatmul(spec: spec, a: aValues, b: bValues, bias: biasValues, initialOutput: initialOutput)

        return CPUReferenceCache(
            aValues: aValues,
            bValues: bValues,
            biasValues: biasValues,
            initialOutput: initialOutput,
            cpuReferenceFloat: cpuReference,
            aLayout: aLayout,
            bLayout: bLayout
        )
    }
    
    private func restoreOutputBuffer(buffer: MTLBuffer, initial: [Float16]) {
        _ = initial.withUnsafeBytes { bytes in
            memcpy(buffer.contents(), bytes.baseAddress!, bytes.count)
        }
    }
    
    private func validateOutput(buffer: MTLBuffer, reference: [Float], elementCount: Int) -> (maxAbsError: Float, maxRelError: Float) {
        let pointer = buffer.contents().bindMemory(to: Float16.self, capacity: elementCount)
        var maxAbs: Float = 0
        var maxRel: Float = 0
        for i in 0..<elementCount {
            let value = Float(pointer[i])
            let ref = reference[i]
            let absErr = abs(value - ref)
            let denom = max(1.0, abs(value), abs(ref))
            let relErr = absErr / denom
            maxAbs = max(maxAbs, absErr)
            maxRel = max(maxRel, relErr)
        }
        return (maxAbs, maxRel)
    }
    
    private func measureGPUTime(commandBuffer: MTLCommandBuffer) -> Double {
        let start = commandBuffer.gpuStartTime
        let end = commandBuffer.gpuEndTime
        if start > 0 && end > 0 && end >= start {
            return (end - start) * 1000.0
        } else {
            return 0.0
        }
    }
}
