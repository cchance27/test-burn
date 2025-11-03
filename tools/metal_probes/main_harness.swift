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

    private func runVariant(spec: MatmulShapeSpec, backend: MatmulShapeSpec.Backend, variant: KernelVariant) throws -> BenchmarkResult {
        // Use unified runner for all kernel backends, MPS as special case
        switch backend {
        case .mps:
            return try runMPS(spec: spec, variant: variant)
        default:
            // For all other backends, use the unified runner
            let runner = UnifiedBackendRunner(
                device: device,
                commandQueue: commandQueue,
                libraries: libraries,
                iterations: iterations,
                warmup: warmup
            )
            return try runner.runVariant(spec: spec, backend: backend, variant: variant)
        }
    }

    private func cacheDirectory() -> URL {
        let cacheDir = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(".cache")
        try? FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        return cacheDir
    }

    private func cacheFileURL(for variantName: String) -> URL {
        // Sanitize the variantName for filesystem compatibility (replace '/' with '_')
        let sanitized = variantName.replacingOccurrences(of: "/", with: "_")
        return cacheDirectory().appendingPathComponent("cache_\(sanitized).json")
    }

    private func loadCachedResults(for variantName: String) -> [BenchmarkResult]? {
        let cacheURL = cacheFileURL(for: variantName)
        
        guard FileManager.default.fileExists(atPath: cacheURL.path) else {
            return nil
        }
        
        do {
            let data = try Data(contentsOf: cacheURL)
            let cachedResults = try JSONDecoder().decode([BenchmarkResult].self, from: data)
            return cachedResults
        } catch {
            print("Warning: Could not load cache for \(variantName): \(error)")
            return nil
        }
    }

    private func saveCachedResults(_ results: [BenchmarkResult], for variantName: String) {
        let cacheURL = cacheFileURL(for: variantName)
        
        do {
            let data = try JSONEncoder().encode(results)
            try data.write(to: cacheURL)
        } catch {
            print("Warning: Could not save cache for \(variantName): \(error)")
        }
    }

    func run(specs: [MatmulShapeSpec]) throws -> [BenchmarkResult] {
        variantFailures.removeAll()
        
        // Load all cached results first
        var allResults: [BenchmarkResult] = []
        var allCachedResults: [String: [BenchmarkResult]] = [:]
        
        // Group specs by backend+variant to enable per-variant caching
        var variantSpecs: [String: [MatmulShapeSpec]] = [:]
        
        for spec in specs {
            let backends = candidateBackends(for: spec)
            if backends.isEmpty {
                variantFailures.append(VariantFailure(spec: spec, backend: spec.backend, variantName: "<none>", errorDescription: "no enabled backends"))
                continue
            }

            for backend in backends {
                guard let variantList = variants[backend], !variantList.isEmpty else {
                    continue
                }
                for variant in variantList {
                    let key = "\(backend.rawValue)/\(variant.name)"  // Use backend/variant as key
                    if VariantManager.variantSupports(backend: backend, variant: variant, spec: spec) {
                        variantSpecs[key, default: []].append(spec)
                    } else {
                        variantFailures.append(
                            VariantFailure(
                                spec: spec,
                                backend: backend,
                                variantName: variant.name,
                                errorDescription: "skipped: unsupported configuration"
                            )
                        )
                        // ensure key exists for bookkeeping, even if no supported specs
                        if variantSpecs[key] == nil {
                            variantSpecs[key] = []
                        }
                    }
                }
            }
        }
        
        // Load cached results for all variants first
        for variantName in variantSpecs.keys {
            if let cached = loadCachedResults(for: variantName) {
                allCachedResults[variantName] = cached
                allResults.append(contentsOf: cached)
            } else {
                allCachedResults[variantName] = []
            }
        }
        
        // Process each variant separately with caching
        for (variantName, specsForVariant) in variantSpecs {
            print("Processing variant: \(variantName)")
            
            let cachedResults = allCachedResults[variantName] ?? []
            
            // Find uncached specs for this variant
            let uncachedSpecs = specsForVariant.filter { spec in
                !cachedResults.contains { result in
                    result.spec.m == spec.m && 
                    result.spec.n == spec.n && 
                    result.spec.k == spec.k &&
                    result.spec.transposeA == spec.transposeA &&
                    result.spec.transposeB == spec.transposeB &&
                    result.spec.batch == spec.batch &&
                    result.spec.bias == spec.bias &&
                    result.spec.op == spec.op  // Added op check for completeness
                }
            }
            
            if uncachedSpecs.isEmpty {
                if specsForVariant.isEmpty {
                    print("  No supported specs for \(variantName).")
                } else {
                    print("  All results for \(variantName) are cached. Skipping benchmark run.")
                }
                continue
            }
            
            print("  Running \(uncachedSpecs.count) uncached specs for \(variantName)...")
            
            // Calculate total runs for progress tracking for this specific variant
            var totalUncachedRuns = 0
            for spec in uncachedSpecs {
                let backends = candidateBackends(for: spec)
                for backend in backends {
                    if let variantList = variants[backend] {
                        for variant in variantList where "\(backend.rawValue)/\(variant.name)" == variantName {
                            totalUncachedRuns += 1
                        }
                    }
                }
            }
            
            var completedUncachedRuns = 0
            
            // Run uncached specs
            for spec in uncachedSpecs {
                let backends = candidateBackends(for: spec)
                if backends.isEmpty {
                    variantFailures.append(VariantFailure(spec: spec, backend: spec.backend, variantName: "<none>", errorDescription: "no enabled backends"))
                    completedUncachedRuns += 1
                    continue
                }

                for backend in backends {
                    guard let variantList = variants[backend], !variantList.isEmpty else {
                        continue
                    }
                    for variant in variantList where "\(backend.rawValue)/\(variant.name)" == variantName {
                        do {
                            print("    Running: \(variant.name) [\(backend.rawValue)] - M:\(spec.m) N:\(spec.n) K:\(spec.k) | \(completedUncachedRuns + 1)/\(totalUncachedRuns)", terminator: "\r")
                            let result = try runVariant(spec: spec, backend: backend, variant: variant)
                            allResults.append(result)  // Only add new results 
                            completedUncachedRuns += 1
                        } catch {
                            variantFailures.append(
                                VariantFailure(
                                    spec: spec,
                                    backend: backend,
                                    variantName: variant.name,
                                    errorDescription: String(describing: error)
                                )
                            )
                            completedUncachedRuns += 1
                        }
                    }
                }
            }
            
            print("") // New line after progress indicator for this variant
            
            // Save all results (new + cached) for this variant
            let resultsForVariant = allResults.filter { "\($0.backend.rawValue)/\($0.variantName)" == variantName }
            saveCachedResults(resultsForVariant, for: variantName)
            
            print("  Saved results for \(variantName). Total cached: \(resultsForVariant.count)")
        }
        
        print("Benchmark completed. Processed \(allResults.count) total results with \(variantFailures.count) failures.")
        return allResults
    }

    private func runMPS(spec: MatmulShapeSpec, variant: KernelVariant) throws -> BenchmarkResult {
        let tensors = loadMatmulTensors(spec: spec)
        guard MPSSupportsMTLDevice(device) else {
            throw HarnessError.metalUnavailable
        }

        let alpha = Double(spec.alpha)
        let beta = Double(spec.beta)

        let aStride = tensors.aLayout.cols * MemoryLayout<Float16>.stride
        let bStride = tensors.bLayout.cols * MemoryLayout<Float16>.stride
        let outStride = spec.n * MemoryLayout<Float16>.stride

        let aDescriptor = MPSMatrixDescriptor(
            rows: tensors.aLayout.rows,
            columns: tensors.aLayout.cols,
            rowBytes: aStride,
            dataType: .float16
        )
        let bDescriptor = MPSMatrixDescriptor(
            rows: tensors.bLayout.rows,
            columns: tensors.bLayout.cols,
            rowBytes: bStride,
            dataType: .float16
        )
        let outDescriptor = MPSMatrixDescriptor(
            rows: spec.m,
            columns: spec.n,
            rowBytes: outStride,
            dataType: .float16
        )

        let op = MPSMatrixMultiplication(
            device: device,
            transposeLeft: spec.transposeA,
            transposeRight: spec.transposeB,
            resultRows: spec.m,
            resultColumns: spec.n,
            interiorColumns: spec.k,
            alpha: alpha,
            beta: beta
        )

        let totalIterations = max(iterations, warmup + 1)
        var gpuTimings: [Double] = []
        var cpuTimings: [Double] = []

        for iteration in 0..<totalIterations {
            restoreOutputBuffer(buffer: tensors.output, initial: tensors.initialOutput)

            guard let commandBuffer = commandQueue.makeCommandBuffer() else {
                throw HarnessError.commandQueueUnavailable
            }

            let aMatrix = MPSMatrix(buffer: tensors.a, descriptor: aDescriptor)
            let bMatrix = MPSMatrix(buffer: tensors.b, descriptor: bDescriptor)
            let resultMatrix = MPSMatrix(buffer: tensors.output, descriptor: outDescriptor)

            op.encode(commandBuffer: commandBuffer, leftMatrix: aMatrix, rightMatrix: bMatrix, resultMatrix: resultMatrix)
            let cpuStart = DispatchTime.now().uptimeNanoseconds
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()

            if iteration >= warmup {
                let duration = measureGPUTime(commandBuffer: commandBuffer)
                gpuTimings.append(duration)
                let cpuElapsed = DispatchTime.now().uptimeNanoseconds &- cpuStart
                cpuTimings.append(Double(cpuElapsed) / 1_000_000.0)
            }
        }

        let validation = validateOutput(
            buffer: tensors.output,
            reference: tensors.cpuReferenceFloat,
            elementCount: tensors.cpuReferenceFloat.count
        )

        // Handle infinity and NaN values for JSON encoding
        let safeMaxAbsError = validation.maxAbsError.isFinite ? validation.maxAbsError : Float.greatestFiniteMagnitude
        let safeMaxRelError = validation.maxRelError.isFinite ? validation.maxRelError : Float.greatestFiniteMagnitude

        return BenchmarkResult(
            spec: spec,
            backend: .mps,
            variantName: variant.name,
            library: nil,
            isBaseline: variant.isBaseline,
            gpuTimingsMs: gpuTimings,
            cpuTimingsMs: cpuTimings,
            maxAbsError: safeMaxAbsError,
            maxRelError: safeMaxRelError
        )
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
                let basename = file.deletingPathExtension().lastPathComponent
                libraries[basename] = library
            } catch {
                throw HarnessError.libraryLoadFailed(file.path)
            }
        }
        return libraries
    }
    
    private func loadMatmulTensors(spec: MatmulShapeSpec) -> MatmulTensors {
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

        for i in 0..<aElements {
            aValues[i] = Float16(rng.nextFloatSigned(scale: 1.0))
        }
        for i in 0..<bElements {
            bValues[i] = Float16(rng.nextFloatSigned(scale: 1.0))
        }
        if var bias = biasValues {
            for i in 0..<bias.count {
                bias[i] = Float16(rng.nextFloatSigned(scale: 1.0))
            }
            biasValues = bias
        }
        for i in 0..<initialOutput.count {
            initialOutput[i] = Float16(rng.nextFloatSigned(scale: 1.0))
        }

        let cpuReference = cpuMatmul(
            spec: spec,
            a: aValues,
            b: bValues,
            bias: biasValues,
            initialOutput: initialOutput
        )

        let aBuffer = device.makeBuffer(array: aValues)
        let bBuffer = device.makeBuffer(array: bValues)
        let biasBuffer = biasValues.map { device.makeBuffer(array: $0) }
        let outputBuffer = device.makeBuffer(array: initialOutput)

        return MatmulTensors(
            a: aBuffer,
            b: bBuffer,
            bias: biasBuffer,
            output: outputBuffer,
            cpuReferenceFloat: cpuReference,
            initialOutput: initialOutput,
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
