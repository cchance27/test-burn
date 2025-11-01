import Foundation
import Metal
import MetalPerformanceShaders

struct MatmulShapeSpec: Hashable {
    enum Backend: String {
        case mlx
        case mps
        case gemv
        case gemmTiled = "gemm_tiled"
    }

    let op: String
    let backend: Backend
    let batch: Int
    let m: Int
    let n: Int
    let k: Int
    let transposeA: Bool
    let transposeB: Bool
    let stridedBatch: Bool
    let accumulate: Bool
    let alpha: Float
    let beta: Float
    let bias: Bool
}

struct BenchmarkResult {
    let spec: MatmulShapeSpec
    let timingsMs: [Double]
    let maxAbsError: Float
    let maxRelError: Float
}

enum HarnessError: Error, CustomStringConvertible {
    case usage
    case metalUnavailable
    case commandQueueUnavailable
    case libraryLoadFailed(String)
    case functionMissing(String)
    case pipelineCreationFailed(String)
    case unsupportedBackend(String)
    case parseFailure(String)

    var description: String {
        switch self {
        case .usage:
            return "usage: run_matmul_probes.swift <build_dir> <matmul_dir> <sizes_markdown>"
        case .metalUnavailable:
            return "Metal device unavailable"
        case .commandQueueUnavailable:
            return "Failed to create Metal command queue"
        case .libraryLoadFailed(let path):
            return "Failed to load metallib at \(path)"
        case .functionMissing(let name):
            return "Metal function not found: \(name)"
        case .pipelineCreationFailed(let fn):
            return "Failed to create pipeline for \(fn)"
        case .unsupportedBackend(let name):
            return "Unsupported backend: \(name)"
        case .parseFailure(let msg):
            return "Failed to parse MATMUL_QWEN25_SIZES.md: \(msg)"
        }
    }
}

struct LCG {
    private var state: UInt64

    init(seed: UInt64) {
        precondition(seed != 0, "Seed must be non-zero")
        state = seed
    }

    mutating func nextUInt32() -> UInt32 {
        state &*= 6364136223846793005
        state &+= 1
        return UInt32(truncatingIfNeeded: state >> 32)
    }

    mutating func nextFloatSigned(scale: Float = 1.0) -> Float {
        let maxValue = Float(UInt32.max)
        let normalized = Float(nextUInt32()) / maxValue
        return (normalized * 2.0 - 1.0) * scale
    }
}

struct MatmulDocParser {
    static func parseSpecs(markdownPath: String) throws -> [MatmulShapeSpec] {
        let url = URL(fileURLWithPath: markdownPath)
        let contents = try String(contentsOf: url, encoding: .utf8)
        let lines = contents.components(separatedBy: .newlines)
        var specs: [MatmulShapeSpec] = []
        var seen: Set<MatmulShapeSpec> = []

        var index = 0
        while index < lines.count {
            let line = lines[index]
            guard line.contains("op=") else {
                index += 1
                continue
            }

            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard trimmed.hasPrefix("op=") else {
                index += 1
                continue
            }

            do {
                let spec = try parseShapeHeader(line: trimmed, followingLines: Array(lines.dropFirst(index + 1)))
                if !seen.contains(spec) {
                    specs.append(spec)
                    seen.insert(spec)
                }
            } catch {
                throw HarnessError.parseFailure("\(error)")
            }

            index += 1
        }

        if specs.isEmpty {
            throw HarnessError.parseFailure("no shape specifications discovered")
        }

        return specs
    }

    private static func parseShapeHeader(line: String, followingLines: [String]) throws -> MatmulShapeSpec {
        var entries: [String: String] = [:]
        let tokens = line.split(separator: "|")
        for rawToken in tokens {
            let token = rawToken.trimmingCharacters(in: .whitespaces)
            guard !token.isEmpty else { continue }
            let pair = token.split(separator: "=", maxSplits: 1, omittingEmptySubsequences: false)
            guard pair.count == 2 else { continue }
            let key = String(pair[0]).trimmingCharacters(in: .whitespaces)
            var value = String(pair[1]).trimmingCharacters(in: .whitespaces)
            if value.hasSuffix(":") {
                value.removeLast()
            }
            entries[key] = value
        }

        guard let op = entries["op"] else {
            throw HarnessError.parseFailure("missing op in line: \(line)")
        }

        let backend = try parseBackend(from: followingLines)

        let batch = Int(entries["batch"] ?? "1") ?? 1
        let m = Int(entries["m"] ?? "0") ?? 0
        let n = Int(entries["n"] ?? "0") ?? 0
        let k = Int(entries["k"] ?? "0") ?? 0
        let transposeA = (entries["tA"] ?? "0") != "0"
        let transposeB = (entries["tB"] ?? "0") != "0"
        let stridedBatch = (entries["strided_batch"] ?? "false").lowercased() == "true"
        let accumulate = (entries["accumulate"] ?? "0") != "0"
        let alpha = Float(entries["alpha"] ?? "1") ?? 1.0
        let beta = Float(entries["beta"] ?? "0") ?? 0.0
        let bias = (entries["bias"] ?? "0") != "0"

        guard m > 0, n > 0, k > 0 else {
            throw HarnessError.parseFailure("invalid dimensions in line: \(line)")
        }

        return MatmulShapeSpec(
            op: op,
            backend: backend,
            batch: batch,
            m: m,
            n: n,
            k: k,
            transposeA: transposeA,
            transposeB: transposeB,
            stridedBatch: stridedBatch,
            accumulate: accumulate,
            alpha: alpha,
            beta: beta,
            bias: bias
        )
    }

    static func determineOptimalBackend(for spec: MatmulShapeSpec) -> MatmulShapeSpec.Backend {
        // Route to specific backends based on shape characteristics
        if spec.n <= 16 && spec.k >= 64 {
            // Small-N case is ideal for GEMV kernels
            return .gemv
        } else if spec.m <= 16 || spec.n <= 16 {
            // Small dimensions are good for specialized kernels
            return .gemv
        } else if spec.m >= 64 && spec.n >= 64 && spec.k >= 64 {
            // Large dimensions may benefit from tiled approach
            return .gemmTiled
        } else {
            // Default to original backend for other cases
            return spec.backend
        }
    }
    
    private static func parseBackend(from lines: [String]) throws -> MatmulShapeSpec.Backend {
        for rawLine in lines {
            let trimmed = rawLine.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty {
                break
            }
            guard let backendRange = trimmed.range(of: "backend=") else {
                continue
            }
            let backendPart = trimmed[backendRange.upperBound...]
            let endIndex = backendPart.firstIndex(of: ":") ?? backendPart.endIndex
            let name = backendPart[..<endIndex]
            if let backend = MatmulShapeSpec.Backend(rawValue: String(name)) {
                return backend
            }
            throw HarnessError.unsupportedBackend(String(name))
        }
        throw HarnessError.parseFailure("backend not found for shape block")
    }
}

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

struct MatmulTensors {
    let a: MTLBuffer
    let b: MTLBuffer
    let bias: MTLBuffer?
    let output: MTLBuffer
    let cpuReferenceFloat: [Float]
    let initialOutput: [Float16]
    let aLayout: (rows: Int, cols: Int)
    let bLayout: (rows: Int, cols: Int)
}

struct MLXFunctionConstants {
    var hasBatch = false
    var useOutSource = false
    var doAxpby = false
    var doBiasAdd = false
    var alignM = false
    var alignN = false
    var alignK = false
}

struct GEMMParams {
    var m: Int32
    var n: Int32
    var k: Int32
    var lda: Int32
    var ldb: Int32
    var ldd: Int32
    var tilesN: Int32
    var tilesM: Int32
    var batchStrideA: Int
    var batchStrideB: Int
    var batchStrideD: Int
    var swizzleLog: Int32
    var gemmKIterationsAligned: Int32
    var batchNDIM: Int32
}

struct GEMMAddMMParams {
    var ldc: Int32
    var fdc: Int32
    var batchStrideC: UInt
    var alpha: Float
    var beta: Float
}

final class MatmulHarness {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let libraries: [String: MTLLibrary]
    private let buildDir: URL
    private let matmulDir: URL
    private let iterations: Int
    private let warmup: Int

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
    }

    func run(specs: [MatmulShapeSpec]) throws -> [BenchmarkResult] {
        var results: [BenchmarkResult] = []
        for spec in specs {
            switch spec.backend {
            case .mlx:
                let result = try runMLX(spec: spec)
                results.append(result)
            case .mps:
                let result = try runMPS(spec: spec)
                results.append(result)
            case .gemv:
                let result = try runGEMV(spec: spec)
                results.append(result)
            case .gemmTiled:
                let result = try runGEMMTiled(spec: spec)
                results.append(result)
            }
        }
        return results
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

    private func runMLX(spec: MatmulShapeSpec) throws -> BenchmarkResult {
        guard let library = libraries["original_mlx"] else {
            throw HarnessError.libraryLoadFailed("original_mlx.metallib")
        }

        let tensors = loadMatmulTensors(spec: spec)
        let alpha = spec.alpha
        let beta = spec.beta

        let (tileBM, tileBN, tileBK, wmSel, wnSel) = selectTile(m: spec.m, n: spec.n)
        let functionName = makeMLXFunctionName(spec: spec, tile: (tileBM, tileBN, tileBK, wmSel, wnSel))
        let constants = makeMLXConstants(
            spec: spec,
            tile: (tileBM, tileBN, tileBK),
            alpha: alpha,
            beta: beta
        )

        let functionConstants = MTLFunctionConstantValues()
        var hasBatch = constants.hasBatch
        var useOutSource = constants.useOutSource
        var doAxpby = constants.doAxpby
        var doBiasAdd = constants.doBiasAdd
        var alignM = constants.alignM
        var alignN = constants.alignN
        var alignK = constants.alignK
        var gatherBias = false

        functionConstants.setConstantValue(&hasBatch, type: .bool, index: 10)
        functionConstants.setConstantValue(&useOutSource, type: .bool, index: 100)
        functionConstants.setConstantValue(&doAxpby, type: .bool, index: 110)
        functionConstants.setConstantValue(&doBiasAdd, type: .bool, index: 120)
        functionConstants.setConstantValue(&alignM, type: .bool, index: 200)
        functionConstants.setConstantValue(&alignN, type: .bool, index: 201)
        functionConstants.setConstantValue(&alignK, type: .bool, index: 202)
        functionConstants.setConstantValue(&gatherBias, type: .bool, index: 300)

        guard let function = try? library.makeFunction(name: functionName, constantValues: functionConstants) else {
            throw HarnessError.functionMissing(functionName)
        }
        guard let pipeline = try? device.makeComputePipelineState(function: function) else {
            throw HarnessError.pipelineCreationFailed(functionName)
        }

        let threadExecutionWidth = pipeline.threadExecutionWidth
        if threadExecutionWidth != 32 {
            // MLX kernels are built around simdgroup size 32.
            print("Warning: unexpected threadExecutionWidth=\(threadExecutionWidth) for \(functionName)")
        }

        let params = buildGEMMParams(
            spec: spec,
            tile: (tileBM, tileBN, tileBK),
            constants: constants
        )

        let addmmParams = constants.useOutSource ? GEMMAddMMParams(
            ldc: params.ldd,
            fdc: determineMinorStride(m: spec.m),
            batchStrideC: UInt(spec.m * spec.n),
            alpha: alpha,
            beta: beta
        ) : nil

        let batchShape: Int32 = constants.hasBatch ? Int32(clamping: spec.batch) : 0
        let batchStrides = makeBatchStrides(
            spec: spec,
            aLayout: tensors.aLayout,
            bLayout: tensors.bLayout,
            useOutSource: constants.useOutSource
        )

        let threadgroups = makeThreadgroups(spec: spec, tile: (tileBM, tileBN), constants: constants)
        let threadsPerTG = MTLSize(width: 32, height: wnSel, depth: wmSel)

        var gpuTimings: [Double] = []
        let totalIterations = max(iterations, warmup + 1)

        for iteration in 0..<totalIterations {
            restoreOutputBuffer(buffer: tensors.output, initial: tensors.initialOutput)

            guard let commandBuffer = commandQueue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw HarnessError.commandQueueUnavailable
            }

            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(tensors.a, offset: 0, index: 0)
            encoder.setBuffer(tensors.b, offset: 0, index: 1)
            if constants.useOutSource {
                encoder.setBuffer(tensors.output, offset: 0, index: 2)
            }
            encoder.setBuffer(tensors.output, offset: 0, index: 3)

            if let biasBuffer = tensors.bias {
                encoder.setBuffer(biasBuffer, offset: 0, index: 4)
                var paramsCopy = params
                encoder.setBytes(&paramsCopy, length: MemoryLayout<GEMMParams>.stride, index: 5)
                if var addmm = addmmParams {
                    encoder.setBytes(&addmm, length: MemoryLayout<GEMMAddMMParams>.stride, index: 6)
                }
                var batchShapeCopy = batchShape
                encoder.setBytes(&batchShapeCopy, length: MemoryLayout.size(ofValue: batchShapeCopy), index: 7)
                var batchStridesCopy = batchStrides
                encoder.setBytes(&batchStridesCopy, length: MemoryLayout.size(ofValue: batchStridesCopy), index: 8)
            } else {
                var paramsCopy = params
                encoder.setBytes(&paramsCopy, length: MemoryLayout<GEMMParams>.stride, index: 4)
                if var addmm = addmmParams {
                    encoder.setBytes(&addmm, length: MemoryLayout<GEMMAddMMParams>.stride, index: 5)
                }
                var batchShapeCopy = batchShape
                encoder.setBytes(&batchShapeCopy, length: MemoryLayout.size(ofValue: batchShapeCopy), index: 6)
                var batchStridesCopy = batchStrides
                encoder.setBytes(&batchStridesCopy, length: MemoryLayout.size(ofValue: batchStridesCopy), index: 7)
            }

            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerTG)
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()

            if iteration >= warmup {
                let duration = measureGPUTime(commandBuffer: commandBuffer)
                gpuTimings.append(duration)
            }
        }

        let validation = validateOutput(
            buffer: tensors.output,
            reference: tensors.cpuReferenceFloat,
            elementCount: tensors.cpuReferenceFloat.count
        )

        return BenchmarkResult(
            spec: spec,
            timingsMs: gpuTimings,
            maxAbsError: validation.maxAbsError,
            maxRelError: validation.maxRelError
        )
    }

    private func runMPS(spec: MatmulShapeSpec) throws -> BenchmarkResult {
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

        for iteration in 0..<totalIterations {
            restoreOutputBuffer(buffer: tensors.output, initial: tensors.initialOutput)

            guard let commandBuffer = commandQueue.makeCommandBuffer() else {
                throw HarnessError.commandQueueUnavailable
            }

            let aMatrix = MPSMatrix(buffer: tensors.a, descriptor: aDescriptor)
            let bMatrix = MPSMatrix(buffer: tensors.b, descriptor: bDescriptor)
            let resultMatrix = MPSMatrix(buffer: tensors.output, descriptor: outDescriptor)

            op.encode(commandBuffer: commandBuffer, leftMatrix: aMatrix, rightMatrix: bMatrix, resultMatrix: resultMatrix)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()

            if iteration >= warmup {
                let duration = measureGPUTime(commandBuffer: commandBuffer)
                gpuTimings.append(duration)
            }
        }

        let validation = validateOutput(
            buffer: tensors.output,
            reference: tensors.cpuReferenceFloat,
            elementCount: tensors.cpuReferenceFloat.count
        )

        return BenchmarkResult(
            spec: spec,
            timingsMs: gpuTimings,
            maxAbsError: validation.maxAbsError,
            maxRelError: validation.maxRelError
        )
    }

    private func runGEMV(spec: MatmulShapeSpec) throws -> BenchmarkResult {
        guard let library = libraries["original_gemv"] else {
            throw HarnessError.libraryLoadFailed("original_gemv.metallib")
        }

        let tensors = loadMatmulTensors(spec: spec)
        
        // Determine function name based on data type and shape
        let functionName: String
        if spec.n == 1 {
            functionName = "gemv_n1_f16"
        } else if spec.n == 2 {
            functionName = "gemv_n2_f16"
        } else if spec.n == 4 {
            functionName = "gemv_n4_f16"
        } else if spec.n == 8 {
            functionName = "gemv_n8_f16"
        } else if spec.n == 16 {
            functionName = "gemv_n16_f16"
        } else {
            // For larger N values, use the general GEMV kernel
            functionName = spec.n <= 8 ? "gemv_f16" : "gemv_f16"
        }

        guard let function = library.makeFunction(name: functionName) else {
            throw HarnessError.functionMissing(functionName)
        }
        guard let pipeline = try? device.makeComputePipelineState(function: function) else {
            throw HarnessError.pipelineCreationFailed(functionName)
        }

        // Note: GEMV assumes A is M x K, B is K x N, and we compute A * B
        // So the dimensions need to match as: (M, K) * (K, N) -> (M, N)
        var m = UInt32(spec.m)
        var k = UInt32(spec.k)
        
        var gpuTimings: [Double] = []
        let totalIterations = max(iterations, warmup + 1)

        for iteration in 0..<totalIterations {
            restoreOutputBuffer(buffer: tensors.output, initial: tensors.initialOutput)

            guard let commandBuffer = commandQueue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw HarnessError.commandQueueUnavailable
            }

            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(tensors.a, offset: 0, index: 0)
            encoder.setBuffer(tensors.b, offset: 0, index: 1)
            encoder.setBuffer(tensors.output, offset: 0, index: 2)
            
            if spec.n <= 16 && [1, 2, 4, 8, 16].contains(spec.n) {
                // For small N GEMV kernels with M, K as constants
                encoder.setBytes(&m, length: MemoryLayout.size(ofValue: m), index: 3)
                encoder.setBytes(&k, length: MemoryLayout.size(ofValue: k), index: 4)
                
                // Calculate threadgroup size based on N dimension
                let threadsPerThreadgroup: MTLSize
                let tgPerGrid: MTLSize
                if spec.n == 1 {
                    threadsPerThreadgroup = MTLSize(width: 32, height: 1, depth: 1) // 32 threads
                    tgPerGrid = MTLSize(width: (spec.m + 31) / 32, height: 1, depth: 1)
                } else if spec.n == 2 {
                    threadsPerThreadgroup = MTLSize(width: 32, height: 1, depth: 1)
                    tgPerGrid = MTLSize(width: (spec.m + 31) / 32, height: 1, depth: 1)
                } else if spec.n == 4 {
                    threadsPerThreadgroup = MTLSize(width: 64, height: 1, depth: 1) // 16 rows x 4 cols = 64 threads
                    tgPerGrid = MTLSize(width: (spec.m + 15) / 16, height: 1, depth: 1)
                } else if spec.n == 8 {
                    threadsPerThreadgroup = MTLSize(width: 64, height: 1, depth: 1) // 8 rows x 8 cols = 64 threads
                    tgPerGrid = MTLSize(width: 1, height: (spec.m + 7) / 8, depth: 1)
                } else { // n == 16
                    threadsPerThreadgroup = MTLSize(width: 64, height: 1, depth: 1) // 4 rows x 16 cols = 64 threads
                    tgPerGrid = MTLSize(width: (spec.m + 3) / 4, height: 1, depth: 1)
                }
                
                encoder.dispatchThreadgroups(tgPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            } else {
                // For general GEMV kernel using gemv_f16
                var params = GemvParams_f16(K: UInt32(spec.k), N: UInt32(spec.n))
                encoder.setBytes(&params, length: MemoryLayout<GemvParams_f16>.size, index: 3)
                
                let threadgroups = MTLSize(width: (spec.n + 255) / 256, height: 1, depth: 1)
                let threadsPerTG = MTLSize(width: 256, height: 1, depth: 1)
                encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerTG)
            }

            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()

            if iteration >= warmup {
                let duration = measureGPUTime(commandBuffer: commandBuffer)
                gpuTimings.append(duration)
            }
        }

        let validation = validateOutput(
            buffer: tensors.output,
            reference: tensors.cpuReferenceFloat,
            elementCount: tensors.cpuReferenceFloat.count
        )

        return BenchmarkResult(
            spec: spec,
            timingsMs: gpuTimings,
            maxAbsError: validation.maxAbsError,
            maxRelError: validation.maxRelError
        )
    }

    private func runGEMMTiled(spec: MatmulShapeSpec) throws -> BenchmarkResult {
        guard let library = libraries["original_gemm_tiled"] else {
            throw HarnessError.libraryLoadFailed("original_gemm_tiled.metallib")
        }

        let tensors = loadMatmulTensors(spec: spec)
        
        let functionName = "gemm_tiled_f16"
        guard let function = library.makeFunction(name: functionName) else {
            throw HarnessError.functionMissing(functionName)
        }
        guard let pipeline = try? device.makeComputePipelineState(function: function) else {
            throw HarnessError.pipelineCreationFailed(functionName)
        }

        // Setup parameters for the GEMM Tiled kernel
        var params = GemmTiledParams_f16(
            m: UInt32(spec.m),
            n: UInt32(spec.n),
            k: UInt32(spec.k),
            lda: UInt32(spec.transposeA ? spec.m : spec.k),  // leading dimension of A
            ldb: UInt32(spec.transposeB ? spec.k : spec.n),  // leading dimension of B
            ldc: UInt32(spec.n),  // leading dimension of C
            tile_m: UInt32(min(spec.m, 64)),  // tile size for M dimension
            tile_n: UInt32(min(spec.n, 64)),  // tile size for N dimension
            tile_k: UInt32(min(spec.k, 16)),  // tile size for K dimension
            use_simdgroup_mm: 1,  // Use SIMD group matrix multiply when available
            alpha: spec.alpha,
            beta: spec.beta
        )

        var gpuTimings: [Double] = []
        let totalIterations = max(iterations, warmup + 1)

        for iteration in 0..<totalIterations {
            restoreOutputBuffer(buffer: tensors.output, initial: tensors.initialOutput)

            guard let commandBuffer = commandQueue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw HarnessError.commandQueueUnavailable
            }

            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(tensors.a, offset: 0, index: 0)
            encoder.setBuffer(tensors.b, offset: 0, index: 1)
            encoder.setBuffer(tensors.output, offset: 0, index: 2)
            encoder.setBytes(&params, length: MemoryLayout<GemmTiledParams_f16>.size, index: 3)
            
            // Set bias buffer if needed
            if let biasBuffer = tensors.bias {
                encoder.setBuffer(biasBuffer, offset: 0, index: 4)
            }

            // Calculate optimal threadgroup and grid dimensions
            let tileM = Int(params.tile_m)
            let tileN = Int(params.tile_n)
            
            let threadgroups = MTLSize(
                width: (spec.n + tileN - 1) / tileN,
                height: (spec.m + tileM - 1) / tileM,
                depth: 1
            )
            
            // Use 64 threads per threadgroup for the tiled implementation
            let threadsPerTG = MTLSize(width: 64, height: 1, depth: 1)

            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerTG)
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()

            if iteration >= warmup {
                let duration = measureGPUTime(commandBuffer: commandBuffer)
                gpuTimings.append(duration)
            }
        }

        let validation = validateOutput(
            buffer: tensors.output,
            reference: tensors.cpuReferenceFloat,
            elementCount: tensors.cpuReferenceFloat.count
        )

        return BenchmarkResult(
            spec: spec,
            timingsMs: gpuTimings,
            maxAbsError: validation.maxAbsError,
            maxRelError: validation.maxRelError
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
}

// Define parameter structs for GEMV and GEMM Tiled kernels
struct GemvParams_f16 {
    var K: UInt32
    var N: UInt32
}

struct GemmTiledParams_f16 {
    var m: UInt32
    var n: UInt32
    var k: UInt32
    var lda: UInt32
    var ldb: UInt32
    var ldc: UInt32
    var tile_m: UInt32
    var tile_n: UInt32
    var tile_k: UInt32
    var use_simdgroup_mm: UInt32
    var alpha: Float
    var beta: Float
}

func measureGPUTime(commandBuffer: MTLCommandBuffer) -> Double {
    let start = commandBuffer.gpuStartTime
    let end = commandBuffer.gpuEndTime
    if start > 0 && end > 0 && end >= start {
        return (end - start) * 1000.0
    } else {
        return 0.0
    }
}

func baseLayout(rows: Int, cols: Int, transposed: Bool) -> (rows: Int, cols: Int) {
    if transposed {
        return (cols, rows)
    } else {
        return (rows, cols)
    }
}

func cpuMatmul(
    spec: MatmulShapeSpec,
    a: [Float16],
    b: [Float16],
    bias: [Float16]?,
    initialOutput: [Float16]
) -> [Float] {
    let batch = max(spec.batch, 1)
    let aLayout = baseLayout(rows: spec.m, cols: spec.k, transposed: spec.transposeA)
    let bLayout = baseLayout(rows: spec.k, cols: spec.n, transposed: spec.transposeB)

    func indexA(batch: Int, row: Int, col: Int) -> Int {
        let baseRows = aLayout.rows
        let baseCols = aLayout.cols

        let actualRow = spec.transposeA ? col : row
        let actualCol = spec.transposeA ? row : col

        return batch * baseRows * baseCols + actualRow * baseCols + actualCol
    }

    func indexB(batch: Int, row: Int, col: Int) -> Int {
        let baseRows = bLayout.rows
        let baseCols = bLayout.cols

        let actualRow = spec.transposeB ? col : row
        let actualCol = spec.transposeB ? row : col

        return batch * baseRows * baseCols + actualRow * baseCols + actualCol
    }

    var reference = [Float](repeating: 0, count: batch * spec.m * spec.n)
    for bIndex in 0..<batch {
        for row in 0..<spec.m {
            for col in 0..<spec.n {
                var acc: Float = 0
                for kk in 0..<spec.k {
                    let aVal = Float(a[indexA(batch: bIndex, row: row, col: kk)])
                    let bVal = Float(b[indexB(batch: bIndex, row: kk, col: col)])
                    acc += aVal * bVal
                }

                if let bias = bias {
                    acc += Float(bias[col])
                }

                let outIndex = bIndex * spec.m * spec.n + row * spec.n + col
                let initial = Float(initialOutput[outIndex])
                let alpha = spec.alpha
                let beta = spec.beta
                let result = alpha * acc + beta * initial
                reference[outIndex] = result
            }
        }
    }
    return reference
}

func selectTile(m: Int, n: Int) -> (Int, Int, Int, Int, Int) {
    if m == 1 {
        return (8, 128, 32, 1, 4)
    } else if m <= 16 && n >= 64 {
        return (16, 64, 16, 1, 4)
    } else if n <= 16 && m >= 64 {
        return (64, 16, 16, 4, 1)
    } else {
        return (32, 32, 16, 2, 2)
    }
}

func makeMLXFunctionName(spec: MatmulShapeSpec, tile: (Int, Int, Int, Int, Int)) -> String {
    let dtype = "f16"
    let aTag = spec.transposeA ? "t" : "n"
    let bTag = spec.transposeB ? "t" : "n"
    let kernel = spec.bias ? "gemm_bias" : "gemm"
    return "\(kernel)_\(aTag)\(bTag)_\(dtype)_\(dtype)_\(tile.0)_\(tile.1)_\(tile.2)_\(tile.3)_\(tile.4)"
}

func makeMLXConstants(
    spec: MatmulShapeSpec,
    tile: (Int, Int, Int),
    alpha: Float,
    beta: Float
) -> MLXFunctionConstants {
    let alignM = spec.m % tile.0 == 0
    let alignN = spec.n % tile.1 == 0
    let alignK = spec.k % tile.2 == 0
    let hasBatch = spec.batch > 1
    let requiresEpilogue = abs(alpha - 1.0) > .ulpOfOne || abs(beta) > .ulpOfOne
    return MLXFunctionConstants(
        hasBatch: hasBatch,
        useOutSource: requiresEpilogue,
        doAxpby: requiresEpilogue,
        doBiasAdd: spec.bias,
        alignM: alignM,
        alignN: alignN,
        alignK: alignK
    )
}

func buildGEMMParams(
    spec: MatmulShapeSpec,
    tile: (Int, Int, Int),
    constants: MLXFunctionConstants
) -> GEMMParams {
    let tnBase = Int(ceil(Double(spec.n) / Double(tile.1)))
    let tmBase = Int(ceil(Double(spec.m) / Double(tile.0)))
    let swizzleLog = (tmBase >= 8 && tnBase >= 8) ? 1 : 0
    let tilePow = 1 << swizzleLog
    let tn = tnBase * tilePow
    let tm = Int(ceil(Double(tmBase) / Double(tilePow)))

    let lda = Int32(spec.transposeA ? spec.m : spec.k)
    let ldb = Int32(spec.transposeB ? spec.k : spec.n)
    let ldd = Int32(spec.n)
    let aLayout = baseLayout(rows: spec.m, cols: spec.k, transposed: spec.transposeA)
    let bLayout = baseLayout(rows: spec.k, cols: spec.n, transposed: spec.transposeB)
    let batchStrideA = constants.hasBatch ? aLayout.rows * aLayout.cols : 0
    let batchStrideB = constants.hasBatch ? bLayout.rows * bLayout.cols : 0
    let batchStrideD = (constants.hasBatch ? spec.m * spec.n : 0)

    return GEMMParams(
        m: Int32(spec.m),
        n: Int32(spec.n),
        k: Int32(spec.k),
        lda: lda,
        ldb: ldb,
        ldd: ldd,
        tilesN: Int32(tn),
        tilesM: Int32(tm),
        batchStrideA: batchStrideA,
        batchStrideB: batchStrideB,
        batchStrideD: batchStrideD,
        swizzleLog: Int32(swizzleLog),
        gemmKIterationsAligned: Int32(spec.k / tile.2),
        batchNDIM: constants.hasBatch ? 1 : 0
    )
}

func determineMinorStride(m: Int) -> Int32 {
    return 1
}

func makeBatchStrides(
    spec: MatmulShapeSpec,
    aLayout: (rows: Int, cols: Int),
    bLayout: (rows: Int, cols: Int),
    useOutSource: Bool
) -> (UInt, UInt, UInt) {
    if spec.batch <= 1 {
        return (0, 0, 0)
    }

    let elementSize = MemoryLayout<Float16>.stride
    let strideA = UInt(aLayout.rows * aLayout.cols * elementSize)
    let strideB = UInt(bLayout.rows * bLayout.cols * elementSize)
    let strideC = useOutSource ? UInt(spec.m * spec.n * elementSize) : 0
    return (strideA, strideB, strideC)
}

func makeThreadgroups(
    spec: MatmulShapeSpec,
    tile: (Int, Int),
    constants: MLXFunctionConstants
) -> MTLSize {
    let tnBase = Int(ceil(Double(spec.n) / Double(tile.1)))
    let tmBase = Int(ceil(Double(spec.m) / Double(tile.0)))
    let swizzleLog = (tmBase >= 8 && tnBase >= 8) ? 1 : 0
    let tilePow = 1 << swizzleLog
    let tn = tnBase * tilePow
    let tm = Int(ceil(Double(tmBase) / Double(tilePow)))
    return MTLSize(width: tn, height: tm, depth: max(spec.batch, 1))
}

func restoreOutputBuffer(buffer: MTLBuffer, initial: [Float16]) {
    initial.withUnsafeBytes { bytes in
        memcpy(buffer.contents(), bytes.baseAddress!, bytes.count)
    }
}

func validateOutput(buffer: MTLBuffer, reference: [Float], elementCount: Int) -> (maxAbsError: Float, maxRelError: Float) {
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

// Structure to hold original profiling data
struct OriginalProfileData {
    var avgTimeMs: Double = 0.0
    var minTimeMs: Double = 0.0
    var maxTimeMs: Double = 0.0
    var count: Int = 0
}

// Global variable to hold the original profiling data
var originalProfilingData: [String: OriginalProfileData] = [:]

func summarize(results: [BenchmarkResult]) {
    // Load original profiling data from markdown file
    guard let markdownPath = CommandLine.arguments.last else {
        print("Error: No markdown path provided")
        return
    }
    
    do {
        try loadOriginalProfilingData(markdownPath: markdownPath)
    } catch {
        print("Warning: Could not load original profiling data: \(error)")
    }
    
    guard !results.isEmpty else {
        print("No benchmark results to display.")
        return
    }

    print("Matmul benchmark summary (iterations=\(results.first?.timingsMs.count ?? 0))")
    for result in results {
        let spec = result.spec
        let avg = result.timingsMs.isEmpty ? 0.0 : result.timingsMs.reduce(0, +) / Double(result.timingsMs.count)
        let maxAbs = result.maxAbsError
        let maxRel = result.maxRelError
        
        // Create a key to match against original profiling data
        let key = makeProfileKey(spec: spec)
        let originalData = originalProfilingData[key] ?? OriginalProfileData()
        
        let tag = "\(spec.backend.rawValue) :: \(spec.op) :: batch=\(spec.batch) m=\(spec.m) n=\(spec.n) k=\(spec.k) tA=\(spec.transposeA ? 1 : 0) tB=\(spec.transposeB ? 1 : 0)"
        
        // Format to match original markdown (3 decimal places, no trailing zeros)
        let avgStr = formatTime(avg)
        let maxAbsStr = String(format: "%.4e", maxAbs)
        let maxRelStr = String(format: "%.4e", maxRel)
        
        var comparisonStr = ""
        if originalData.avgTimeMs > 0 {
            let percentChange = ((avg - originalData.avgTimeMs) / originalData.avgTimeMs) * 100.0
            let sign = percentChange >= 0 ? "+" : ""
            comparisonStr = " [vs orig \(formatTime(originalData.avgTimeMs))ms] (\(sign)\(String(format: "%.2f", percentChange))%)"
        } else {
            comparisonStr = " [vs orig N/A]"
        }
        
        print(" - \(tag): avg=\(avgStr)ms\(comparisonStr) maxAbs=\(maxAbsStr) maxRel=\(maxRelStr)")
    }
}

// Create a key to match spec against original profiling data
func makeProfileKey(spec: MatmulShapeSpec) -> String {
    return "op=\(spec.op)|batch=\(spec.batch)|m=\(spec.m)|n=\(spec.n)|k=\(spec.k)|tA=\(spec.transposeA ? 1 : 0)|tB=\(spec.transposeB ? 1 : 0)"
}

func formatTime(_ time: Double) -> String {
    // Format to 3 decimal places but remove trailing zeros after decimal point
    let formatted = String(format: "%.3f", time)
    // Remove trailing zeros but keep at least one decimal place
    var result = formatted
    while result.count > 3 && result.hasSuffix("0") && result.contains(".") {
        result = String(result.dropLast())
    }
    if result.hasSuffix(".") {
        result = String(result.dropLast()) + ".0"
    }
    return result
}

func loadOriginalProfilingData(markdownPath: String) throws {
    let url = URL(fileURLWithPath: markdownPath)
    let contents = try String(contentsOf: url, encoding: .utf8)
    let lines = contents.components(separatedBy: .newlines)
    
    var lastOpLine: String = ""
    
    for i in 0..<lines.count {
        let line = lines[i]
        let trimmedLine = line.trimmingCharacters(in: .whitespaces)
        
        // Check if this line is an "op=" line (shape configuration header)
        if trimmedLine.hasPrefix("op=") && trimmedLine.contains("|") {
            lastOpLine = trimmedLine
        }
        // Check if the next line contains backend timing data
        else if trimmedLine.contains("backend=") && trimmedLine.contains("avg=") {
            // Extract the avg timing from this line
            if let avgRange = trimmedLine.range(of: "avg="),
               let msRange = trimmedLine.range(of: "ms", range: avgRange.upperBound..<trimmedLine.endIndex) {
                let avgValueStr = String(trimmedLine[avgRange.upperBound..<msRange.lowerBound]).trimmingCharacters(in: .whitespaces)
                if let avgTime = Double(avgValueStr) {
                    // Use the last op line to create the key
                    if !lastOpLine.isEmpty {
                        // Create the key from the operation line
                        let key = extractKeyFromOpLine(lastOpLine)
                        originalProfilingData[key] = OriginalProfileData(avgTimeMs: avgTime, minTimeMs: 0, maxTimeMs: 0, count: 0)
                    }
                }
            }
        }
    }
}

func extractKeyFromOpLine(_ opLine: String) -> String {
    // Extract parameters from a line like "op=matmul_cache | batch=1 | m=1 | n=9728 | k=896 | tA=0 | tB=1 | strided_batch=false:"
    var params: [String: String] = [:]
    
    let parts = opLine.components(separatedBy: "|")
    for part in parts {
        let trimmedPart = part.trimmingCharacters(in: .whitespaces)
        if trimmedPart.contains("=") {
            let keyValue = trimmedPart.components(separatedBy: "=")
            if keyValue.count >= 2 {
                let key = keyValue[0].trimmingCharacters(in: .whitespaces)
                var value = keyValue[1].trimmingCharacters(in: .whitespaces)
                if value.hasSuffix(":") {
                    value = String(value.dropLast())
                }
                params[key] = value
            }
        }
    }
    
    // Create key in the same format as makeProfileKey
    if let op = params["op"], let batch = params["batch"], let m = params["m"], let n = params["n"], let k = params["k"], let tA = params["tA"], let tB = params["tB"] {
        return "op=\(op)|batch=\(batch)|m=\(m)|n=\(n)|k=\(k)|tA=\(tA)|tB=\(tB)"
    }
    
    return ""
}

func getOriginalPerformanceData(spec: MatmulShapeSpec) -> OriginalProfileData {
    let key = makeProfileKey(spec: spec)
    return originalProfilingData[key] ?? OriginalProfileData()
}

func main() -> Int32 {
    let args = CommandLine.arguments
    guard args.count >= 4 else {
        print(HarnessError.usage)
        return 1
    }

    let buildDir = URL(fileURLWithPath: args[1])
    let matmulDir = URL(fileURLWithPath: args[2])
    let markdownPath = args[3]

    do {
        let specs = try MatmulDocParser.parseSpecs(markdownPath: markdownPath)
        let harness = try MatmulHarness(buildDir: buildDir, matmulDir: matmulDir)
        let results = try harness.run(specs: specs)
        summarize(results: results)
    } catch {
        print("Error: \(error)")
        return 1
    }

    return 0
}

exit(main())
