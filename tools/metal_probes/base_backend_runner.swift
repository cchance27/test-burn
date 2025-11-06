import Foundation
import Metal
import MetalPerformanceShaders

protocol BackendRunner {
    func runVariant(
        spec: MatmulShapeSpec,
        backend: MatmulShapeSpec.Backend,
        variant: KernelVariant,
        tensors: MatmulTensors
    ) throws -> BenchmarkResult
    func validateVariant(variant: KernelVariant) throws
}

// Cache for compiled Metal pipelines
final class PipelineCache {
    var pipelines: [String: MTLComputePipelineState] = [:]
}

// Base class that handles common functionality
class BaseBackendRunner {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let libraries: [String: MTLLibrary]
    let iterations: Int
    let warmup: Int
    let pipelineCache: PipelineCache
    
    init(device: MTLDevice, commandQueue: MTLCommandQueue, libraries: [String: MTLLibrary], iterations: Int, warmup: Int, pipelineCache: PipelineCache) {
        self.device = device
        self.commandQueue = commandQueue
        self.libraries = libraries
        self.iterations = iterations
        self.warmup = warmup
        self.pipelineCache = pipelineCache
    }
    

    func restoreOutputBuffer(buffer: MTLBuffer, initial: [Float16]) {
        _ = initial.withUnsafeBytes { bytes in
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
    
    func measureGPUTime(commandBuffer: MTLCommandBuffer) -> Double {
        let start = commandBuffer.gpuStartTime
        let end = commandBuffer.gpuEndTime
        if start > 0 && end > 0 && end >= start {
            return (end - start) * 1000.0
        } else {
            return 0.0
        }
    }
}

// Common utility functions
func baseLayout(rows: Int, cols: Int, transposed: Bool) -> TensorLayout {
    if transposed {
        return TensorLayout(rows: cols, cols: rows)
    } else {
        return TensorLayout(rows: rows, cols: cols)
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
    aLayout: TensorLayout,
    bLayout: TensorLayout,
    useOutSource: Bool
) -> (UInt, UInt, UInt) {
    if spec.batch <= 1 {
        return (0, 0, 0)
    }

    // Strides are in elements, not bytes, for use with typed pointers in Metal
    let strideA = UInt(aLayout.rows * aLayout.cols)
    let strideB = UInt(bLayout.rows * bLayout.cols)
    let strideC = useOutSource ? UInt(spec.m * spec.n) : 0
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

// Version without constants for generic usage
func makeThreadgroups(
    spec: MatmulShapeSpec,
    tile: (Int, Int)
) -> MTLSize {
    let tnBase = Int(ceil(Double(spec.n) / Double(tile.1)))
    let tmBase = Int(ceil(Double(spec.m) / Double(tile.0)))
    let swizzleLog = (tmBase >= 8 && tnBase >= 8) ? 1 : 0
    let tilePow = 1 << swizzleLog
    let tn = tnBase * tilePow
    let tm = Int(ceil(Double(tmBase) / Double(tilePow)))
    return MTLSize(width: tn, height: tm, depth: max(spec.batch, 1))
}
