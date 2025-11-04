import Foundation
import Metal
import MetalPerformanceShaders
import Darwin

struct MatmulShapeSpec: Hashable, Codable {
    enum Backend: String, CaseIterable, Codable {
        case mlx
        case mps
        case gemv
        case gemmTiled = "gemm_tiled"
        case m1Optimized = "m1_optimized"
        case m1OptimizedV2 = "m1_optimized_v2"
        case m1OptimizedV3 = "m1_optimized_v3"
        case m1OptimizedV4 = "m1_optimized_v4"
        case m1OptimizedV5 = "m1_optimized_v5"
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

struct BenchmarkResult: Codable {
    let spec: MatmulShapeSpec
    let backend: MatmulShapeSpec.Backend
    let variantName: String
    let library: String?
    let isBaseline: Bool
    let gpuTimingsMs: [Double]
    let cpuTimingsMs: [Double]
    let maxAbsError: Float
    let maxRelError: Float

    var averageGpuMs: Double? {
        guard !gpuTimingsMs.isEmpty else { return nil }
        return gpuTimingsMs.reduce(0, +) / Double(gpuTimingsMs.count)
    }

    var averageCpuMs: Double? {
        guard !cpuTimingsMs.isEmpty else { return nil }
        return cpuTimingsMs.reduce(0, +) / Double(cpuTimingsMs.count)
    }
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
            return "usage: run_matmul_probes.swift [--verbose-failures] <build_dir> <matmul_dir> <sizes_markdown>"
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

struct KernelVariantConfig: Decodable {
    let name: String
    let library: String?
    let enabled: Bool?
    let baseline: Bool?
    let tileBM: Int?
    let tileBN: Int?
    let tileBK: Int?
    let warpM: Int?
    let warpN: Int?
    let transposeBOverride: Bool?
    let supports: KernelSupport?
}

struct KernelSupport: Decodable {
    let transposeA: Bool
    let transposeB: Bool
    let smallK: Bool
    let smallMN: Bool
    let batch: Bool
    let bias: Bool
    let accumulate: Bool
    let supportedNValues: [Int]?
    let functionName: String
    let expectedTransposeA: Bool?
    let expectedTransposeB: Bool?
    let functionNameBias: String?
    let functionNameAccumulate: String?
}

struct KernelVariant {
    let name: String
    let library: String?
    let isBaseline: Bool
    let tileOverride: (Int, Int, Int, Int, Int)?
    let transposeBOverride: Bool?
    let supports: KernelSupport?
}

struct VariantFailure {
    let spec: MatmulShapeSpec
    let backend: MatmulShapeSpec.Backend
    let variantName: String
    let errorDescription: String
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

// Structure to hold original profiling data
struct OriginalProfileData {
    var avgTimeMs: Double = 0.0
    var minTimeMs: Double = 0.0
    var maxTimeMs: Double = 0.0
    var count: Int = 0
}
