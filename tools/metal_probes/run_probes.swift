import Foundation
import Metal

struct ProbeResult {
    let name: String
    let passed: Bool
    let details: [String]
}

// Simple deterministic RNG for probe data generation.
struct LCG {
    private var state: UInt64

    init(seed: UInt64) {
        precondition(seed != 0, "Seed must be non-zero")
        state = seed
    }

    mutating func nextUInt32() -> UInt32 {
        state = state &* 6364136223846793005 &+ 1
        return UInt32(truncatingIfNeeded: state >> 32)
    }

    mutating func nextFloat01() -> Float {
        let maxValue = Float(UInt32.max)
        return Float(nextUInt32()) / maxValue
    }

    mutating func nextFloatSigned() -> Float {
        return nextFloat01() * 2.0 - 1.0
    }
}

enum ProbeHarnessError: Error, CustomStringConvertible {
    case metalUnavailable
    case libraryNotFound(String)
    case functionNotFound(String)
    case commandQueueUnavailable
    case pipelineLimitExceeded(String)
    case validationFailed(String)

    var description: String {
        switch self {
        case .metalUnavailable:
            return "Metal device unavailable"
        case .libraryNotFound(let path):
            return "Failed to load Metal library at \(path)"
        case .functionNotFound(let fn):
            return "Failed to find kernel function '\(fn)'"
        case .commandQueueUnavailable:
            return "Failed to create Metal command queue"
        case .pipelineLimitExceeded(let msg):
            return "Pipeline configuration error: \(msg)"
        case .validationFailed(let msg):
            return "Validation failed: \(msg)"
        }
    }
}

// MARK: - Helpers

func loadPipeline(device: MTLDevice, buildDir: URL, metallibName: String, kernel: String) throws -> MTLComputePipelineState {
    let libURL = buildDir.appendingPathComponent("\(metallibName).metallib")
    guard FileManager.default.fileExists(atPath: libURL.path) else {
        throw ProbeHarnessError.libraryNotFound(libURL.path)
    }
    let library = try device.makeLibrary(URL: libURL)
    guard let function = library.makeFunction(name: kernel) else {
        throw ProbeHarnessError.functionNotFound("\(metallibName)::\(kernel)")
    }
    return try device.makeComputePipelineState(function: function)
}

func makeBuffer<T>(device: MTLDevice, data: [T]) -> MTLBuffer {
    precondition(!data.isEmpty, "Buffers must have at least one element")
    return data.withUnsafeBytes { raw in
        device.makeBuffer(bytes: raw.baseAddress!, length: raw.count, options: .storageModeShared)!
    }
}

func makeZeroBuffer<T>(device: MTLDevice, count: Int, as type: T.Type) -> MTLBuffer {
    precondition(count > 0, "Buffers must have at least one element")
    let length = count * MemoryLayout<T>.stride
    guard let buffer = device.makeBuffer(length: length, options: .storageModeShared) else {
        fatalError("Failed to allocate Metal buffer of length \(length)")
    }
    memset(buffer.contents(), 0, length)
    return buffer
}

func bufferContents<T>(_ buffer: MTLBuffer, count: Int, as type: T.Type = T.self) -> [T] {
    let pointer = buffer.contents().bindMemory(to: type, capacity: count)
    return Array(UnsafeBufferPointer(start: pointer, count: count))
}

func relativeDiff(_ a: Float, _ b: Float) -> Float {
    let denom = max(1.0, abs(a), abs(b))
    return abs(a - b) / denom
}

// MARK: - Probes

func runReduceMax(device: MTLDevice, buildDir: URL, queue: MTLCommandQueue) throws -> ProbeResult {
    let pipeline = try loadPipeline(device: device, buildDir: buildDir, metallibName: "simdgroup_reduce_max", kernel: "simdgroup_reduce_max")

    let tew = pipeline.threadExecutionWidth
    let maxThreads = pipeline.maxTotalThreadsPerThreadgroup
    let maxSimdGroups = maxThreads / tew
    guard maxSimdGroups >= 1 else {
        throw ProbeHarnessError.pipelineLimitExceeded("threadExecutionWidth=\(tew) maxThreads=\(maxThreads)")
    }

    var details: [String] = []
    var passed = true

    var rng = LCG(seed: 0xC0FFEE)

    let cases: [(name: String, builder: (Int, Int, inout LCG) -> (groups: Int, input: [Float], expected: [Float]))] = [
        ("ascending", { tew, maxGroups, _ in
            let groups = min(max(1, maxGroups), 2)
            var input = [Float](repeating: 0, count: tew * groups)
            var expected = [Float](repeating: 0, count: groups)
            for group in 0..<groups {
                for lane in 0..<tew {
                    let idx = group * tew + lane
                    input[idx] = Float(group * 1000 + lane)
                }
                expected[group] = Float(group * 1000 + (tew - 1))
            }
            return (groups, input, expected)
        }),
        ("random_full", { tew, maxGroups, rng in
            let groups = min(max(1, maxGroups), 1)
            var input = [Float](repeating: 0, count: tew * groups)
            var expected = [Float](repeating: -Float.infinity, count: groups)
            for group in 0..<groups {
                var groupMax = -Float.infinity
                for lane in 0..<tew {
                    let idx = group * tew + lane
                    var value = rng.nextFloatSigned() * 512.0
                    if lane % 7 == 0 {
                        value -= 128.0
                    }
                    input[idx] = value
                    groupMax = max(groupMax, value)
                }
                expected[group] = groupMax
            }
            return (groups, input, expected)
        }),
        ("random_tail_mask", { tew, maxGroups, rng in
            let groups = min(max(1, maxGroups), 1)
            var input = [Float](repeating: 0, count: tew * groups)
            var expected = [Float](repeating: -Float.infinity, count: groups)
            let validLanes = max(1, tew - 5)
            for group in 0..<groups {
                var groupMax = -Float.infinity
                for lane in 0..<tew {
                    let idx = group * tew + lane
                    if lane < validLanes {
                        let value = rng.nextFloatSigned() * 256.0
                        input[idx] = value
                        groupMax = max(groupMax, value)
                    } else {
                        input[idx] = -Float.infinity
                    }
                }
                expected[group] = groupMax
            }
            return (groups, input, expected)
        })
    ]

    for testCase in cases {
        let caseData = testCase.builder(tew, maxSimdGroups, &rng)
        let groups = caseData.groups
        guard groups > 0 else { continue }
        let threadsPerThreadgroup = tew * groups

        let inputBuffer = makeBuffer(device: device, data: caseData.input)
        let outputBuffer = makeZeroBuffer(device: device, count: groups, as: Float.self)

        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw ProbeHarnessError.commandQueueUnavailable
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)

        let threads = MTLSize(width: threadsPerThreadgroup, height: 1, depth: 1)
        let threadgroups = MTLSize(width: 1, height: 1, depth: 1)
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threads)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let results: [Float] = bufferContents(outputBuffer, count: groups)

        for group in 0..<groups {
            let expected = caseData.expected[group]
            let diff = relativeDiff(results[group], expected)
            details.append("[\(testCase.name)] group \(group): expected \(expected), got \(results[group]), diff \(diff)")
            if diff > 1e-5 && !(expected.isInfinite && results[group].isInfinite) {
                passed = false
            }
        }
    }

    return ProbeResult(name: "simdgroup_reduce_max", passed: passed, details: details)
}

func runBallotCompact(device: MTLDevice, buildDir: URL, queue: MTLCommandQueue) throws -> ProbeResult {
    let pipeline = try loadPipeline(device: device, buildDir: buildDir, metallibName: "simdgroup_ballot_compact", kernel: "simdgroup_ballot_compact")

    let tew = pipeline.threadExecutionWidth
    if tew > 32 {
        throw ProbeHarnessError.validationFailed("simdgroup_ballot_compact probe assumes SIMD width ≤ 32; device reported \(tew)")
    }
    let maxThreads = pipeline.maxTotalThreadsPerThreadgroup
    let maxSimdGroups = maxThreads / tew
    guard maxSimdGroups >= 1 else {
        throw ProbeHarnessError.pipelineLimitExceeded("threadExecutionWidth=\(tew) maxThreads=\(maxThreads)")
    }

    var passed = true
    var details: [String] = []

    var rng = LCG(seed: 0xBADBEEF)

    let cases: [(name: String, generator: (Int, Int, inout LCG) -> [[Bool]])] = [
        ("none_active", { tew, maxGroups, _ in
            let groups = min(max(1, maxGroups), 1)
            return Array(repeating: Array(repeating: false, count: tew), count: groups)
        }),
        ("all_active", { tew, maxGroups, _ in
            let groups = min(max(1, maxGroups), 1)
            return Array(repeating: Array(repeating: true, count: tew), count: groups)
        }),
        ("alternating", { tew, maxGroups, _ in
            let groups = min(max(1, maxGroups), 1)
            var pattern = [Bool]()
            pattern.reserveCapacity(tew)
            for lane in 0..<tew {
                pattern.append((lane % 2) == 0)
            }
            return Array(repeating: pattern, count: groups)
        }),
        ("prefix_valid", { tew, maxGroups, _ in
            let groups = min(max(1, maxGroups), 1)
            let valid = max(1, tew / 2)
            var pattern = [Bool](repeating: false, count: tew)
            for lane in 0..<valid {
                pattern[lane] = true
            }
            return Array(repeating: pattern, count: groups)
        }),
        ("sparse_random", { tew, maxGroups, rng in
            let groups = min(max(1, maxGroups), 2)
            var result: [[Bool]] = []
            for _ in 0..<groups {
                var pattern = [Bool](repeating: false, count: tew)
                for lane in 0..<tew {
                    pattern[lane] = rng.nextFloat01() < 0.3
                }
                result.append(pattern)
            }
            return result
        }),
        ("dense_random", { tew, maxGroups, rng in
            let groups = min(max(1, maxGroups), 2)
            var result: [[Bool]] = []
            for _ in 0..<groups {
                var pattern = [Bool](repeating: false, count: tew)
                for lane in 0..<tew {
                    pattern[lane] = rng.nextFloat01() < 0.75
                }
                result.append(pattern)
            }
            return result
        }),
        ("tail_only_valid", { tew, maxGroups, _ in
            let groups = min(max(1, maxGroups), 1)
            let valid = max(1, tew - 5)
            var pattern = [Bool](repeating: false, count: tew)
            for lane in 0..<valid {
                pattern[lane] = lane % 3 == 0
            }
            return Array(repeating: pattern, count: groups)
        })
    ]

    for testCase in cases {
        let boolGroups = testCase.generator(tew, maxSimdGroups, &rng)
        let groups = boolGroups.count
        guard groups > 0 else { continue }
        let threadsPerThreadgroup = tew * groups

        var input = [Float](repeating: 0, count: threadsPerThreadgroup)
        for group in 0..<groups {
            for lane in 0..<tew {
                let idx = group * tew + lane
                input[idx] = boolGroups[group][lane] ? 1.0 : 0.0
            }
        }

        let inputBuffer = makeBuffer(device: device, data: input)
        let compactedBuffer = makeZeroBuffer(device: device, count: threadsPerThreadgroup, as: UInt32.self)
        let countsBuffer = makeZeroBuffer(device: device, count: groups, as: UInt32.self)
        let masksBuffer = makeZeroBuffer(device: device, count: groups, as: UInt64.self)

        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw ProbeHarnessError.commandQueueUnavailable
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(compactedBuffer, offset: 0, index: 1)
        encoder.setBuffer(countsBuffer, offset: 0, index: 2)
        encoder.setBuffer(masksBuffer, offset: 0, index: 3)

        let threads = MTLSize(width: threadsPerThreadgroup, height: 1, depth: 1)
        let threadgroups = MTLSize(width: 1, height: 1, depth: 1)
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threads)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let counts: [UInt32] = bufferContents(countsBuffer, count: groups)
        let masks: [UInt64] = bufferContents(masksBuffer, count: groups)
        let compacted: [UInt32] = bufferContents(compactedBuffer, count: threadsPerThreadgroup)

        for group in 0..<groups {
            let bools = boolGroups[group]
            let expectedCount = UInt32(bools.filter { $0 }.count)
            let observedCount = counts[group]
            if observedCount != expectedCount {
                passed = false
                details.append("[\(testCase.name)] group \(group): expected count \(expectedCount), got \(observedCount)")
            }

            var mask: UInt64 = 0
            for lane in 0..<tew {
                if bools[lane] {
                    mask |= UInt64(1 << lane)
                }
            }
            let anyBit: UInt64 = expectedCount > 0 ? 1 : 0
            let allBit: UInt64 = expectedCount == UInt32(tew) ? 1 : 0
            let expectedMeta = mask | (anyBit << 62) | (allBit << 63)
            if masks[group] != expectedMeta {
                passed = false
                details.append("[\(testCase.name)] group \(group): expected mask/meta \(expectedMeta), got \(masks[group])")
            }

            let base = group * tew
            let observedLanes = (0..<Int(expectedCount)).map { compacted[base + $0] }
            let expectedLanes = bools.enumerated()
                .filter { $0.element }
                .map { UInt32($0.offset) }
            if observedLanes != expectedLanes {
                passed = false
                details.append("[\(testCase.name)] group \(group): expected lanes \(expectedLanes), got \(observedLanes)")
            }
        }
    }

    return ProbeResult(name: "simdgroup_ballot_compact", passed: passed, details: details)
}

func runRegisterTopK(device: MTLDevice, buildDir: URL, queue: MTLCommandQueue) throws -> ProbeResult {
    let metallibName = "simdgroup_register_topk"
    let kernelName = "simdgroup_register_topk"
    let pipeline = try loadPipeline(device: device, buildDir: buildDir, metallibName: metallibName, kernel: kernelName)

    let tew = pipeline.threadExecutionWidth
    let maxThreads = pipeline.maxTotalThreadsPerThreadgroup
    let maxSimdGroups = maxThreads / tew
    guard maxSimdGroups >= 1 else {
        throw ProbeHarnessError.pipelineLimitExceeded("threadExecutionWidth=\(tew) maxThreads=\(maxThreads)")
    }

    let itemsPerLane = 8
    let keep = 4

    var passed = true
    var details: [String] = []

    var rng = LCG(seed: 0xACEDFACE)

    let cases: [(name: String, builder: (Int, Int, inout LCG) -> (threads: Int, input: [Float], expected: [[(Float, UInt32)]]))] = [
        ("monotonic", { tew, maxGroups, _ in
            let groups = min(max(1, maxGroups), 2)
            let threads = tew * groups
            let totalValues = threads * itemsPerLane
            var input = [Float](repeating: 0, count: totalValues)
            var expected = Array(repeating: [(Float, UInt32)](), count: threads)
            for thread in 0..<threads {
                var local: [(Float, UInt32)] = []
                for item in 0..<itemsPerLane {
                    let index = thread + item * threads
                    let value = Float(index)
                    input[index] = value
                    local.append((value, UInt32(index)))
                }
                local.sort { lhs, rhs in
                    if lhs.0 == rhs.0 {
                        return lhs.1 < rhs.1
                    }
                    return lhs.0 > rhs.0
                }
                expected[thread] = Array(local.prefix(keep))
            }
            return (threads, input, expected)
        }),
        ("random_values", { tew, maxGroups, rng in
            let groups = min(max(1, maxGroups), 2)
            let threads = tew * groups
            let totalValues = threads * itemsPerLane
            var input = [Float](repeating: 0, count: totalValues)
            var expected = Array(repeating: [(Float, UInt32)](), count: threads)
            for thread in 0..<threads {
                var local: [(Float, UInt32)] = []
                for item in 0..<itemsPerLane {
                    let index = thread + item * threads
                    let value = rng.nextFloatSigned() * 1024.0
                    input[index] = value
                    local.append((value, UInt32(index)))
                }
                local.sort { lhs, rhs in
                    if lhs.0 == rhs.0 {
                        return lhs.1 < rhs.1
                    }
                    return lhs.0 > rhs.0
                }
                expected[thread] = Array(local.prefix(keep))
            }
            return (threads, input, expected)
        }),
        ("masked_values", { tew, maxGroups, rng in
            let groups = min(max(1, maxGroups), 1)
            let threads = tew * groups
            let totalValues = threads * itemsPerLane
            var input = [Float](repeating: 0, count: totalValues)
            var expected = Array(repeating: [(Float, UInt32)](), count: threads)
            let validItems = itemsPerLane - 2
            for thread in 0..<threads {
                var local: [(Float, UInt32)] = []
                for item in 0..<itemsPerLane {
                    let index = thread + item * threads
                    if item < validItems {
                        let value = rng.nextFloatSigned() * 256.0
                        input[index] = value
                        local.append((value, UInt32(index)))
                    } else {
                        input[index] = -Float.infinity
                    }
                }
                local.sort { lhs, rhs in
                    if lhs.0 == rhs.0 {
                        return lhs.1 < rhs.1
                    }
                    return lhs.0 > rhs.0
                }
                expected[thread] = Array(local.prefix(min(keep, local.count)))
            }
            return (threads, input, expected)
        })
    ]

    for testCase in cases {
        let caseData = testCase.builder(tew, maxSimdGroups, &rng)
        let threads = caseData.threads
        guard threads > 0 else { continue }
        let valuesBuffer = makeZeroBuffer(device: device, count: threads * keep, as: Float.self)
        let indicesBuffer = makeZeroBuffer(device: device, count: threads * keep, as: UInt32.self)
        let inputBuffer = makeBuffer(device: device, data: caseData.input)

        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw ProbeHarnessError.commandQueueUnavailable
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(valuesBuffer, offset: 0, index: 1)
        encoder.setBuffer(indicesBuffer, offset: 0, index: 2)

        let threadsSize = MTLSize(width: threads, height: 1, depth: 1)
        let threadgroups = MTLSize(width: 1, height: 1, depth: 1)
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsSize)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let values: [Float] = bufferContents(valuesBuffer, count: threads * keep)
        let indices: [UInt32] = bufferContents(indicesBuffer, count: threads * keep)

        for thread in 0..<threads {
            let start = thread * keep
            let actualIndices = Array(indices[start..<(start + keep)])
            let actualValues = Array(values[start..<(start + keep)])

            let expectedPairs = caseData.expected[thread]
            let expectedIndices = expectedPairs.map { $0.1 }

            let actualIndexSet = Set(actualIndices)
            let expectedIndexSet = Set(expectedIndices)
            if actualIndexSet != expectedIndexSet {
                passed = false
                details.append("[\(testCase.name)] thread \(thread): expected indices \(expectedIndices), got \(actualIndices)")
            }

            var expectedMap: [UInt32: Float] = [:]
            for pair in expectedPairs {
                expectedMap[pair.1] = pair.0
            }
            for (idx, value) in zip(actualIndices, actualValues) {
                if let expectedValue = expectedMap[idx] {
                    let diff = relativeDiff(value, expectedValue)
                    if diff > 1e-4 && !(expectedValue.isInfinite && value.isInfinite) {
                        passed = false
                        details.append("[\(testCase.name)] thread \(thread): index \(idx) expected value \(expectedValue), got \(value), diff \(diff)")
                    }
                } else if idx == 0 && expectedPairs.isEmpty {
                    // When no candidates exist, kernel currently writes zeros; allow this path.
                    continue
                } else {
                    passed = false
                    details.append("[\(testCase.name)] thread \(thread): unexpected index \(idx)")
                }
            }
        }
    }

    return ProbeResult(name: "simdgroup_register_topk", passed: passed, details: details)
}

// MARK: - Entry

func runUnrolledReductions(device: MTLDevice, buildDir: URL, queue: MTLCommandQueue) throws -> ProbeResult {
    let libName = "simdgroup_unrolled_reductions"
    let sumPipeline = try loadPipeline(device: device, buildDir: buildDir, metallibName: libName, kernel: "simdgroup_unrolled_sum")
    let argmaxPipeline = try loadPipeline(device: device, buildDir: buildDir, metallibName: libName, kernel: "simdgroup_unrolled_argmax")

    let tew = sumPipeline.threadExecutionWidth
    let maxThreads = min(sumPipeline.maxTotalThreadsPerThreadgroup, argmaxPipeline.maxTotalThreadsPerThreadgroup)
    let groups = max(1, maxThreads / tew)

    var details: [String] = []
    var passed = true

    // Sum test: values are lane_id + group*1000. Sum per group is sum_{i=0..tew-1} (i + group*1000)
    var sumInput = [UInt32]()
    sumInput.reserveCapacity(tew * groups)
    var expectedSums = [UInt32](repeating: 0, count: groups)
    for g in 0..<groups {
        var acc: UInt32 = 0
        for lane in 0..<tew {
            let v = UInt32(g * 1000 + lane)
            sumInput.append(v)
            acc &+= v
        }
        expectedSums[g] = acc
    }

    let device = device
    let queue = queue
    let sumInBuf = makeBuffer(device: device, data: sumInput)
    let sumOutBuf = makeZeroBuffer(device: device, count: groups, as: UInt32.self)

    if let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() {
        enc.setComputePipelineState(sumPipeline)
        enc.setBuffer(sumInBuf, offset: 0, index: 0)
        enc.setBuffer(sumOutBuf, offset: 0, index: 1)
        let threads = MTLSize(width: tew * groups, height: 1, depth: 1)
        let tgs = MTLSize(width: 1, height: 1, depth: 1)
        enc.dispatchThreadgroups(tgs, threadsPerThreadgroup: threads)
        enc.endEncoding()
        cb.commit(); cb.waitUntilCompleted()
    } else {
        throw ProbeHarnessError.commandQueueUnavailable
    }

    let gotSums: [UInt32] = bufferContents(sumOutBuf, count: groups)
    for g in 0..<groups {
        if gotSums[g] != expectedSums[g] {
            passed = false
            details.append("[sum] group \(g): expected \(expectedSums[g]), got \(gotSums[g])")
        }
    }

    // Argmax test: Put a known max per group at a lane that varies modulo tew, with tie-breaker on index.
    var argInput = [Float](repeating: -Float.infinity, count: tew * groups)
    var expectedBestVal = [Float](repeating: 0, count: groups)
    var expectedBestIdx = [UInt32](repeating: 0, count: groups)
    var expectedBestLane = [UInt32](repeating: 0, count: groups)
    for g in 0..<groups {
        let bestLane = (3 * g + 5) % tew
        let base = g * tew
        for lane in 0..<tew {
            argInput[base + lane] = Float(g * 10 + lane)
        }
        argInput[base + bestLane] = 9999.0 // guaranteed max
        expectedBestVal[g] = 9999.0
        expectedBestIdx[g] = UInt32(base + bestLane)
        expectedBestLane[g] = UInt32(bestLane)
    }

    let argInBuf = makeBuffer(device: device, data: argInput)
    let outBestVal = makeZeroBuffer(device: device, count: groups, as: Float.self)
    let outBestIdx = makeZeroBuffer(device: device, count: groups, as: UInt32.self)
    let outBestLane = makeZeroBuffer(device: device, count: groups, as: UInt32.self)

    if let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() {
        enc.setComputePipelineState(argmaxPipeline)
        enc.setBuffer(argInBuf, offset: 0, index: 0)
        enc.setBuffer(outBestVal, offset: 0, index: 1)
        enc.setBuffer(outBestIdx, offset: 0, index: 2)
        enc.setBuffer(outBestLane, offset: 0, index: 3)
        let threads = MTLSize(width: tew * groups, height: 1, depth: 1)
        let tgs = MTLSize(width: 1, height: 1, depth: 1)
        enc.dispatchThreadgroups(tgs, threadsPerThreadgroup: threads)
        enc.endEncoding()
        cb.commit(); cb.waitUntilCompleted()
    } else {
        throw ProbeHarnessError.commandQueueUnavailable
    }

    let gotBestVal: [Float] = bufferContents(outBestVal, count: groups)
    let gotBestIdx: [UInt32] = bufferContents(outBestIdx, count: groups)
    let gotBestLane: [UInt32] = bufferContents(outBestLane, count: groups)

    for g in 0..<groups {
        if relativeDiff(gotBestVal[g], expectedBestVal[g]) > 1e-6 || gotBestIdx[g] != expectedBestIdx[g] || gotBestLane[g] != expectedBestLane[g] {
            passed = false
            details.append("[argmax] group \(g): expected (\(expectedBestVal[g]), idx=\(expectedBestIdx[g]), lane=\(expectedBestLane[g])) got (\(gotBestVal[g]), idx=\(gotBestIdx[g]), lane=\(gotBestLane[g]))")
        }
    }

    return ProbeResult(name: "simdgroup_unrolled_reductions", passed: passed, details: details)
}

func runQuadlaneReductions(device: MTLDevice, buildDir: URL, queue: MTLCommandQueue) throws -> ProbeResult {
    let libName = "simdgroup_quadlane_reductions"
    let sumPipeline = try loadPipeline(device: device, buildDir: buildDir, metallibName: libName, kernel: "simdgroup_quadlane_sum")
    let argmaxPipeline = try loadPipeline(device: device, buildDir: buildDir, metallibName: libName, kernel: "simdgroup_quadlane_argmax")

    let tew = sumPipeline.threadExecutionWidth
    let maxThreads = min(sumPipeline.maxTotalThreadsPerThreadgroup, argmaxPipeline.maxTotalThreadsPerThreadgroup)
    let groups = max(1, maxThreads / tew)

    var details: [String] = []
    var passed = true

    // Sum test with patterned inputs to stress intra-quad and cross-quad paths.
    var sumInput = [UInt32]()
    sumInput.reserveCapacity(tew * groups)
    var expectedSums = [UInt32](repeating: 0, count: groups)
    for g in 0..<groups {
        var acc: UInt32 = 0
        for lane in 0..<tew {
            let val: UInt32 = UInt32(((lane % 4) * 10) + (lane / 4) + g * 100)
            sumInput.append(val)
            acc &+= val
        }
        expectedSums[g] = acc
    }

    let sumInBuf = makeBuffer(device: device, data: sumInput)
    let sumOutBuf = makeZeroBuffer(device: device, count: groups, as: UInt32.self)

    if let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() {
        enc.setComputePipelineState(sumPipeline)
        enc.setBuffer(sumInBuf, offset: 0, index: 0)
        enc.setBuffer(sumOutBuf, offset: 0, index: 1)
        let threads = MTLSize(width: tew * groups, height: 1, depth: 1)
        let tgs = MTLSize(width: 1, height: 1, depth: 1)
        enc.dispatchThreadgroups(tgs, threadsPerThreadgroup: threads)
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    } else { throw ProbeHarnessError.commandQueueUnavailable }

    let gotSums: [UInt32] = bufferContents(sumOutBuf, count: groups)
    for g in 0..<groups { if gotSums[g] != expectedSums[g] { passed = false; details.append("[sum] group \(g): expected \(expectedSums[g]) got \(gotSums[g])") } }

    // Argmax test: Fix a very large value at a lane that cycles per group to force quad boundaries.
    var argInput = [Float](repeating: -Float.infinity, count: tew * groups)
    var expectedBestVal = [Float](repeating: 0, count: groups)
    var expectedBestIdx = [UInt32](repeating: 0, count: groups)
    var expectedBestLane = [UInt32](repeating: 0, count: groups)
    for g in 0..<groups {
        let bestLane = (g * 5 + 1) % tew
        let base = g * tew
        for lane in 0..<tew { argInput[base + lane] = Float(lane) + Float(g) * 0.01 }
        argInput[base + bestLane] = 1e6
        expectedBestVal[g] = 1e6
        expectedBestIdx[g] = UInt32(base + bestLane)
        expectedBestLane[g] = UInt32(bestLane)
    }

    let argInBuf = makeBuffer(device: device, data: argInput)
    let outBestVal = makeZeroBuffer(device: device, count: groups, as: Float.self)
    let outBestIdx = makeZeroBuffer(device: device, count: groups, as: UInt32.self)
    let outBestLane = makeZeroBuffer(device: device, count: groups, as: UInt32.self)

    if let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() {
        enc.setComputePipelineState(argmaxPipeline)
        enc.setBuffer(argInBuf, offset: 0, index: 0)
        enc.setBuffer(outBestVal, offset: 0, index: 1)
        enc.setBuffer(outBestIdx, offset: 0, index: 2)
        enc.setBuffer(outBestLane, offset: 0, index: 3)
        let threads = MTLSize(width: tew * groups, height: 1, depth: 1)
        let tgs = MTLSize(width: 1, height: 1, depth: 1)
        enc.dispatchThreadgroups(tgs, threadsPerThreadgroup: threads)
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    } else { throw ProbeHarnessError.commandQueueUnavailable }

    let gotBestVal: [Float] = bufferContents(outBestVal, count: groups)
    let gotBestIdx: [UInt32] = bufferContents(outBestIdx, count: groups)
    let gotBestLane: [UInt32] = bufferContents(outBestLane, count: groups)

    for g in 0..<groups {
        if relativeDiff(gotBestVal[g], expectedBestVal[g]) > 1e-6 || gotBestIdx[g] != expectedBestIdx[g] || gotBestLane[g] != expectedBestLane[g] {
            passed = false
            details.append("[argmax] group \(g): expected (\(expectedBestVal[g]), idx=\(expectedBestIdx[g]), lane=\(expectedBestLane[g])) got (\(gotBestVal[g]), idx=\(gotBestIdx[g]), lane=\(gotBestLane[g]))")
        }
    }

    return ProbeResult(name: "simdgroup_quadlane_reductions", passed: passed, details: details)
}

func runHarness(buildDir: URL) throws -> [ProbeResult] {
    guard let device = MTLCreateSystemDefaultDevice() else {
        throw ProbeHarnessError.metalUnavailable
    }
    guard let queue = device.makeCommandQueue() else {
        throw ProbeHarnessError.commandQueueUnavailable
    }

    var results: [ProbeResult] = []
    results.append(try runReduceMax(device: device, buildDir: buildDir, queue: queue))
    results.append(try runBallotCompact(device: device, buildDir: buildDir, queue: queue))
    results.append(try runRegisterTopK(device: device, buildDir: buildDir, queue: queue))
    results.append(try runUnrolledReductions(device: device, buildDir: buildDir, queue: queue))
    results.append(try runQuadlaneReductions(device: device, buildDir: buildDir, queue: queue))
    return results
}

do {
    let args = CommandLine.arguments
    let buildDir: URL
    if args.count > 1 {
        buildDir = URL(fileURLWithPath: args[1], isDirectory: true)
    } else {
        buildDir = URL(fileURLWithPath: ".build", isDirectory: true)
    }

    let results = try runHarness(buildDir: buildDir)
    var anyFailed = false

    for result in results {
        let status = result.passed ? "PASS" : "FAIL"
        print("[\(status)] \(result.name)")
        for line in result.details {
            print("  - \(line)")
        }
        if !result.passed {
            anyFailed = true
        }
    }

    if anyFailed {
        exit(1)
    }
} catch {
    fputs("Probe harness error: \(error)\n", stderr)
    exit(1)
}
