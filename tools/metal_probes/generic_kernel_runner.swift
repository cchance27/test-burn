import Foundation
import Metal

// Generic kernel runner that can dispatch any kernel type based on its signature
class GenericKernelRunner: BaseBackendRunner {
    
    func runGenericKernel(
        spec: MatmulShapeSpec,
        variant: KernelVariant,
        backendType: MatmulShapeSpec.Backend,
        functionName: String
    ) throws -> BenchmarkResult {
        
        guard let libraryName = variant.library else {
            throw HarnessError.libraryLoadFailed("No library specified for \(backendType.rawValue)/\(variant.name)")
        }
        
        guard let library = libraries[libraryName] else {
            throw HarnessError.libraryLoadFailed("\(libraryName).metallib")
        }
        
        let tensors = loadMatmulTensors(spec: spec)
        let alpha = spec.alpha
        let beta = spec.beta

        // Handle function constants if needed (for MLX-style kernels)
        let function: MTLFunction
        if variant.tileOverride != nil {
            // For MLX-style kernels with tile configuration
            let selectedTile = variant.tileOverride!
            let constants = makeMLXConstants(
                spec: spec,
                tile: (selectedTile.0, selectedTile.1, selectedTile.2),
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

            function = try library.makeFunction(name: functionName, constantValues: functionConstants)
        } else {
            function = library.makeFunction(name: functionName)!
        }
        
        guard let pipeline = try? device.makeComputePipelineState(function: function) else {
            throw HarnessError.pipelineCreationFailed(functionName)
        }

        var gpuTimings: [Double] = []
        var cpuTimings: [Double] = []
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
            
            // Set bias buffer if needed
            var paramBufferIndex = 3
            if spec.bias {
                if let biasBuffer = tensors.bias {
                    encoder.setBuffer(biasBuffer, offset: 0, index: paramBufferIndex)
                    paramBufferIndex += 1
                }
            }
            
            // For MLX-style kernels with function constants, we may need to pass additional parameters
            if variant.tileOverride != nil {
                // For MLX kernels, set up parameters similar to the original implementation
                let constants = makeMLXConstants(
                    spec: spec,
                    tile: (variant.tileOverride!.0, variant.tileOverride!.1, variant.tileOverride!.2),
                    alpha: alpha,
                    beta: beta
                )
                
                var params = buildGEMMParams(  // Changed from 'let' to 'var' to allow passing as inout
                    spec: spec,
                    tile: (variant.tileOverride!.0, variant.tileOverride!.1, variant.tileOverride!.2),
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
                
                // Adjust buffer indices based on presence of bias
                if spec.bias {
                    encoder.setBytes(&params, length: MemoryLayout<GEMMParams>.stride, index: paramBufferIndex)
                    if var addmm = addmmParams {  // Changed from 'let' to 'var' to allow passing as inout
                        encoder.setBytes(&addmm, length: MemoryLayout<GEMMAddMMParams>.stride, index: paramBufferIndex + 1)
                    } else if addmmParams != nil {
                        // If addmm is needed but not changed, we still need to handle it
                        var addmm = addmmParams!
                        encoder.setBytes(&addmm, length: MemoryLayout<GEMMAddMMParams>.stride, index: paramBufferIndex + 1)
                    }
                    var batchShapeCopy = batchShape
                    encoder.setBytes(&batchShapeCopy, length: MemoryLayout.size(ofValue: batchShapeCopy), index: paramBufferIndex + 2)
                    var batchStridesCopy = batchStrides
                    encoder.setBytes(&batchStridesCopy, length: MemoryLayout.size(ofValue: batchStridesCopy), index: paramBufferIndex + 3)
                    
                    paramBufferIndex += 4
                } else {
                    encoder.setBytes(&params, length: MemoryLayout<GEMMParams>.stride, index: paramBufferIndex)
                    if var addmm = addmmParams {  // Changed from 'let' to 'var' to allow passing as inout
                        encoder.setBytes(&addmm, length: MemoryLayout<GEMMAddMMParams>.stride, index: paramBufferIndex + 1)
                    } else if addmmParams != nil {
                        // If addmm is needed but not changed, we still need to handle it
                        var addmm = addmmParams!
                        encoder.setBytes(&addmm, length: MemoryLayout<GEMMAddMMParams>.stride, index: paramBufferIndex + 1)
                    }
                    var batchShapeCopy = batchShape
                    encoder.setBytes(&batchShapeCopy, length: MemoryLayout.size(ofValue: batchShapeCopy), index: paramBufferIndex + 2)
                    var batchStridesCopy = batchStrides
                    encoder.setBytes(&batchStridesCopy, length: MemoryLayout.size(ofValue: batchStridesCopy), index: paramBufferIndex + 3)
                    
                    paramBufferIndex += 4
                }
            } else {
                // For other kernel types, pass basic dimensions
                var mParam = Int32(spec.m)
                var nParam = Int32(spec.n)
                var kParam = Int32(spec.k)
                encoder.setBytes(&mParam, length: MemoryLayout.size(ofValue: mParam), index: paramBufferIndex)
                encoder.setBytes(&nParam, length: MemoryLayout.size(ofValue: nParam), index: paramBufferIndex + 1)
                encoder.setBytes(&kParam, length: MemoryLayout.size(ofValue: kParam), index: paramBufferIndex + 2)
                
                paramBufferIndex += 3
            }
            
            // Calculate threadgroups based on the specific kernel requirements
            let (threadgroups, threadsPerTG) = calculateDispatchForKernel(spec: spec, variant: variant, pipeline: pipeline, backendType: backendType)
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerTG)
            encoder.endEncoding()
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
            backend: backendType,
            variantName: variant.name,
            library: libraryName,
            isBaseline: variant.isBaseline,
            gpuTimingsMs: gpuTimings,
            cpuTimingsMs: cpuTimings,
            maxAbsError: safeMaxAbsError,
            maxRelError: safeMaxRelError
        )
    }
    
    private func calculateDispatchForKernel(spec: MatmulShapeSpec, variant: KernelVariant, pipeline: MTLComputePipelineState, backendType: MatmulShapeSpec.Backend) -> (MTLSize, MTLSize) {
        // Handle different backend types differently
        switch backendType {
        case .mlx:
            // For MLX-style kernels, use the original threadgroup calculation
            if let tileOverride = variant.tileOverride {
                let (tileBM, tileBN, _, wmSel, wnSel) = tileOverride
                let threadgroups = makeThreadgroups(spec: spec, tile: (tileBM, tileBN), constants: MLXFunctionConstants())
                let threadsPerTG = MTLSize(width: 32, height: wnSel, depth: wmSel)  // MLX uses 32-width with specific heights
                return (threadgroups, threadsPerTG)
            } else {
                // Fallback for MLX without tile override
                return calculateGenericDispatch(spec: spec, pipeline: pipeline)
            }
        case .m1Optimized, .m1OptimizedV2, .m1OptimizedV3, .m1OptimizedV4:
            // For M1 optimized kernels
            return calculateM1Dispatch(spec: spec, pipeline: pipeline, variant: variant)
        case .gemv:
            // For GEMV kernels
            return calculateGEMVDispatch(spec: spec, pipeline: pipeline)
        case .gemmTiled:
            // For GEMM Tiled kernels
            return calculateGEMMTiledDispatch(spec: spec, pipeline: pipeline)
        default:
            // Generic fallback
            return calculateGenericDispatch(spec: spec, pipeline: pipeline)
        }
    }
    
    private func calculateGenericDispatch(spec: MatmulShapeSpec, pipeline: MTLComputePipelineState) -> (MTLSize, MTLSize) {
        // Default threadgroup size based on pipeline characteristics
        let threadExecutionWidth = pipeline.threadExecutionWidth
        let maxThreadsPerGroup = pipeline.maxTotalThreadsPerThreadgroup
        let height = max(1, min(maxThreadsPerGroup / threadExecutionWidth, 8)) // Limit height to reasonable value
        
        let threadgroupSize = MTLSize(width: threadExecutionWidth, height: height, depth: 1)
        
        // Calculate grid size based on problem dimensions
        let widthGroups = (spec.n + 127) / 128  // 128 elements per group width
        let heightGroups = (spec.m + 127) / 128 // 128 elements per group height
        let depthGroups = spec.batch > 1 ? spec.batch : 1
        
        let threadgroups = MTLSize(width: widthGroups, height: heightGroups, depth: depthGroups)
        
        return (threadgroups, threadgroupSize)
    }
    
    private func calculateM1Dispatch(spec: MatmulShapeSpec, pipeline: MTLComputePipelineState, variant: KernelVariant) -> (MTLSize, MTLSize) {
        // Robust token parsing: find bnXX and tgYY only when immediately followed by digits.
        func extractInt(after token: String, in name: String, default def: Int) -> Int {
            var found: Int? = nil
            var searchStart = name.startIndex
            while let r = name[searchStart...].range(of: token) {
                let start = r.upperBound
                if start < name.endIndex, name[start].isNumber {
                    var i = start
                    var digits = ""
                    while i < name.endIndex, name[i].isNumber {
                        digits.append(name[i])
                        i = name.index(after: i)
                    }
                    if let val = Int(digits) { found = val }
                }
                searchStart = name.index(after: r.lowerBound)
            }
            return found ?? def
        }

        let name = variant.name
        if name.contains("small_k") {
            let threadgroupSize = MTLSize(width: 32, height: 1, depth: 1)
            let threadgroups = MTLSize(width: (spec.n + 32 - 1) / 32, height: 1, depth: 1)
            return (threadgroups, threadgroupSize)
        }

        let cols = extractInt(after: "bn", in: name, default: (name.contains("bn64") ? 64 : 128))
        let tg   = extractInt(after: "tg", in: name, default: 128)

        let columnsPerThreadgroup = max(1, cols)
        let threadgroupWidth = max(1, tg)
        let threadgroupSize = MTLSize(width: threadgroupWidth, height: 1, depth: 1)
        let threadgroups = MTLSize(width: (spec.n + columnsPerThreadgroup - 1) / columnsPerThreadgroup, height: 1, depth: 1)

        return (threadgroups, threadgroupSize)
    }
    
    private func calculateGEMVDispatch(spec: MatmulShapeSpec, pipeline: MTLComputePipelineState) -> (MTLSize, MTLSize) {
        // GEMV typically dispatches based on N dimension for small N values
        let threadgroupSize: MTLSize
        let threadgroups: MTLSize
        
        if spec.n <= 16 && [1, 2, 4, 8, 16].contains(spec.n) {
            // For small N GEMV kernels
            if spec.n == 1 || spec.n == 2 {
                threadgroupSize = MTLSize(width: 32, height: 1, depth: 1)
                let groups = (spec.m + 31) / 32
                threadgroups = MTLSize(width: groups, height: 1, depth: 1)
            } else if spec.n == 4 {
                threadgroupSize = MTLSize(width: 64, height: 1, depth: 1) // 16 rows x 4 cols
                let groups = (spec.m + 15) / 16
                threadgroups = MTLSize(width: groups, height: 1, depth: 1)
            } else if spec.n == 8 {
                threadgroupSize = MTLSize(width: 64, height: 1, depth: 1) // 8 rows x 8 cols
                let groups = max(1, (spec.m + 7) / 8)
                threadgroups = MTLSize(width: 1, height: groups, depth: 1)
            } else { // spec.n == 16
                threadgroupSize = MTLSize(width: 64, height: 1, depth: 1) // 4 rows x 16 cols
                let groups = (spec.m + 3) / 4
                threadgroups = MTLSize(width: groups, height: 1, depth: 1)
            }
        } else {
            // For general GEMV kernel
            threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
            threadgroups = MTLSize(width: (spec.n + 255) / 256, height: 1, depth: 1)
        }
        
        return (threadgroups, threadgroupSize)
    }
    
    private func calculateGEMMTiledDispatch(spec: MatmulShapeSpec, pipeline: MTLComputePipelineState) -> (MTLSize, MTLSize) {
        // GEMM Tiled uses tile-based dispatch
        let tileM = min(spec.m, 64)  // tile size for M dimension
        let tileN = min(spec.n, 64)  // tile size for N dimension
        
        let threadgroups = MTLSize(
            width: (spec.n + tileN - 1) / tileN,
            height: (spec.m + tileM - 1) / tileM,
            depth: 1
        )
        
        // Use 64 threads per threadgroup for the tiled implementation
        let threadgroupSize = MTLSize(width: 64, height: 1, depth: 1)
        
        return (threadgroups, threadgroupSize)
    }
}

// Unified backend runner that handles all kernel types
class UnifiedBackendRunner: BaseBackendRunner, BackendRunner {
    private struct MLXSpecialization {
        let tile: (Int, Int, Int, Int, Int)
        let constants: MLXFunctionConstants
    }

    private struct MLXLaunchContext {
        var params: GEMMParams
        var addmmParams: GEMMAddMMParams?
        var batchShape: Int32
        var batchStrides: (UInt, UInt, UInt)
        var threadgroups: MTLSize
        var threadsPerThreadgroup: MTLSize
        var useOutSource: Bool
    }
    
    func runVariant(
        spec: MatmulShapeSpec,
        backend: MatmulShapeSpec.Backend,
        variant: KernelVariant
    ) throws -> BenchmarkResult {
        let kernelSpec = applyOverridesIfNeeded(spec: spec, variant: variant)
        let functionName = determineFunctionName(spec: kernelSpec, variant: variant)
        
        return try runGenericKernelInlined(
            originalSpec: spec,
            kernelSpec: kernelSpec,
            backendType: backend,
            variant: variant,
            functionName: functionName
        )
    }
    
    private func applyOverridesIfNeeded(
        spec: MatmulShapeSpec,
        variant: KernelVariant
    ) -> MatmulShapeSpec {
        guard let transposeBOverride = variant.transposeBOverride else {
            return spec
        }
        
        return MatmulShapeSpec(
            op: spec.op,
            backend: spec.backend,
            batch: spec.batch,
            m: spec.m,
            n: spec.n,
            k: spec.k,
            transposeA: spec.transposeA,
            transposeB: transposeBOverride,
            stridedBatch: spec.stridedBatch,
            accumulate: spec.accumulate,
            alpha: spec.alpha,
            beta: spec.beta,
            bias: spec.bias
        )
    }
    
    private func runGenericKernelInlined(
        originalSpec: MatmulShapeSpec,
        kernelSpec: MatmulShapeSpec,
        backendType: MatmulShapeSpec.Backend,
        variant: KernelVariant,
        functionName: String
    ) throws -> BenchmarkResult {
        guard let libraryName = variant.library else {
            throw HarnessError.libraryLoadFailed("No library specified for \(backendType.rawValue)/\(variant.name)")
        }
        guard let library = libraries[libraryName] else {
            throw HarnessError.libraryLoadFailed("\(libraryName).metallib")
        }
        
        let alpha = kernelSpec.alpha
        let beta = kernelSpec.beta
        
        let (function, specialization) = try prepareFunction(
            library: library,
            backendType: backendType,
            variant: variant,
            kernelSpec: kernelSpec,
            functionName: functionName,
            alpha: alpha,
            beta: beta
        )
        
        let pipeline: MTLComputePipelineState
        do {
            pipeline = try device.makeComputePipelineState(function: function)
        } catch {
            throw HarnessError.pipelineCreationFailed("\(functionName): \(error)")
        }
        
        let tensors = loadMatmulTensors(spec: kernelSpec)
        let mlxLaunch = specialization.map {
            prepareMLXLaunch(
                specialization: $0,
                spec: kernelSpec,
                tensors: tensors,
                alpha: alpha,
                beta: beta
            )
        }
        
        let dispatch: (MTLSize, MTLSize)
        if let launch = mlxLaunch {
            dispatch = (launch.threadgroups, launch.threadsPerThreadgroup)
        } else {
            dispatch = calculateDispatchForKernel(
                spec: kernelSpec,
                variant: variant,
                pipeline: pipeline,
                backendType: backendType
            )
        }
        
        var gpuTimings: [Double] = []
        var cpuTimings: [Double] = []
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
            
            if let launch = mlxLaunch {
                if launch.useOutSource {
                    encoder.setBuffer(tensors.output, offset: 0, index: 2)
                }
                encoder.setBuffer(tensors.output, offset: 0, index: 3)
                
                if kernelSpec.bias {
                    guard let biasBuffer = tensors.bias else {
                        throw HarnessError.pipelineCreationFailed("Missing bias buffer for \(variant.name)")
                    }
                    encoder.setBuffer(biasBuffer, offset: 0, index: 4)
                    
                    var paramsCopy = launch.params
                    encoder.setBytes(&paramsCopy, length: MemoryLayout<GEMMParams>.stride, index: 5)
                    
                    if var addmm = launch.addmmParams {
                        encoder.setBytes(&addmm, length: MemoryLayout<GEMMAddMMParams>.stride, index: 6)
                    }
                    
                    var batchShapeCopy = launch.batchShape
                    encoder.setBytes(&batchShapeCopy, length: MemoryLayout.size(ofValue: batchShapeCopy), index: 7)
                    
                    var batchStridesCopy = launch.batchStrides
                    encoder.setBytes(&batchStridesCopy, length: MemoryLayout.size(ofValue: batchStridesCopy), index: 8)
                } else {
                    var paramsCopy = launch.params
                    encoder.setBytes(&paramsCopy, length: MemoryLayout<GEMMParams>.stride, index: 4)
                    
                    if var addmm = launch.addmmParams {
                        encoder.setBytes(&addmm, length: MemoryLayout<GEMMAddMMParams>.stride, index: 5)
                    }
                    
                    var batchShapeCopy = launch.batchShape
                    encoder.setBytes(&batchShapeCopy, length: MemoryLayout.size(ofValue: batchShapeCopy), index: 6)
                    
                    var batchStridesCopy = launch.batchStrides
                    encoder.setBytes(&batchStridesCopy, length: MemoryLayout.size(ofValue: batchStridesCopy), index: 7)
                }
            } else if backendType == .gemv {
                guard !kernelSpec.bias else {
                    throw HarnessError.pipelineCreationFailed("GEMV kernels do not support bias add")
                }

                encoder.setBuffer(tensors.output, offset: 0, index: 2)

                let isSmallN = kernelSpec.n <= 16 && [1, 2, 4, 8, 16].contains(kernelSpec.n)
                if isSmallN {
                    var mParam = UInt32(kernelSpec.m)
                    var kParam = UInt32(kernelSpec.k)
                    encoder.setBytes(&mParam, length: MemoryLayout.size(ofValue: mParam), index: 3)
                    encoder.setBytes(&kParam, length: MemoryLayout.size(ofValue: kParam), index: 4)
                } else {
                    var params = GemvParams_f16(K: UInt32(kernelSpec.k), N: UInt32(kernelSpec.n))
                    encoder.setBytes(&params, length: MemoryLayout<GemvParams_f16>.size, index: 3)
                }
            } else {
                encoder.setBuffer(tensors.output, offset: 0, index: 2)
                
                var paramBufferIndex = 3
                if kernelSpec.bias {
                    guard let biasBuffer = tensors.bias else {
                        throw HarnessError.pipelineCreationFailed("Missing bias buffer for \(variant.name)")
                    }
                    encoder.setBuffer(biasBuffer, offset: 0, index: paramBufferIndex)
                    paramBufferIndex += 1
                }
                
                var mParam = Int32(kernelSpec.m)
                var nParam = Int32(kernelSpec.n)
                var kParam = Int32(kernelSpec.k)
                encoder.setBytes(&mParam, length: MemoryLayout.size(ofValue: mParam), index: paramBufferIndex)
                encoder.setBytes(&nParam, length: MemoryLayout.size(ofValue: nParam), index: paramBufferIndex + 1)
                encoder.setBytes(&kParam, length: MemoryLayout.size(ofValue: kParam), index: paramBufferIndex + 2)
            }
            
            encoder.dispatchThreadgroups(dispatch.0, threadsPerThreadgroup: dispatch.1)
            encoder.endEncoding()
            
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
        
        let safeMaxAbsError = validation.maxAbsError.isFinite ? validation.maxAbsError : Float.greatestFiniteMagnitude
        let safeMaxRelError = validation.maxRelError.isFinite ? validation.maxRelError : Float.greatestFiniteMagnitude
        
        return BenchmarkResult(
            spec: originalSpec,
            backend: backendType,
            variantName: variant.name,
            library: libraryName,
            isBaseline: variant.isBaseline,
            gpuTimingsMs: gpuTimings,
            cpuTimingsMs: cpuTimings,
            maxAbsError: safeMaxAbsError,
            maxRelError: safeMaxRelError
        )
    }
    
    private func calculateDispatchForKernel(spec: MatmulShapeSpec, variant: KernelVariant, pipeline: MTLComputePipelineState, backendType: MatmulShapeSpec.Backend) -> (MTLSize, MTLSize) {
        // Handle different backend types differently
        switch backendType {
        case .mlx:
            // For MLX-style kernels, use the original threadgroup calculation
            if let tileOverride = variant.tileOverride {
                let (tileBM, tileBN, _, wmSel, wnSel) = tileOverride
                let threadgroups = makeThreadgroups(spec: spec, tile: (tileBM, tileBN), constants: MLXFunctionConstants())
                let threadsPerTG = MTLSize(width: 32, height: wnSel, depth: wmSel)  // MLX uses 32-width with specific heights
                return (threadgroups, threadsPerTG)
            } else {
                // Fallback for MLX without tile override
                return calculateGenericDispatch(spec: spec, pipeline: pipeline)
            }
        case .m1Optimized, .m1OptimizedV2, .m1OptimizedV3, .m1OptimizedV4:
            // For M1 optimized kernels
            return calculateM1Dispatch(spec: spec, pipeline: pipeline, variant: variant)
        case .gemv:
            // For GEMV kernels
            return calculateGEMVDispatch(spec: spec, pipeline: pipeline)
        case .gemmTiled:
            // For GEMM Tiled kernels
            return calculateGEMMTiledDispatch(spec: spec, pipeline: pipeline)
        default:
            // Generic fallback
            return calculateGenericDispatch(spec: spec, pipeline: pipeline)
        }
    }
    
    private func calculateGenericDispatch(spec: MatmulShapeSpec, pipeline: MTLComputePipelineState) -> (MTLSize, MTLSize) {
        // Default threadgroup size based on pipeline characteristics
        let threadExecutionWidth = pipeline.threadExecutionWidth
        let maxThreadsPerGroup = pipeline.maxTotalThreadsPerThreadgroup
        let height = max(1, min(maxThreadsPerGroup / threadExecutionWidth, 8)) // Limit height to reasonable value
        
        let threadgroupSize = MTLSize(width: threadExecutionWidth, height: height, depth: 1)
        
        // Calculate grid size based on problem dimensions
        let widthGroups = (spec.n + 127) / 128  // 128 elements per group width
        let heightGroups = (spec.m + 127) / 128 // 128 elements per group height
        let depthGroups = spec.batch > 1 ? spec.batch : 1
        
        let threadgroups = MTLSize(width: widthGroups, height: heightGroups, depth: depthGroups)
        
        return (threadgroups, threadgroupSize)
    }
    
    private func calculateM1Dispatch(spec: MatmulShapeSpec, pipeline: MTLComputePipelineState, variant: KernelVariant) -> (MTLSize, MTLSize) {
        // Match v2 expectations; robust parsing for bnXX/tgYY only when followed by digits.
        func extractInt(after token: String, in name: String, default def: Int) -> Int {
            var found: Int? = nil
            var searchStart = name.startIndex
            while let r = name[searchStart...].range(of: token) {
                let start = r.upperBound
                if start < name.endIndex, name[start].isNumber {
                    var i = start
                    var digits = ""
                    while i < name.endIndex, name[i].isNumber {
                        digits.append(name[i])
                        i = name.index(after: i)
                    }
                    if let val = Int(digits) { found = val }
                }
                searchStart = name.index(after: r.lowerBound)
            }
            return found ?? def
        }

        let name = variant.name
        if name.contains("small_k") {
            let threadgroupSize = MTLSize(width: 32, height: 1, depth: 1)
            let threadgroups = MTLSize(width: (spec.n + 32 - 1) / 32, height: 1, depth: 1)
            return (threadgroups, threadgroupSize)
        }

        let cols = extractInt(after: "bn", in: name, default: (name.contains("bn64") ? 64 : 128))
        let tg   = extractInt(after: "tg", in: name, default: 128)

        let columnsPerTG = max(1, cols)
        let threadgroupWidth = max(1, tg)
        let threadgroupSize = MTLSize(width: threadgroupWidth, height: 1, depth: 1)
        let threadgroups = MTLSize(width: (spec.n + columnsPerTG - 1) / columnsPerTG, height: 1, depth: 1)
        return (threadgroups, threadgroupSize)
    }
    
    private func calculateGEMVDispatch(spec: MatmulShapeSpec, pipeline: MTLComputePipelineState) -> (MTLSize, MTLSize) {
        // GEMV typically dispatches based on N dimension for small N values
        let threadgroupSize: MTLSize
        let threadgroups: MTLSize
        
        if spec.n <= 16 && [1, 2, 4, 8, 16].contains(spec.n) {
            // For small N GEMV kernels
            let threadsPerGroup: Int
            let tgPerGridWidth: Int
            if spec.n == 1 {
                threadsPerGroup = 32
                tgPerGridWidth = (spec.m + 31) / 32
            } else if spec.n == 2 {
                threadsPerGroup = 32
                tgPerGridWidth = (spec.m + 31) / 32
            } else if spec.n == 4 {
                threadsPerGroup = 64  // 16 rows x 4 cols = 64 threads
                tgPerGridWidth = (spec.m + 15) / 16
            } else if spec.n == 8 {
                threadsPerGroup = 64  // 8 rows x 8 cols = 64 threads
                tgPerGridWidth = 1
            } else { // n == 16
                threadsPerGroup = 64  // 4 rows x 16 cols = 64 threads
                tgPerGridWidth = (spec.m + 3) / 4
            }
            
            threadgroupSize = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
            threadgroups = MTLSize(width: tgPerGridWidth, height: 1, depth: 1)
        } else {
            // For general GEMV kernel
            threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
            threadgroups = MTLSize(width: (spec.n + 255) / 256, height: 1, depth: 1)
        }
        
        return (threadgroups, threadgroupSize)
    }
    
    private func calculateGEMMTiledDispatch(spec: MatmulShapeSpec, pipeline: MTLComputePipelineState) -> (MTLSize, MTLSize) {
        // GEMM Tiled uses tile-based dispatch
        let tileM = min(spec.m, 64)  // tile size for M dimension
        let tileN = min(spec.n, 64)  // tile size for N dimension
        
        let threadgroups = MTLSize(
            width: (spec.n + tileN - 1) / tileN,
            height: (spec.m + tileM - 1) / tileM,
            depth: 1
        )
        
        // Use 64 threads per threadgroup for the tiled implementation
        let threadgroupSize = MTLSize(width: 64, height: 1, depth: 1)
        
        return (threadgroups, threadgroupSize)
    }
    
    private func determineFunctionName(spec: MatmulShapeSpec, variant: KernelVariant) -> String {
        func needsAccumulate(_ spec: MatmulShapeSpec) -> Bool {
            return spec.accumulate || abs(spec.alpha - 1.0) > Float.ulpOfOne || abs(spec.beta) > Float.ulpOfOne
        }
        
        guard let supports = variant.supports else {
            fatalError("CONFIGURATION ERROR: Missing supports metadata for variant '\(variant.name)' (backend: \(spec.backend.rawValue)).")
        }
        
        if spec.bias {
            if let biasName = supports.functionNameBias {
                return biasName
            }
            fatalError("CONFIGURATION ERROR: Spec requires bias but variant '\(variant.name)' (backend: \(spec.backend.rawValue)) does not provide functionNameBias.")
        }
        
        if needsAccumulate(spec), let accumulateName = supports.functionNameAccumulate {
            return accumulateName
        }
        
        return supports.functionName
    }

    private func prepareFunction(
        library: MTLLibrary,
        backendType: MatmulShapeSpec.Backend,
        variant: KernelVariant,
        kernelSpec: MatmulShapeSpec,
        functionName: String,
        alpha: Float,
        beta: Float
    ) throws -> (MTLFunction, MLXSpecialization?) {
        switch backendType {
        case .mlx:
            let effectiveTile = variant.tileOverride ?? selectTile(m: kernelSpec.m, n: kernelSpec.n)
            let constants = makeMLXConstants(
                spec: kernelSpec,
                tile: (effectiveTile.0, effectiveTile.1, effectiveTile.2),
                alpha: alpha,
                beta: beta
            )
            
            let functionConstants = makeFunctionConstantValues(from: constants, gatherBias: false)
            let function = try library.makeFunction(name: functionName, constantValues: functionConstants)
            let specialization = MLXSpecialization(tile: effectiveTile, constants: constants)
            return (function, specialization)
        default:
            guard let function = library.makeFunction(name: functionName) else {
                throw HarnessError.functionMissing(functionName)
            }
            return (function, nil)
        }
    }
    
    private func makeFunctionConstantValues(
        from constants: MLXFunctionConstants,
        gatherBias: Bool
    ) -> MTLFunctionConstantValues {
        let functionConstants = MTLFunctionConstantValues()
        var hasBatch = constants.hasBatch
        var useOutSource = constants.useOutSource
        var doAxpby = constants.doAxpby
        var doBiasAdd = constants.doBiasAdd
        var alignM = constants.alignM
        var alignN = constants.alignN
        var alignK = constants.alignK
        var gatherBiasValue = gatherBias
        
        functionConstants.setConstantValue(&hasBatch, type: .bool, index: 10)
        functionConstants.setConstantValue(&useOutSource, type: .bool, index: 100)
        functionConstants.setConstantValue(&doAxpby, type: .bool, index: 110)
        functionConstants.setConstantValue(&doBiasAdd, type: .bool, index: 120)
        functionConstants.setConstantValue(&alignM, type: .bool, index: 200)
        functionConstants.setConstantValue(&alignN, type: .bool, index: 201)
        functionConstants.setConstantValue(&alignK, type: .bool, index: 202)
        functionConstants.setConstantValue(&gatherBiasValue, type: .bool, index: 300)
        return functionConstants
    }
    
    private func prepareMLXLaunch(
        specialization: MLXSpecialization,
        spec: MatmulShapeSpec,
        tensors: MatmulTensors,
        alpha: Float,
        beta: Float
    ) -> MLXLaunchContext {
        let tile = specialization.tile
        let constants = specialization.constants
        
        var params = buildGEMMParams(
            spec: spec,
            tile: (tile.0, tile.1, tile.2),
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
        
        let threadgroups = makeThreadgroups(spec: spec, tile: (tile.0, tile.1), constants: constants)
        let threadsPerThreadgroup = MTLSize(width: 32, height: tile.4, depth: tile.3)
        
        return MLXLaunchContext(
            params: params,
            addmmParams: addmmParams,
            batchShape: batchShape,
            batchStrides: batchStrides,
            threadgroups: threadgroups,
            threadsPerThreadgroup: threadsPerThreadgroup,
            useOutSource: constants.useOutSource
        )
    }
    
    private func getM1KernelFunctionName(variant: KernelVariant) -> String {
        // Handle M1 optimized naming
        switch variant.name {
        case "simd_basic":
            return "m1_dot_product_simd"
        case "simd_tiled_1":
            return "m1_dot_product_tiled_simd1"
        case "simd_tiled_2":
            return "m1_dot_product_tiled_simd2"
        case "simd_tiled_4":
            return "m1_dot_product_tiled_simd4"
        case "basic":
            return "m1_dot_product_v2_basic"
        case "tiled_1":
            return "m1_dot_product_v2_tiled1"
        case "tiled_2":
            return "m1_dot_product_v2_tiled2"
        case "tiled_4":
            return "m1_dot_product_v2_tiled4"
        case "simd_naive":
            return "m1_dot_product_v2_simd_naive"
        case "small_k":
            return "m1_dot_product_v2_small_k"
        default:
            return variant.name.replacingOccurrences(of: " ", with: "_").lowercased()
        }
    }
    
    private func getGEMVFunctionName(spec: MatmulShapeSpec) -> String {
        // Handle GEMV naming based on N dimension
        if [1, 2, 4, 8, 16].contains(spec.n) {
            return "gemv_n\(spec.n)_f16"
        } else {
            return "gemv_f16"
        }
    }
    
    private func getGEMMTiledFunctionName() -> String {
        return "gemm_tiled_f16"
    }
    
    func validateVariant(variant: KernelVariant) throws {
        // Basic validation that library exists if required
        if variant.library != nil && libraries[variant.library!] == nil {
            throw HarnessError.libraryLoadFailed("Library \(variant.library!) not loaded")
        }
    }
}
