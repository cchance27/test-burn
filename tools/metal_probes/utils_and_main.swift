import Foundation

private let outputSupportsColor: Bool = {
    if ProcessInfo.processInfo.environment["NO_COLOR"] != nil {
        return false
    }
    return isatty(fileno(stdout)) != 0
}()

private func colorize(_ text: String, code: String) -> String {
    guard outputSupportsColor else { return text }
    return "\u{001B}[\(code)m\(text)\u{001B}[0m"
}

private func formatTag(_ tag: String) -> String {
    switch tag {
    case "baseline":
        return colorize("[baseline]", code: "33;1")
    case "best-gpu":
        return colorize("[best-gpu]", code: "32;1")
    default:
        return "[\(tag)]"
    }
}

// Global variable to hold the original profiling data
var originalProfilingData: [String: OriginalProfileData] = [:]

func summarize(specs: [MatmulShapeSpec], results: [BenchmarkResult], failures: [VariantFailure]) {
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
    
    if results.isEmpty {
        print("No benchmark results to display.")
        return
    }

    let iterationCount = results.first?.gpuTimingsMs.count ?? 0
    let grouped = Dictionary(grouping: results, by: { $0.spec })

    print("Matmul benchmark summary (iterations=\(iterationCount))")

    for spec in specs {
        guard let specResults = grouped[spec], !specResults.isEmpty else {
            let tag = "op=\(spec.op) batch=\(spec.batch) m=\(spec.m) n=\(spec.n) k=\(spec.k) tA=\(spec.transposeA ? 1 : 0) tB=\(spec.transposeB ? 1 : 0)"
            print("Spec \(tag): no successful variants")
            continue
        }

        let key = makeProfileKey(spec: spec)
        let originalData = originalProfilingData[key] ?? OriginalProfileData()

        let tag = "op=\(spec.op) | batch=\(spec.batch) | m=\(spec.m) | n=\(spec.n) | k=\(spec.k) | tA=\(spec.transposeA ? 1 : 0) | tB=\(spec.transposeB ? 1 : 0)"
        print("Spec \(tag):")

        let bestCpu = specResults.min(by: { ($0.averageCpuMs ?? .greatestFiniteMagnitude) < ($1.averageCpuMs ?? .greatestFiniteMagnitude) })
        if let bestCpu = bestCpu, let avgCpu = bestCpu.averageCpuMs, avgCpu.isFinite {
            let line = "  best_avg_cpu: \(bestCpu.backend.rawValue)/\(bestCpu.variantName) = \(formatTime(avgCpu))ms"
            print(colorize(line, code: "32;1"))
        }

        let bestGpu = specResults.min(by: { lhs, rhs in
            (lhs.averageGpuMs ?? .greatestFiniteMagnitude) < (rhs.averageGpuMs ?? .greatestFiniteMagnitude)
        })

        if let bestGpu = bestGpu, let avgGpu = bestGpu.averageGpuMs, avgGpu.isFinite {
            let line = "  best_avg_gpu: \(bestGpu.backend.rawValue)/\(bestGpu.variantName) = \(formatTime(avgGpu))ms"
            print(colorize(line, code: "32;1"))
        }

        let sortedVariants = specResults.sorted { lhs, rhs in
            let lhsIndex = VariantManager.backendDisplayOrder.firstIndex(of: lhs.backend) ?? Int.max
            let rhsIndex = VariantManager.backendDisplayOrder.firstIndex(of: rhs.backend) ?? Int.max
            if lhsIndex != rhsIndex {
                return lhsIndex < rhsIndex
            }
            if lhs.variantName != rhs.variantName {
                return lhs.variantName < rhs.variantName
            }
            // Stable fallback on CPU average to keep deterministic ordering even when names match
            return (lhs.averageCpuMs ?? .greatestFiniteMagnitude) < (rhs.averageCpuMs ?? .greatestFiniteMagnitude)
        }

        for result in sortedVariants {
            let avgGpuStr = result.averageGpuMs.map { "\(formatTime($0))ms" } ?? "N/A"
            let avgCpuStr = result.averageCpuMs.map { "\(formatTime($0))ms" } ?? "N/A"
            let maxAbsStr = String(format: "%.4e", result.maxAbsError)
            let maxRelStr = String(format: "%.4e", result.maxRelError)
            var tags: [String] = []
            if result.isBaseline && result.backend == spec.backend {
                tags.append("baseline")
            }
            if let bestGpu = bestGpu, bestGpu.backend == result.backend, bestGpu.variantName == result.variantName {
                tags.append("best-gpu")
            }
            let tagSuffix: String
            if tags.isEmpty {
                tagSuffix = ""
            } else {
                let colored = tags.map(formatTag).joined(separator: " ")
                tagSuffix = " " + colored
            }

            var comparisonStr = ""
            if originalData.avgTimeMs > 0, let avgCpu = result.averageCpuMs {
                let percentChange = ((avgCpu - originalData.avgTimeMs) / originalData.avgTimeMs) * 100.0
                let sign = percentChange >= 0 ? "+" : ""
                comparisonStr = " [vs orig \(formatTime(originalData.avgTimeMs))ms] (\(sign)\(String(format: "%.2f", percentChange))%)"
            } else {
                comparisonStr = " [vs orig N/A]"
            }

            print("   - \(result.backend.rawValue)/\(result.variantName)\(tagSuffix): avg_gpu=\(avgGpuStr) | avg_cpu=\(avgCpuStr)\(comparisonStr) maxAbs=\(maxAbsStr) maxRel=\(maxRelStr)")
        }
    }

    if !failures.isEmpty {
        print("\nVariant failures:")
        for failure in failures {
            let spec = failure.spec
            let specKey = "op=\(spec.op) batch=\(spec.batch) m=\(spec.m) n=\(spec.n) k=\(spec.k)"
            print(" - \(failure.backend.rawValue)/\(failure.variantName) @ \(specKey): \(failure.errorDescription)")
        }
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
        let parser = MatmulDocParser()
        let specs = try parser.parseSpecs(markdownPath: markdownPath)
        let harness = try MatmulHarness(buildDir: buildDir, matmulDir: matmulDir)
        let results = try harness.run(specs: specs)
        summarize(specs: specs, results: results, failures: harness.variantFailures)
    } catch {
        print("Error: \(error)")
        return 1
    }

    return 0
}

exit(main())