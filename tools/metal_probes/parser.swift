import Foundation

class MatmulDocParser {
    func parseSpecs(markdownPath: String) throws -> [MatmulShapeSpec] {
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

    private func parseShapeHeader(line: String, followingLines: [String]) throws -> MatmulShapeSpec {
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
        // Infer bias=true if the op name indicates a bias epilogue (e.g., matmul_bias_add)
        let opLower = op.lowercased()
        let impliedBias = opLower.contains("bias")
        let biasExplicit = (entries["bias"] ?? "0") != "0"
        let bias = biasExplicit || impliedBias

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
    
    private func parseBackend(from lines: [String]) throws -> MatmulShapeSpec.Backend {
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
