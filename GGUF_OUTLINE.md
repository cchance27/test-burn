# GGUF Module Documentation

This document outlines all the primitives and modules that the GGUF module implements and what it currently supports.

## Core Primitives

### gguf::GGUFFile
Main struct for loading and managing GGUF files. Contains header information, metadata, tensor information, and memory-mapped data.

### gguf::GGUFHeader
Structure representing the GGUF file header with magic number, version, tensor count, and metadata count.

### gguf::GGUFMetadata
Container for GGUF file metadata entries stored in a HashMap.

### gguf::GGUTensorInfo
Information about a tensor in the GGUF file including name, dimensions, data type, and offset.

### gguf::GGUFValue
Enum representing different types of values that can be stored in GGUF metadata.

### gguf::GGUFDataType
Enum representing the various data types supported in GGUF files including floating point, integer, boolean, string, array, and quantized types.

### gguf::GGUFError
Error type for GGUF operations including IO errors, invalid data, unsupported versions, and tensor-related errors.

## Core Modules

### gguf::quant
Quantization support module for GGUF format.

#### gguf::quant::q8
Support for Q8 quantization formats (Q8_0 and Q8_1).
- `dequantize_q8_to_f32` - Dequantize Q8_0/Q8_1 tensor data to F32
- `debug_q8_format` - Temporary function to debug Q8 format

#### gguf::quant::q8_simd
SIMD-optimized Q8 quantization support for GGUF format.
- `dequantize_q8_to_f32_simd` - Dequantize Q8_0/Q8_1 tensor data to F32 using SIMD optimization

### gguf::model_loader
Model loader that can construct a Metallic model from GGUF tensors.
- `GGUFModelLoader` - Main model loader struct
- `GGUFModel` - Model loaded from a GGUF file with tensor access and metadata queries

## Key Functions

### gguf::GGUFFile::load
Load a GGUF file from a path and parse its contents into memory.

### gguf::GGUFFile::get_tensor_data
Retrieve raw tensor data from the memory-mapped file.

### gguf::GGUFFile::calculate_actual_tensor_size
Calculate the actual size of tensor data in bytes, accounting for quantization formats.

### Tensor Conversion
Implementation of `TryFrom<(&GGUFFile, &GGUTensorInfo)>` for Metallic `Tensor` that handles conversion of GGUF tensors to Metallic tensors, including dequantization of Q8 formats.

### gguf::model_loader::GGUFModelLoader::load_model
Load a model from the GGUF file into Metallic tensors.

### gguf::model_loader::GGUFModel::get_tensor
Retrieve a tensor by name from the loaded model.

### gguf::model_loader::GGUFModel::get_architecture
Get model architecture from metadata.

### gguf::model_loader::GGUFModel::get_context_length
Get context length from metadata.

## Supported Data Types

- F64, F32, F16, BF16 (floating point)
- Q8_0, Q8_1 (8-bit quantized)
- Q4_0, Q4_1, Q5_0, Q5_1 (4 and 5-bit quantized)
- Q2K, Q3K, Q4K, Q5K, Q6K, Q8K (K-quantized variants)
- I8, I16, I32, I64 (signed integers)
- U8, U16, U32, U64 (unsigned integers)
- Bool, String, Array (other types)