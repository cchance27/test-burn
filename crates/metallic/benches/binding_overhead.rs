use std::{collections::HashMap, time::Instant};

fn main() {
    let mut bindings = HashMap::new();
    let n_layers = 24;
    let seq_len = 1000;

    // Simulate prepare_bindings
    for i in 0..n_layers {
        bindings.insert(format!("layer.attn_q_{}", i), 0);
        bindings.insert(format!("layer.attn_k_{}", i), 0);
        bindings.insert(format!("layer.attn_v_{}", i), 0);
        bindings.insert(format!("layer.attn_out_{}", i), 0);
    }

    let mut scope = HashMap::new();

    let start = Instant::now();
    let mut sum = 0;

    for _token in 0..seq_len {
        for i in 0..n_layers {
            scope.insert("i".to_string(), i.to_string());

            // Simulate INTERPOLATION and LOOKUP for 4 tensors per layer
            let keys = [
                format!("layer.attn_q_{{i}}"),
                format!("layer.attn_k_{{i}}"),
                format!("layer.attn_v_{{i}}"),
                format!("layer.attn_out_{{i}}"),
            ];

            for key in keys {
                let resolved = interpolate(&key, &scope);
                sum += bindings.get(&resolved).unwrap_or(&0);
            }
        }
    }

    println!("String path: {:?} for {} iterations", start.elapsed(), seq_len * n_layers * 4);

    // Vec path
    let vec_bindings = vec![0; n_layers * 4];
    let start = Instant::now();
    for _token in 0..seq_len {
        for i in 0..n_layers {
            // Simulate COMPILED INDICES (pre-resolved)
            let indices = [i * 4 + 0, i * 4 + 1, i * 4 + 2, i * 4 + 3];
            for idx in indices {
                sum += vec_bindings[idx];
            }
        }
    }
    let _ = sum;

    println!("Vec path:    {:?}", start.elapsed());
}

fn interpolate(s: &str, scope: &HashMap<String, String>) -> String {
    let mut res = s.to_string();
    if let Some(start) = res.find('{') {
        if let Some(end) = res[start..].find('}') {
            let key = &res[start + 1..start + end];
            if let Some(val) = scope.get(key) {
                res.replace_range(start..=start + end, val);
            }
        }
    }
    res
}
