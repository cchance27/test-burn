use super::RunningAverage;

#[derive(Clone)]
pub struct HierarchicalMetric {
    pub label: String,
    pub last_ms: f64,
    pub running_average: RunningAverage,
    pub children: Vec<HierarchicalMetric>,
    /// Optional metadata for display (e.g., batch_size)
    pub metadata: Option<std::collections::HashMap<String, String>>,
}

impl HierarchicalMetric {
    pub fn new(label: String, last_ms: f64) -> Self {
        let mut running_average = RunningAverage::default();
        running_average.record(last_ms);
        Self {
            label,
            last_ms,
            running_average,
            children: Vec::new(),
            metadata: None,
        }
    }

    pub fn with_metadata(label: String, last_ms: f64, metadata: Option<std::collections::HashMap<String, String>>) -> Self {
        let mut running_average = RunningAverage::default();
        running_average.record(last_ms);
        Self {
            label,
            last_ms,
            running_average,
            children: Vec::new(),
            metadata,
        }
    }

    pub fn ensure_child(&mut self, label: &str) -> &mut HierarchicalMetric {
        if let Some(position) = self.children.iter().position(|child| child.label == label) {
            &mut self.children[position]
        } else {
            self.children.push(HierarchicalMetric::new(label.to_string(), 0.0));
            self.children
                .last_mut()
                .expect("children vector cannot be empty immediately after push")
        }
    }

    pub fn upsert_path(&mut self, path: &[&str], last_ms: f64) {
        if path.is_empty() {
            return;
        }

        let label = path[0];
        let child = self.ensure_child(label);

        if path.len() == 1 {
            child.last_ms = last_ms;
            child.running_average.record(last_ms);
        } else {
            child.upsert_path(&path[1..], last_ms);
        }
    }

    pub fn upsert_path_with_metadata(&mut self, path: &[&str], last_ms: f64, metadata: Option<std::collections::HashMap<String, String>>) {
        if path.is_empty() {
            return;
        }

        let label = path[0];
        let child = self.ensure_child(label);

        if path.len() == 1 {
            child.last_ms = last_ms;
            child.running_average.record(last_ms);
            child.metadata = metadata;
        } else {
            child.upsert_path_with_metadata(&path[1..], last_ms, metadata);
        }
    }

    // Calculate inclusive timing (the time for this node plus all its descendants)
    // without modifying the stored values
    pub fn get_inclusive_timing(&self) -> (f64, f64) {
        let mut child_last_total = 0.0;
        let mut child_average_total = 0.0;

        for child in &self.children {
            let (child_last, child_avg) = child.get_inclusive_timing();
            child_last_total += child_last;
            child_average_total += child_avg;
        }

        let self_avg = self.running_average.average();

        let last_ms_total = if self.last_ms > 0.0 {
            self.last_ms.max(child_last_total)
        } else {
            child_last_total
        };

        let average_ms_total = if self_avg > 0.0 {
            self_avg.max(child_average_total)
        } else {
            child_average_total
        };

        (last_ms_total, average_ms_total)
    }
}
