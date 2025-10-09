
/// Running average calculator for latency metrics
const RUNNING_AVERAGE_WINDOW: usize = 10;

#[derive(Clone)]
pub struct RunningAverage {
    values: [f64; RUNNING_AVERAGE_WINDOW],
    next_index: usize,
    initialized: bool,
}

impl Default for RunningAverage {
    fn default() -> Self {
        Self {
            values: [0.0; RUNNING_AVERAGE_WINDOW],
            next_index: 0,
            initialized: false,
        }
    }
}

impl RunningAverage {
    pub fn record(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }

        if !self.initialized {
            self.values = [value; RUNNING_AVERAGE_WINDOW];
            self.initialized = true;
            self.next_index = 1 % RUNNING_AVERAGE_WINDOW;
        } else {
            self.values[self.next_index] = value;
            self.next_index = (self.next_index + 1) % RUNNING_AVERAGE_WINDOW;
        }
    }

    pub fn average(&self) -> f64 {
        if !self.initialized {
            0.0
        } else {
            self.values.iter().sum::<f64>() / RUNNING_AVERAGE_WINDOW as f64
        }
    }

    pub fn has_samples(&self) -> bool {
        self.initialized
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}
