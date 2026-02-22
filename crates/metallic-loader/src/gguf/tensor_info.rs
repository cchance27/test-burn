#[derive(Debug, Clone)]
pub struct GGUTensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub data_type: super::GGUFDataType,
    pub offset: u64,
}
