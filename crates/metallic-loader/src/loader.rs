use std::path::Path;

#[cfg(feature = "gguf")]
use crate::gguf::file::GGUFFile;
#[cfg(feature = "gguf")]
use crate::gguf::model_loader::GGUFModelLoader;
use crate::{LoadedModel, LoaderError};

/// A registry for model loaders.
/// Allows registering loaders for specific file extensions or formats.
pub struct ModelLoader;

impl ModelLoader {
    /// Load a model from a file path.
    /// It attempts to detect the format based on extension.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Box<dyn LoadedModel>, LoaderError> {
        let path = path.as_ref();
        let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();

        match extension.as_str() {
            "gguf" => {
                #[cfg(feature = "gguf")]
                {
                    let file = GGUFFile::load_mmap_and_get_metadata(path).map_err(|e| LoaderError::Io(std::io::Error::other(e)))?;
                    let loader = GGUFModelLoader::new(file);
                    let model = loader.load_model().map_err(|e| LoaderError::Io(std::io::Error::other(e)))?;
                    Ok(Box::new(model))
                }
                #[cfg(not(feature = "gguf"))]
                {
                    Err(LoaderError::FeatureNotEnabled("GGUF support is not enabled".to_string()))
                }
            }
            _ => Err(LoaderError::InvalidData(format!("Unsupported model extension: {extension}"))),
        }
    }
}
