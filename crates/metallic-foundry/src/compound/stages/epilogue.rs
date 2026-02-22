//! EpilogueStage - Adapts Epilogue into a post-processing Stage.

use std::marker::PhantomData;

use crate::{
    compound::{BufferArg, Stage}, fusion::Epilogue
};

/// Adapts an Epilogue into a post-processing Stage.
#[derive(Clone)]
pub struct EpilogueStage<E: Epilogue> {
    _epilogue: PhantomData<E>,
}

impl<E: Epilogue + Default> EpilogueStage<E> {
    pub fn new() -> Self {
        Self { _epilogue: PhantomData }
    }
}

impl<E: Epilogue + Default> Default for EpilogueStage<E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<E: Epilogue + Default + 'static> Stage for EpilogueStage<E> {
    fn includes(&self) -> Vec<&'static str> {
        vec![E::default().header()]
    }

    fn buffer_args(&self) -> Vec<BufferArg> {
        vec![] // Epilogues typically don't add buffer args
    }

    fn emit(&self, input_var: &str) -> (String, String) {
        let epilogue = E::default();
        let struct_name = epilogue.struct_name();

        let code = if struct_name == "EpilogueNone" {
            format!("    half epilogue_out = {};", input_var)
        } else {
            format!("    half epilogue_out = {}::apply({});", struct_name, input_var)
        };

        ("epilogue_out".to_string(), code)
    }
}
