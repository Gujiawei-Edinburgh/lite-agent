use std::future::Future;
use std::pin::Pin;

pub use lite_agent_kernel::model::{
    FunctionSpec, ModelFunctionCall, ModelRequest, ModelResponse, ModelStreamEvent,
};

pub type ModelStreamHandler<'a> = dyn FnMut(ModelStreamEvent) + Send + 'a;

pub trait ModelClient: Send + Sync {
    fn stream_complete<'a>(
        &'a self,
        request: ModelRequest,
        on_event: &'a mut ModelStreamHandler<'a>,
    ) -> Pin<Box<dyn Future<Output = crate::Result<ModelResponse>> + Send + 'a>>;
}
