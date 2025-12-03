use std::marker::PhantomData;

use flashlight_tensor::{prelude::{GpuRunner, MemoryMetric, Sample}, tensor::Tensor};

use crate::layers::{LayerCpu, LayerGpu, Backend, Cpu, Gpu};
use async_trait::async_trait;

/// ReLU activation layer
pub struct Relu<B>{
    //change it to Opt<Vec<Ten<f32>>> to store all activations in layer, and use last element in
    //backprop, and pop it.
    input_cache: Vec<Tensor<f32>>,

    forward_runner: Option<GpuRunner>,
    backward_runner: Option<GpuRunner>,

    _marker: std::marker::PhantomData<B>,
}

impl<B> Relu<B>{
    /// Create new relu activation layer with its own input_cache
    pub fn new() -> Self{
        Self{
            input_cache: Vec::new(),

            forward_runner: None,
            backward_runner: None,

            _marker: PhantomData,
        }
    }
}

impl LayerCpu for Relu<Cpu>{
    /// Forward propagation for sigmoid layer
    fn forward(&mut self, input: &Tensor<f32>) -> Tensor<f32> {
        self.input_cache.push(input.clone());
        input.relu()
    }
    /// Backpropagation for sigmoid layer, uses grad_output
    fn backward(&mut self, grad_output: &Tensor<f32>) -> Tensor<f32> {
        if self.input_cache.is_empty(){
            panic!();
        }

        self.input_cache.pop().unwrap().relu_der().tens_broadcast_mul(grad_output).unwrap()
    }
}

#[async_trait]
impl LayerGpu for Relu<Gpu>{
    async fn forward(&mut self, input: &Tensor<f32>) -> Tensor<f32>{
        self.input_cache.push(input.clone());

        let sample = Sample::from_data(vec!{input.clone()}, vec!{}, &[]);

        if(self.forward_runner.is_none()){
            self.forward_runner = Some(GpuRunner::init(2, MemoryMetric::GB));
        }

        let runner = self.forward_runner.as_mut().unwrap();
        runner.append(sample);

        let output_data = runner.relu().await;

        runner.clear();

        output_data[0].clone()
    }
    async fn backward(&mut self, grad_output: &Tensor<f32>) -> Tensor<f32> {
        if self.input_cache.is_empty(){
            panic!();
        }
        
        // add real gpu relu backprop
        self.input_cache.pop().unwrap().relu_der().tens_broadcast_mul(grad_output).unwrap()
    }
}
