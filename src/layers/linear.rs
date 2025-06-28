use std::{marker::PhantomData, panic};

use flashlight_tensor::prelude::*;
use rand::Rng;
use async_trait::async_trait;

use crate::{layers::{LayerCpu, LayerGpu}, prelude::xavier_weights};

pub trait Backend{
    fn forward(&mut self, input: Tensor<f32>){}
    fn backward(&mut self, grad_output: Tensor<f32>){}
}

pub struct Cpu;
pub struct Gpu;

pub trait Dtype: Copy + 'static {
    fn from_f32(f: f32) -> Self;
}
impl Dtype for f32{
    fn from_f32(f: f32) -> Self {
        f
    }
}


/// Linear layer for neural network
pub struct Linear<B>{
    pub weights: Tensor<f32>,
    pub biases: Tensor<f32>,
    learning_rate: f32,
    input_cache: Option<Tensor<f32>>,

    forward_runner: Option<GpuRunner>,

    backward_weights_runner: Option<GpuRunner>,
    backward_bias_runner: Option<GpuRunner>,
    backward_runner: Option<GpuRunner>,

    _marker: std::marker::PhantomData<B>,
}

impl<B> Linear<B>{
    
    /// Create new linear layer using input_size, output_size and learning_rate
    ///
    /// # Example
    ///
    /// ```
    /// use flashlight::prelude::*;
    ///
    /// let linear = Linear::new(2, 1, 0.1);
    /// ```

     pub fn new(input_size: u32, output_size: u32, learning_rate: f32) -> Self{
        let rand_range = xavier_weights(input_size, output_size);

        Self{
            weights: Tensor::rand(rand_range, &[output_size, input_size]),
            biases: Tensor::rand(rand_range, &[output_size, 1]),
            learning_rate,
            input_cache: None,

            forward_runner: None,

            backward_weights_runner: None,
            backward_bias_runner: None,
            backward_runner: None,

            _marker: PhantomData,
        }
    }

    pub fn clear(&mut self){
        self.forward_runner = None;
        self.backward_weights_runner = None;
        self.backward_bias_runner = None;
        self.backward_runner = None;
    }
}

impl LayerCpu for Linear<Cpu>{
   /// Forward propagation for linear layer. returns Tensor<f32>
   /// does not support included activations
    fn forward(&mut self, data: &Tensor<f32>) -> Tensor<f32>{

        self.input_cache = Some(data.clone());

        let weights_data = self.weights.matrix_mul(&data).unwrap();

        let bias_data = weights_data.tens_broadcast_add(&self.biases).unwrap();

        bias_data
    }
    /// Backward propagation for linear laer. Returns Tensor<f32> that is a partial derivative to
    /// next layer/activation to this layer
    /// does not support included activations
    fn backward(&mut self, grad_output: &Tensor<f32>) -> Tensor<f32>{
        if self.input_cache.is_none(){
            panic!("Program paniced!\nNo input cache in layer. Perform a forward pass before backward");
        }

        let weight_grad = grad_output.matrix_mul(&self.input_cache.clone().unwrap().matrix_transpose().unwrap()).unwrap();
    
        let bias_grad = grad_output.matrix_col_sum().unwrap().mul(1.0 / self.input_cache.clone().unwrap().get_shape()[0] as f32);

        self.weights = self.weights.tens_sub(&weight_grad.mul(self.learning_rate)).unwrap();
        self.biases = self.biases.tens_sub(&bias_grad.mul(self.learning_rate)).unwrap();

        let input_grad = self.weights.matrix_transpose().unwrap().matrix_mul(grad_output).unwrap();

        input_grad
    }
}

#[async_trait]
impl LayerGpu for Linear<Gpu>{
   /// Forward propagation for linear layer. returns Tensor<f32>
    async fn forward(&mut self, data: &Tensor<f32>) -> Tensor<f32>{

        self.input_cache = Some(data.clone());

        let sample = Sample::from_data(vec!{self.weights.clone(), data.clone()}, vec!{}, &[]);

        if(self.forward_runner.is_none()){
            self.forward_runner = Some(GpuRunner::init(2, MemoryMetric::GB));
        }

        let runner = self.forward_runner.as_mut().unwrap();
        runner.append(sample);

        let output_data = runner.forward_no_activ().await;

        runner.clear();

        output_data[0].clone()
    }
    /// Backward propagation for linear laer. Returns Tensor<f32> that is a partial derivative to
    /// next layer/activation to this layer
    async fn backward(&mut self, grad_output: &Tensor<f32>) -> Tensor<f32>{
        if self.input_cache.is_none(){
            panic!("Program paniced!\nNo input cache in layer. Perform a forward pass before backward");
        }

        if self.backward_runner.is_none(){
            self.backward_runner = Some(GpuRunner::init(2, MemoryMetric::GB));
        }
        if self.backward_weights_runner.is_none(){
            self.backward_weights_runner = Some(GpuRunner::init(2, MemoryMetric::GB));
        }
        if self.backward_bias_runner.is_none(){
            self.backward_bias_runner = Some(GpuRunner::init(2, MemoryMetric::GB));
        }

        let bias_runner = self.backward_bias_runner.as_mut().unwrap();
        let weight_runner = self.backward_weights_runner.as_mut().unwrap();
        let grad_runner = self.backward_runner.as_mut().unwrap();

        let bias_sample = Sample::from_data(vec!{self.biases.clone(), grad_output.clone(), self.input_cache.clone().unwrap()}, vec!{self.learning_rate.clone()}, &[]);
        let weight_sample = Sample::from_data(vec!{self.weights.clone(), grad_output.clone(), self.input_cache.clone().unwrap()}, vec!{self.learning_rate.clone()}, &[]);
        let grad_sample = Sample::from_data(vec!{self.weights.clone(), grad_output.clone()}, vec!{}, &[]);
        
        bias_runner.append(bias_sample);
        weight_runner.append(weight_sample);
        grad_runner.append(grad_sample);

        let bias_output = bias_runner.backward_bias().await;
        let weight_output = weight_runner.backward_weight().await;
        let grad_output = grad_runner.backward_grad().await;

        bias_runner.clear();
        weight_runner.clear();
        grad_runner.clear();

        self.biases = bias_output[0].clone();
        self.weights = weight_output[0].clone();

        grad_output[0].clone()
    }
}
