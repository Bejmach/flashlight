use flashlight::{layers::Layer, prelude::*};
#[allow(unused)]
use flashlight_tensor::prelude::*;

use rand::prelude::*;
use std::time::Instant;

pub struct NewModel{
    linear1: Linear,
    linear2: Linear,
    activation: Relu,
    output_activation: Sigmoid,
}



impl NewModel{
    fn new() -> Self{
        Self{
            linear1: Linear::new(2, 3, 0.01, xavier_weights(2, 1)),
            linear2: Linear::new(3, 1, 0.01, xavier_weights(2, 1)),
            activation: Relu::new(),
            output_activation: Sigmoid::new(),
        }
    }
    fn grad_output(&self, target: &Tensor<f32>) -> Tensor<f32>{
        self.output_activation.grad_output(target)
    }
}

impl Model for NewModel{
    fn forward(&mut self, input: Tensor<f32>) -> Tensor<f32> {
        let x = self.linear1.forward(&input);
        let x = self.activation.forward(&x);
        let x = self.linear2.forward(&x);
        
        self.output_activation.forward(&x)
    }
    fn backward(&mut self, grad_output: Tensor<f32>) {
        let x = self.output_activation.backward(&grad_output);

        let x = self.linear2.backward(&x);
        let x = self.activation.backward(&x);
        self.linear1.backward(&x);
    }
}

fn main() {
    let input: Tensor<f32> = Tensor::from_data(&[50.0, 70.0, 20.0, 90.0, 99.0, 1.0], &[3, 2]).unwrap().matrix_transpose().unwrap();
    let target: Tensor<f32> = Tensor::from_data(&[0.0, 0.0, 1.0], &[1, 3]).unwrap();

    let mut model: NewModel = NewModel::new();

    let number_of_epochs = 1000000;

    for epoch in 0..=number_of_epochs{
        let output = model.forward(input.clone());

        println!("output:\n{}\n\ntarget:\n{}\n", output.matrix_to_string().unwrap(), target.matrix_to_string().unwrap());

        println!("Epoch {}, Cost: {}", epoch, cross_entropy_cost(&output, &target).unwrap());

        let grad_output = model.grad_output(&target);

        println!("gradient: \n{}\n\n", grad_output.matrix_to_string().unwrap());

        model.backward(grad_output);
    }
}
