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
            linear1: Linear::new(2, 16, 0.01),
            linear2: Linear::new(16, 1, 0.01),
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
    let mut pre_prepared: InputPrePrepared = InputPrePrepared::new(&Tensor::from_data(&[50.0, 70.0], &[1, 2]).unwrap(), &Tensor::from_data(&[0.0], &[1, 1]).unwrap());
    pre_prepared.append(&Tensor::from_data(&[-100.0, 100.0], &[1, 2]).unwrap(), &Tensor::from_data(&[0.0], &[1, 1]).unwrap());
    pre_prepared.append(&Tensor::from_data(&[100.0, -100.0], &[1, 2]).unwrap(), &Tensor::from_data(&[1.0], &[1, 1]).unwrap());

    pre_prepared.set_bach_size(3);

    let handler = pre_prepared.to_handler();

    let mut model: NewModel = NewModel::new();

    let number_of_epochs = 100000;

    for epoch in 0..=number_of_epochs{
        for i in 0..handler.len(){
            let output = model.forward(handler.input_bach(i));
            
            let grad_output = model.grad_output(&handler.output_bach(i));

            model.backward(grad_output);

            if epoch % 100 == 0{
                println!("Epoch {}, Cost: {}", epoch, cross_entropy_cost(&output, &handler.output_bach(i)).unwrap());
                println!("output:\n{}\n\ntarget:\n{}\n", output.matrix_to_string().unwrap(), handler.output_bach(i).matrix_to_string().unwrap());
            }
        }
    }
}
