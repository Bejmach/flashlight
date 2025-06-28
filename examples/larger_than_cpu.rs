use flashlight::{layers::LayerCpu, prelude::*};
#[allow(unused)]
use flashlight_tensor::prelude::*;

use rand::prelude::*;
use std::time::Instant;

pub struct NewModel{
    linear1: Linear<Cpu>,
    linear2: Linear<Cpu>,
    linear3: Linear<Cpu>,
    activation: Relu,
    output_activation: Sigmoid,
}



impl NewModel{
    fn new() -> Self{
        Self{
            linear1: Linear::new(2, 16, 0.01),
            linear2: Linear::new(16, 16, 0.01),
            linear3: Linear::new(16, 1, 0.01),
            activation: Relu::new(),
            output_activation: Sigmoid::new(),
        }
    }
    fn grad_output(&self, target: &Tensor<f32>) -> Tensor<f32>{
        self.output_activation.grad_output(target)
    }
}

impl ModelCpu for NewModel{
    fn forward(&mut self, input: Tensor<f32>) -> Tensor<f32> {
        let x = self.linear1.forward(&input);
        let x = self.activation.forward(&x);
        let x = self.linear2.forward(&x);
        let x = self.activation.forward(&x);
        let x = self.linear3.forward(&x);
        
        self.output_activation.forward(&x)
    }
    fn backward(&mut self, grad_output: Tensor<f32>) {
        let x = self.output_activation.backward(&grad_output);

        let x = self.linear3.backward(&x);
        let x = self.activation.backward(&x);
        let x = self.linear2.backward(&x);
        let x = self.activation.backward(&x);
        self.linear1.backward(&x);
    }
}

fn main() {
    let number_of_epochs = 100;
    let number_of_samples = 100;
    let bach_size = 100;

    let mut input_data: DataPreparaton = DataPreparaton::new();

    let mut rng = rand::rng();

    for _i in 0..number_of_samples{
        let num1 = rng.random_range(-100.0..100.0);
        let num2 = rng.random_range(-100.0..100.0);
        let input_vec = vec!{num1, num2};

        let output_vec;
        if num1>num2 {
            output_vec = vec!{1.0};
        }
        else{
            output_vec = vec!{0.0};
        }

        //println!("Input: {}, {}", input_vec[0], input_vec[1]);

        //print!("Output: ");
        for _i in 0..output_vec.len(){
            //print!("{}", output_vec[i]);
        }
        //println!("");

        input_data.append(&Tensor::from_data(&input_vec, &[1, input_vec.len() as u32]).unwrap(), &Tensor::from_data(&output_vec, &[1, output_vec.len() as u32]).unwrap());
    }

    input_data.set_bach_size(bach_size);

    let handler = input_data.to_handler();

    let mut model: NewModel = NewModel::new();

    let mut correct_counter: u32 = 0;

    for i in 0..input_data.data.len(){
        let output_data = model.forward(input_data.data[i].input_data.matrix_transpose().unwrap());

        //println!("Sample {}", i);
        //println!("Data: {}", input_data.input_data[i].matrix_to_string().unwrap());
        //println!("Expected: {}", input_data.output_data[i].matrix_to_string().unwrap());
        //println!("Output: {}", outpud_data[outpud_data.len()-1].matrix_to_string().unwrap());
        if input_data.data[i].output_data.get_data()[0] == 1.0 && output_data.get_data()[0] > 0.5 {
                //println!("correct");
                correct_counter += 1;
        }
        else if input_data.data[i].output_data.get_data()[0] == 0.0 && output_data.get_data()[0] < 0.5 {
                //println!("correct");
                correct_counter += 1;
        }
        else {
            //println!("WRONG");
        }
        //println!("\n");
    }
    println!("Ratio: {}/{}", correct_counter, number_of_samples);

    

    let start = Instant::now();
    for epoch in 0..=number_of_epochs{
    
        let mut cost_sum: f32 = 0.0;

        for i in 0..handler.len(){
            let output = model.forward(handler.input_bach(i));
            
            let grad_output = model.grad_output(&handler.output_bach(i));

            model.backward(grad_output);

            cost_sum += cross_entropy_cost(&output, &handler.output_bach(i)).unwrap();
        }
        if epoch % 10 == 0{
            println!("Epoch {}, Cost: {}", epoch, cost_sum/handler.len() as f32);
        }
    }
    let duration = start.elapsed();

    let mut correct_counter: u32 = 0;

    for _i in 0..number_of_samples{
        let num1 = rng.random_range(-100.0..100.0);
        let num2 = rng.random_range(-100.0..100.0);
        let input_vec = vec!{num1, num2};

        let output_vec;
        if num1>num2 {
            output_vec = vec!{1.0};
        }
        else{
            output_vec = vec!{0.0};
        }
        let input_data = Tensor::from_data(&input_vec, &[2, 1]).unwrap();
        let output_data = model.forward(input_data.clone());

        println!("Sample {}", _i);
        println!("Data: {}", input_data.matrix_transpose().unwrap().matrix_to_string().unwrap());
        println!("Expected: {}", output_vec[0]);
        println!("Output: {}", output_data.matrix_to_string().unwrap());
        if output_vec[0] == 1.0 && output_data.get_data()[0] >= 0.5 {
                println!("correct");
                correct_counter += 1;
        }
        else if output_vec[0] == 0.0 && output_data.get_data()[0] < 0.5 {
                println!("correct");
                correct_counter += 1;
        }
        else {
            println!("WRONG");
        }
        println!("\n");
    }
    println!("Ratio: {}/{}", correct_counter, number_of_samples);
    println!("Learning time: {:?}", duration);
}
