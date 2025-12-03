use flashlight::{layers::*, prelude::*};
#[allow(unused)]
use flashlight_tensor::prelude::*;
use async_trait::async_trait;

use rand::prelude::*;
use std::time::Instant;

pub struct NewModel{
    linear1: Linear<Gpu>,
    linear2: Linear<Gpu>,
    linear3: Linear<Gpu>,
    activation: Relu<Cpu>,
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

#[async_trait]
impl ModelGpu for NewModel{
    async fn forward(&mut self, input: Tensor<f32>) -> Tensor<f32> {
        let x = self.linear1.forward(&input).await;
        let x = self.activation.forward(&x);
        let x = self.linear2.forward(&x).await;
        let x = self.activation.forward(&x);
        let x = self.linear3.forward(&x).await;
        
        self.output_activation.forward(&x)
    }
    async fn backward(&mut self, grad_output: Tensor<f32>) {
        let x = self.output_activation.backward(&grad_output);

        let x = self.linear3.backward(&x).await;
        let x = self.activation.backward(&x);
        let x = self.linear2.backward(&x).await;
        let x = self.activation.backward(&x);
        self.linear1.backward(&x).await;
    }
    fn clear_buffers(&mut self){
        self.linear1.clear();
        self.linear2.clear();
        self.linear3.clear();
    }
}

#[tokio::main]
async fn main() {
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
        let output_data = model.forward(input_data.data[i].input_data.matrix_transpose().unwrap()).await;

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

    model.clear_buffers();

    //learning time
    let start = Instant::now();

    let mut forward_duration: std::time::Duration = std::time::Duration::new(0, 0);
    let mut backward_duration: std::time::Duration = std::time::Duration::new(0, 0);
    let mut clear_duration: std::time::Duration = std::time::Duration::new(0, 0);
    let mut cpu_duration: std::time::Duration = std::time::Duration::new(0, 0);

    for epoch in 0..=number_of_epochs{
    
        let mut cost_sum: f32 = 0.0;

        for i in 0..handler.len(){
            let start = Instant::now();
            let input_bach = handler.input_bach(i);
            let output_bach = &handler.output_bach(i);
            cpu_duration += start.elapsed();

            let start = Instant::now();
            let output = model.forward(input_bach).await;
            forward_duration += start.elapsed();

            let start = Instant::now();
            let grad_output = model.grad_output(output_bach);
            cpu_duration += start.elapsed();

            let start = Instant::now();
            model.backward(grad_output).await;
            backward_duration += start.elapsed();
            
            let start = Instant::now();
            //println!("{:?}, {:?}", output.get_data(), handler.output_bach(i).get_data());
            cost_sum += cross_entropy_cost(&output, &handler.output_bach(i)).unwrap();
            //println!("{}", cross_entropy_cost(&output, &handler.output_bach(i)).unwrap());
            cpu_duration += start.elapsed();

            let start = Instant::now();
            model.clear_buffers();
            clear_duration += start.elapsed();
        }
        if epoch % 10 == 0{
            println!("Epoch {}, Cost: {}", epoch, cost_sum/handler.len() as f32);
        }
    }
    let duration: std::time::Duration = start.elapsed();

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
        let output_data = model.forward(input_data.clone()).await;

        //println!("Sample {}", _i);
        //println!("Data: {}", input_data.matrix_transpose().unwrap().matrix_to_string().unwrap());
        //println!("Expected: {}", output_vec[0]);
        //println!("Output: {}", output_data.matrix_to_string().unwrap());
        if output_vec[0] == 1.0 && output_data.get_data()[0] >= 0.5 {
                //println!("correct");
                correct_counter += 1;
        }
        else if output_vec[0] == 0.0 && output_data.get_data()[0] < 0.5 {
                //println!("correct");
                correct_counter += 1;
        }
        else {
            //println!("WRONG");
        }
        //println!("\n");
    }
    println!("Ratio: {}/{}", correct_counter, number_of_samples);
    println!("Learning time: {:?}", duration);
    println!("Forward time: {:?}\nBackward time: {:?}\nClear time: {:?}\nCpu time: {:?}", forward_duration, backward_duration, clear_duration, cpu_duration);
}
