use std::time::Instant;

use flashlight::{layers::{*, linear::Linear}, prelude::*};
#[allow(unused)]
use flashlight_tensor::prelude::*;

#[tokio::main]
async fn main() {
    let weights = Tensor::rand(1.0, &[16, 2]);
    let biases = Tensor::rand(1.0, &[16, 1]);

    let start = Instant::now();
    let mut linear_gpu: Linear<Gpu> = Linear::with_weights_and_bias(weights.clone(), biases.clone(), 0.01);
    let gpu_create_duration = start.elapsed();

    let start = Instant::now();
    let mut linear_cpu: Linear<Cpu> = Linear::with_weights_and_bias(weights.clone(), biases.clone(), 0.01);
    let cpu_create_duration = start.elapsed();

    let input_tensor = Tensor::rand(1.0, &[2, 100]);

    let start = Instant::now();
    let _cpu_output = linear_cpu.forward(&input_tensor.clone());
    let cpu_forward_duration = start.elapsed();

    let start = Instant::now();
    let _gpu_output = linear_gpu.forward(&input_tensor.clone()).await;
    let gpu_first_forward_duration = start.elapsed();

    let start = Instant::now();
    linear_gpu.clear();
    let gpu_clear_time = start.elapsed();

    let start = Instant::now();
    linear_gpu.forward(&input_tensor.clone()).await;
    let gpu_second_forward_duration = start.elapsed();

    //println!("Forward:\nCpu: {:?}\nGpu: {:?}", cpu_output.get_data(), gpu_output.get_data());

    let grad_output = Tensor::rand(1.0, &[16, 100]);

    let start = Instant::now();
    let _cpu_output = linear_cpu.backward(&grad_output);
    let cpu_backward_duration = start.elapsed();

    let start = Instant::now();
    let _gpu_output = linear_gpu.backward(&grad_output).await;
    let gpu_first_backward_duration = start.elapsed();
    
    let start = Instant::now();
    linear_gpu.backward(&grad_output).await;
    let gpu_second_backward_duration = start.elapsed();

    //println!("Backward:\nCpu: {:?}\nGpu: {:?}", cpu_output.get_data(), gpu_output.get_data());

    println!("Creation: \n\t-cpu: {:?}\n\t-gpu: {:?} \nForward: \n\t-cpu: {:?}\n\t-gpu1: {:?}\n\t-gpu2: {:?}\nBackward: \n\t-cpu: {:?}\n\t-gpu1: {:?}\n\t-gpu2: {:?}\nClear: \n\t-gpu: {:?}", cpu_create_duration, gpu_create_duration, cpu_forward_duration, gpu_first_forward_duration, gpu_second_forward_duration, cpu_backward_duration, gpu_first_backward_duration, gpu_second_backward_duration, gpu_clear_time);
}

