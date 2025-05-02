# Flashlight

[![Rust](https://github.com/Bejmach/flashlight/actions/workflows/rust.yml/badge.svg?event=push)](https://github.com/Bejmach/flashlight/actions/workflows/rust.yml)
[![Crates.io](https://img.shields.io/crates/v/flashlight.svg)](https://crates.io/crates/flashlight)
[![Docs.rs](https://docs.rs/flashlight_tensor/badge.svg)](https://docs.rs/flashlight_tensor)

> Package currently in development, use something else, like burn

> project not related to similarly named [flashlight](https://github.com/flashlight/flashlight). The name was coincidental and chosen independently.


## Already done
- Neural network structure
- forward propagation
- cost
- backpropagation
- batch handler kinda works, still need to upgrade it

## To do
- easier usage
- f64 and f128 support(if I will try to do a more acurate calculator)
- saving and loading the model

## Dependencies
- [flashlight_tensor](https://crates.io/crates/flashlight_tensor)
- [rand](https://crates.io/crates/rand)

## Instalation
```toml
[depencencies]
flashlight = "0.0.11"
flashlight_tensor = "0.2.6"
```

## Documentation

[Docs](https://docs.rs/flashlight_tensor/latest/flashlight_tensor/)

## Quick Start
```rust
use flashlight::prelude::*;
use flashlight_tensor::prelude::*;

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
```

## Why the name "Flashlight"

Before I decided on the name, I wanted to use some  mythology reference, because I assumed that the "Torch" library based its name on Prometheus myth, but it didn't click.

Some time after that, I thought to myself "Why base the name on some mythical meaning of stuff, when I can refer to them literally.", and that's when I came up with three name ideas for project: "**Lamp**", "**Bulb**" and "**Flashlight**", and well... I decided on "Flashlight" because It sounded the **goofiest** of them all.

Sure, I could've tried to justify it with something like, "*Flashlight is written in Rust and Torch is written In C, so Flashlight is safer, like how flashlights are safer than torches...*" but I would be lying if I did. I decided on that name **because it was funny**, and I didn’t think it would actually fit the project — but here we are.

I hope you had a good time reading this short personal story. I just rly wanted to include that in the readme, so deal with it.

P.S.
I literally have no idea what I'm doing here. I just thought one day, "Creating neural network from scratch while learning new language is a great idea". I dont even know how all math is working. I understand the concept, and try to deal with it.


### Patch notes
- 0.0.9
  - Input handler
  - Input normalization
- 0.0.10
  - tbh, nothing much
- 0.0.11
  - modular model

