# Flashlight

[![Rust](https://github.com/Bejmach/flashlight/actions/workflows/rust.yml/badge.svg?event=push)](https://github.com/Bejmach/flashlight/actions/workflows/rust.yml)
[![Crates.io](https://img.shields.io/crates/v/flashlight.svg)](https://crates.io/crates/flashlight)

> Package abandoned, use something else

> project not related to similarly named [flashlight](https://github.com/flashlight/flashlight). The name was coincidental and chosen independently.

## Current state of package
Package abandoned, because I lack knowledge and will for further development.  
"Flashlight" was supposed to be my small project that I will have fun developing, and with more knowledge I gained on the topic, the more I knew I need to rewrite whole project from base.  
This is my own decision, and no one can change it. I will still develop a github version, but mostly to learn about machine learning.  
Once I feel confident in my knowledge, I plan to create another neural network package, with a proper structure and propably a different name.  
The package will stay on crates but without any updates.

If someone wants to publish a crate with that name, please contact me, and I will transfer the ownershit to you

## Already done
- Neural network structure
- forward propagation
- cost
- backpropagation, but still in dev
- normalizing input when using input handler(create InputPrePrepared(name will change), append input and output data, set bach size, generate input handler, run backprop with baches from input handler)

## To do
- cleaner code
- relu instead of sigmoid on hidden
- easier usage
- f64 and f128 support
- saving and loading the model

> I wanted to be able to publish a pseudo finished version after version 0.0.10, but it will propably need to wait. Will try to publish V0.1.0 in like 3 months, because for the next month I wont have much time for this, and I want to also work on some games.

## Dependencies
- [flashlight_tensor](https://crates.io/crates/flashlight_tensor)
- [rand](https://crates.io/crates/rand)

### Patch notes
- 0.0.9
  - Input handler
  - Input normalization

