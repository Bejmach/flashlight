# Flashlight

[![Rust](https://github.com/Bejmach/flashlight/actions/workflows/rust.yml/badge.svg?event=push)](https://github.com/Bejmach/flashlight/actions/workflows/rust.yml)
[![Crates.io](https://img.shields.io/crates/v/flashlight.svg)](https://crates.io/crates/flashlight)

> Package currently in development, use something else, like burn

> project not related to similarly named [flashlight](https://github.com/flashlight/flashlight). The name was coincidental and chosen independently.


## Already done
- Neural network structure
- forward propagation
- cost
- BACKPROP IS WORKING!!! (dont ask for math. I kinda did something using the explanation written in txt file in repo. Its not entirely correct, but a concept is there)

## To do
- easier usage
- f64 and f128 support(if I will try to do a more acurate calculator)
- saving and loading the model

## Additional info
- Don't use that for creating calculator. I tried, and you can see it in main.rs in repo, and a small difference in output genereta a big difference in real number, like expected = 0.7310586 = 100, output = 0.731054 = 43

## Dependencies
- [flashlight_tensor](https://crates.io/crates/flashlight_tensor)
- [rand](https://crates.io/crates/rand)


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

