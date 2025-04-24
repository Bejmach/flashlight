# Flashlight

[![Rust](https://github.com/Bejmach/flashlight/actions/workflows/rust.yml/badge.svg?event=push)](https://github.com/Bejmach/flashlight/actions/workflows/rust.yml)
[![Crates.io](https://img.shields.io/crates/v/flashlight.svg)](https://crates.io/crates/flashlight)

> Package currently in development, you can use it, but it's hard for now

> project not related to similarly named [flashlight](https://github.com/flashlight/flashlight). The name was coincidental and chosen independently.


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

