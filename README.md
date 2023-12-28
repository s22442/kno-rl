# Reinforcement Learning examples

This repository illustrates the use of a few DRL algorithms on a simple [cart pole](https://coneural.org/florian/papers/05_cart_pole.pdf) environment:

- Deep Q-Network (DQN) - implemented in Python using [TensorFlow](https://github.com/tensorflow/tensorflow)
- Advantage Actor-Critic (A2C) - written both in Python with TensorFlow and in [PyTorch-powered](https://github.com/LaurentMazare/tch-rs) Rust
- Asynchronous Advantage Actor-Critic (A3C) - made in Rust and PyTorch

## Prerequisites

- Python scripts:

  - [Python 3.10+](https://www.python.org/downloads/)

  - Packages:

  ```bash
  pip install -r python_requirements.txt
  ```

- Rust codebase:

  - [Rust 1.74+](https://www.rust-lang.org/tools/install)

  - Building Rust binaries may require additional OS packages:

    - Ubuntu:

      ```bash
      sudo apt install -y build-essential libssl-dev
      ```

  - Nightly toolchain:

  ```bash
  rustup toolchain install nightly
  ```

## Usage

- ### Simplistic DQN (Python & TensorFlow)

  ```bash
  python python/basic_dqn.py
  ```

- ### DQN (Python & TensorFlow)

  ```bash
  python python/dqn.py
  ```

- ### A2C (Python & TensorFlow)

  ```bash
  python python/a2c.py
  ```

- ### A2C (Rust & PyTorch)

  ```bash
  # dev build
  cargo run --bin a2c

  # release build (optimized)
  cargo run --bin a2c --release
  ```

- ### A3C (Rust & PyTorch)

  ```bash
  # dev build
  cargo run --bin a3c

  # release build (optimized)
  cargo run --bin a3c --release
  ```

- ### Python benchmarks

  ```bash
  # DQN
  python python/benchmarks/dqn.py

  # A2C
  python python/benchmarks/a2c.py
  ```

- ### Rust benchmarks

  ```bash
  # all benchmarks
  cargo bench

  # A2C only
  cargo bench --bench a2c

  # A3C only
  cargo bench --bench a3c
  ```
