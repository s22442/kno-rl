# Reinforcement Learning examples

This repository illustrates the use of a few DRL algorithms on a simple [cart pole](https://coneural.org/florian/papers/05_cart_pole.pdf) environment:

- Deep Q-Network (DQN) - implemented in Python using [TensorFlow](https://github.com/tensorflow/tensorflow)
- Advantage Actor-Critic (A2C) - written both in Python with TensorFlow and in [PyTorch-powered](https://github.com/LaurentMazare/tch-rs) Rust
- Asynchronous Advantage Actor-Critic (A3C) - made in Rust and PyTorch
- Proximal Policy Optimization (PPO) - Rust and PyTorch as well

_This is a playground kind of project. Several learning optimizations could be added, tests don't exist, the code quality is rather average and bugs may appear._

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

- ### PPO (Rust & PyTorch)

  ```bash
  # dev build
  cargo run --bin ppo

  # release build (optimized)
  cargo run --bin ppo --release
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

  # PPO only
  cargo bench --bench ppo
  ```

## Benchmarks

Shared specification for all benchmarked programs:

- all models consist of 2 fully connected hidden layers with 100 neurons each
- model evaluation consists of 20 episodes
- only CPU is used

Training durations were measured on a PC with an AMD Ryzen 5 3600X 6-core processor.

<table>
  <tr>
    <th>Specification</th>
    <th>Technology</th>
    <th>Model evaluation frequency</th>
    <th>Samples</th>
    <th>Training median duration</th>
  </tr>
  <tr>
    <td>
      DQN
      <ul>
        <li>Learning rate: 1e-3</li>
        <li>Buffer size = 100 000 steps</li>
        <li>Initial steps in buffer performed by random policy = 200</li>
        <li>Steps collected per iteration = 10</li>
        <li>Training batch size = 200</li>
      </ul>
    </td>
    <td>Python & TensorFlow</td>
    <td>Every 500 steps</td>
    <td>20</td>
    <td>2142.15s</td>
  </tr>
  <tr>
    <td>
      A2C
      <ul>
        <li>Policy and value function are heads of the same model</li>
        <li>Learning rate: 1e-3</li>
        <li>Gamma = 0.99</li>
        <li>Model synchronization: every 100 steps or at episode end</li>
      </ul>
    </td>
    <td>Python & TensorFlow</td>
    <td>Every 500 steps*</td>
    <td>50</td>
    <td>379.38s</td>
  </tr>
  <tr>
    <td>
      A2C
      <ul>
        <li>Policy and value function are heads of the same model</li>
        <li>Learning rate: 1e-3</li>
        <li>Gamma = 0.99</li>
        <li>Model synchronization: every 100 steps or at episode end</li>
      </ul>
    </td>
    <td>Rust & PyTorch</td>
    <td>Every 500 steps*</td>
    <td>200</td>
    <td>19.54s</td>
  </tr>
  <tr>
    <td>
      A3C
      <ul>
        <li>Policy and value function are heads of the same model</li>
        <li>Learning rate: 1e-3</li>
        <li>Gamma = 0.99</li>
        <li>Model synchronization: every 100 steps or at episode end</li>
        <li>Threads = 8</li>
      </ul>
    </td>
    <td>Rust & PyTorch</td>
    <td>Every 500 steps*</td>
    <td>200</td>
    <td>5.71s</td>
  </tr>
  <tr>
    <td>
      PPO
     <ul>
        <li>Policy and value function are separate models</li>
        <li>Policy learning rate: 3e-4</li>
        <li>Value function learning rate: 1e-3</li>
        <li>Gamma = 0.99</li>
        <li>GAE lambda = 0.95</li>
        <li>Steps collected per epoch = 10 000</li>
        <li>Steps are collected using 8 parallel threads</li>
        <li>Policy training iterations = 50</li>
        <li>Value function training iterations = 50</li>
        <li>Target KL = 5e-3</li>
        <li>Clip value = 0.2</li>
      </ul>
    </td>
    <td>Rust & PyTorch</td>
    <td>Every 2 epochs</td>
    <td>200</td>
    <td>7.26s</td>
  </tr>
</table>

_\* - evaluation awaits model synchronization and is performed just after_

See the implementation for more details.

## License

[MIT License](https://opensource.org/licenses/MIT)

Copyright (c) 2024-PRESENT Kajetan Welc

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
