# CATCODER: Repository-Level Code Generation with Relevant Code and Type Context

## Structure

- The `catcoder` directory contains the source code of our approach CatCoder, benchmark datasets and evaluation scripts
- The `results` directory contains evaluation results used in the paper, including the metric values and detailed generated code

## Basic Requirements

- Linux
- Python 3.10+
- NVIDIA GPUs (with enough VRAM for LLMs)
- Install Defects4J and its dependencies (please refer to the instructions at https://defects4j.org/)
- Install Rust, Cargo and rust-analyzer (by running `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh` and then `rustup component add rust-analyzer`)

## Installation

- A clean `conda` environment is recommended, so that the following operations only affect a specific environment.
- cd into `catcoder/rust` and unzip `crates.zip` to the current directory.
    - In the unzipped folder, there are several repositories. Users are advised to run `cargo test` in each repo before evaluation. This would download dependencies and build reusable binary objects. Thus, this can speedup future testing procedures and prevent unexpected test failures or timeouts.
- To run evaluation code, install the dependencies by running `pip install -r requirements.txt` in `catcoder`.
- To use CatCoder's code for further research, it has to be configured in addition to the previous steps:
    - cd into `catcoder/tools/java`, and run `python setup.py install`.
    - cd into `catcoder/tools/intellirust`, and run `./configure && cargo cmd install`.

## Usage

- To run the experiments in the paper, and evaluate CatCoder (and other methods/LLMs) on the benchmarks:
    - cd into `catcoder/{java|rust}`, modify the `__main__` block (specify the method and the model for evaluation) in `evaluation.py`, and then run `python evaluation.py`. (The context data of all methods, including the baselines, has already been generated and stored in the benchmark datasets)
- For further research:
    - The benchmark datasets are located at `catcoder/{java|rust}/datasets`.
    - `catcoder/tools`, `catcoder/{java|rust}/retrieve_relevant_code.py` and `catcoder/{java|rust}/extract_type_context.py` contain the implementation of CatCoder.
