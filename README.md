# NeuPI: A Library for Neural Probabilistic Inference

NeuPI is a PyTorch-based library for solving inference tasks in Probabilistic Models (PMs) using neural network surrogates. It provides a modular framework for training neural models in a self-supervised fashion, where the Probabilistic Model itself provides the supervisory signal.

This approach eliminates the need for labeled training data, enabling neural networks to learn to solve tasks like Most Probable Explanation (MPE) by directly optimizing for the log-likelihood of their proposed solutions.

## Documentation

Documentation is available at [https://neupi.readthedocs.io/en/latest/](https://neupi.readthedocs.io/en/latest/).
GitHub repository: [https://github.com/Shivvrat/NeuPI](https://github.com/Shivvrat/NeuPI).

## Key Features

* **Self-Supervised Training**: Train neural surrogates using only the PGM definition—no labeled data required.
* **Advanced Inference**: Includes the **ITSELF** (`Inference Time Self-Supervised Training`) engine for test-time refinement, significantly improving inference accuracy.
* **Modular Architecture**: A clean separation of components:
    * **PGM Evaluators** (`MarkovNetwork`, `SumProductNetwork`)
    * **Neural Solvers** (`MLP`)
    * **Input Embedders** (`DiscreteEmbedder`)
    * **Trainers** (`SelfSupervisedTrainer`)
    * **Inference Engines** (`SinglePassInferenceEngine`, `ITSELF_Engine`)
* **Extensible**: Easily register your own custom components using the built-in factory system.
* **Efficient Backend**: Utilizes a high-performance Cython backend for parsing `.uai` files.

## Core Workflow

The core idea of NeuPI is to train a neural network to act as a fast approximator for a complex PM query. The workflow is as follows:

1.  **Load a PM**: This PM acts as the loss and evaluates the quality of solutions.
2.  **Define a Neural Surrogate**: This is the "solver," which learns to generate high-quality solutions.
3.  **Train**: Use the `SelfSupervisedTrainer` to train the solver. The loss is the negative log-likelihood of the solver's solutions, as computed by the PM.
4.  **Infer**: Use the trained model with an `InferenceEngine` to answer new queries.

## Installation

First, ensure you have PyTorch installed (with a GPU version if you want to use it). Then, clone the repository and install the library and its dependencies.

```bash
# Clone the repository
git clone https://github.com/Shivvrat/NeuPI
cd NeuPI

# Install the library in editable mode
pip install -e .
```

## Quick Start

Examples provided in the `examples` directory demonstrate:

    1.  Computing the negative log-likelihood (loss) of a solution to a MPE query on a Markov Network (Probabilistic Graphical odels) and Sum-Product Network (Probabilistic Circuits).
    2.  Training a neural solver to solve MPE queries on a Markov Network and Sum-Product Network.
    3.  Performing inference with a trained neural solver on a Markov Network and Sum-Product Network.

## Implemented Methods

The core components of NeuPI are directly linked to our published research. The following table maps the key classes in the library to the papers that introduce the primary concepts.

| Type          | Method / Component             | Primary Reference(s)                                                                                                                                                                                                                                                                                        | Description                                                                                                                                                  |
| :------------ | :----------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Trainer + Loss | `SelfSupervisedTrainer`        | [Arya et al., AAAI 2024](https://shivvrat.github.io/papers/AAAI_arya_2024_networkapproximatorsa.pdf), [Arya et al., NeurIPS 2024](https://shivvrat.github.io/papers/NeurIPS_12727_A_Neural_Network_Approac.pdf), [Arya et al., AISTATS 2025 (SINE)](https://shivvrat.github.io/papers/SINE%20Scalable%20MPE%20Inference.pdf)                                                                                                                                                                                                                                                                              | The core training loop for learning a neural solver by minimizing the negative log-likelihood provided by a PGM/PC evaluator.                              |
| Embedding | `DiscreteEmbedder`             | [Arya et al., AAAI 2024](https://shivvrat.github.io/papers/AAAI_arya_2024_networkapproximatorsa.pdf), [Arya et al., NeurIPS 2024](https://shivvrat.github.io/papers/NeurIPS_12727_A_Neural_Network_Approac.pdf)                                                                                                                                                                                                                                                                        | A feature engineering module that creates rich (discrete) input representations from variable assignments and bucket information (evidence, query, unobserved).       |
| Inference | `SinglePassInferenceEngine`    | [Arya et al., AAAI 2024](https://shivvrat.github.io/papers/AAAI_arya_2024_networkapproximatorsa.pdf), [Arya et al., AISTATS 2025 (SINE)](https://shivvrat.github.io/papers/SINE%20Scalable%20MPE%20Inference.pdf) | A standard inference pipeline involving a single forward pass of the neural network to produce an MPE/MMAP solution.                                        |
| Inference | `ITSELF_Engine`                | [Arya et al., NeurIPS 2024](https://shivvrat.github.io/papers/NeurIPS_12727_A_Neural_Network_Approac.pdf)                                                                                                                                                                                                                                                                              | Implements **I**nference **T**ime **S**elf-**S**upervised **L**earning **F**ine-tuning, an advanced engine that optimizes the model for each specific test instance to refine solution quality. |
| Discretizer | `KNearestDiscretizer`          | [Arya et al., AISTATS 2025 (SINE)](https://shivvrat.github.io/papers/SINE%20Scalable%20MPE%20Inference.pdf)                                                                                                                                                                                                                                                                       | A sophisticated discretization method that performs a beam search over the k-nearest binary vectors to find a high-quality discrete assignment.            |
| Discretizer | `OAUAI`   | [Arya et al., AISTATS 2025 (SINE)](https://shivvrat.github.io/papers/SINE%20Scalable%20MPE%20Inference.pdf)                                                                                                                                                                                                                                                                       | A heuristic discretizer that uses an oracle to answer the query over the variables with the highest uncertainty (probabilities closest to 0.5).                |
---

## Running Tests

To ensure all components are working correctly, run the test suite using `pytest`:

```bash
pytest
```

### 📖 Citation

If you use **NeuPI** in your research, please cite the following repository:

```bibtex
@article{arya2025neupi,
  title={NeuPI: A Library for Neural Probabilistic Inference},
  author={Arya, Shivvrat and Rahman, Tahrima and Gogate, Vibhav},
  link={https://github.com/Shivvrat/NeuPI},
  year={2025}
}
```

In addition, cite the relevant papers that introduce the core methods implemented in **NeuPI**:

* **Single-Pass Inference** and **Marginal MAP** inference in probabilistic circuits:

```bibtex
@article{aryaNeuralNetworkApproximators2024,
  title = {Neural {{Network Approximators}} for {{Marginal MAP}} in {{Probabilistic Circuits}}},
  author = {Arya, Shivvrat and Rahman, Tahrima and Gogate, Vibhav},
  year = {2024},
  month = mar,
  journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume = {38},
  number = {10},
  pages = {10918--10926},
  issn = {2374-3468},
  doi = {10.1609/aaai.v38i10.28966},
  urldate = {2024-03-27},
  copyright = {Copyright (c) 2024 Association for the Advancement of Artificial Intelligence},
  langid = {english},
}
```

* **Single-Pass Inference** and neural embedding techniques for Constrained MPE:

```bibtex
@inproceedings{aryaLearningSolveConstrained2024,
  title = {Learning to {{Solve}} the {{Constrained Most Probable Explanation Task}} in {{Probabilistic Graphical Models}}},
  booktitle = {Proceedings of {{The}} 27th {{International Conference}} on {{Artificial Intelligence}} and {{Statistics}}},
  author = {Arya, Shivvrat and Rahman, Tahrima and Gogate, Vibhav},
  year = {2024},
  month = apr,
  pages = {2791--2799},
  publisher = {PMLR},
  issn = {2640-3498},
  urldate = {2024-04-21},
  langid = {english},
}
```

* **ITSELF engine** and the general framework for MPE inference over Probabilistic Models:

```bibtex
@inproceedings{aryaNeuralNetworkApproach2024,
  title = {A Neural Network Approach for Efficiently Answering Most Probable Explanation Queries in Probabilistic Models},
  booktitle = {The {{Thirty-eighth Annual Conference}} on {{Neural Information Processing Systems}}},
  author = {Arya, Shivvrat and Rahman, Tahrima and Gogate, Vibhav Giridhar},
  year = {2024},
  month = nov,
  urldate = {2024-11-17},
  langid = {english},
}

```

* **Single-Pass Inference** and better embedding and discretization techniques in probabilistic graphical models:

```bibtex
@inproceedings{aryaSINEScalableMPE2025,
  title = {{{SINE}}: {{Scalable MPE}} Inference for Probabilistic Graphical Models Using Advanced Neural Embeddings},
  shorttitle = {{{SINE}}},
  booktitle = {The 28th {{International Conference}} on {{Artificial Intelligence}} and {{Statistics}}},
  author = {Arya, Shivvrat and Rahman, Tahrima and Gogate, Vibhav Giridhar},
  year = {2025},
  month = feb,
  urldate = {2025-06-22},
  langid = {english},
}
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.


## Disclaimer

This code was written for research purposes and therefore might not strictly adhere to established coding practices and guidelines. View and use at your own risk!

### Acknowledgments

This work was supported in part by the DARPA Perceptually-Enabled Task Guidance (PTG) Program under contract number HR00112220005, by the DARPA Assured Neuro Symbolic Learning and Reasoning (ANSR) Program under contract number HR001122S0039, by the National Science Foundation grant IIS-1652835 and by the AFOSR award FA9550-23-1-0239.
