# NeuPI: A Library for Neural Probabilistic Inference

[![DOI](https://zenodo.org/badge/1012621004.svg)](https://doi.org/10.5281/zenodo.15873539)
[![Build Status](https://github.com/Shivvrat/NeuPI/actions/workflows/release.yml/badge.svg?branch=main)](https://github.com/Shivvrat/NeuPI/actions/workflows/release.yml)

NeuPI is a PyTorch-based library for solving inference tasks in Probabilistic Models (PMs) using neural network solvers. It provides a modular framework for training neural models in a self-supervised fashion, where the Probabilistic Model itself provides the supervisory signal.

This approach eliminates the need for labeled training data, enabling neural networks to learn to solve tasks like Most Probable Explanation (MPE), Constrained MPE, and Marginal MAP by directly optimizing for the log-likelihood of their proposed solutions.

## Documentation

Documentation is available at [https://neupi.readthedocs.io/en/latest/](https://neupi.readthedocs.io/en/latest/).

GitHub repository: [https://github.com/Shivvrat/NeuPI](https://github.com/Shivvrat/NeuPI).

## Installation

First, ensure you have PyTorch installed (with a GPU version if you want to use it). 

### Using PyPI

```bash
pip install neupi
```

### From Source

Then, clone the repository and install the library and its dependencies.

```bash
# Clone the repository
git clone https://github.com/Shivvrat/NeuPI
cd NeuPI

# Install the library in editable mode
pip install -e .
```
## Key Features

* **Self-Supervised Training**: Train neural solvers using only the Probabilistic Model—no labeled data required.
* **Advanced Inference**: Includes the **ITSELF** (`Inference Time Self-Supervised Training`) engine for test-time refinement, significantly improving inference accuracy.
* **Discretization**: Includes a simple `ThresholdDiscretizer` and more advanced methods like `KNearestDiscretizer` and `OAUAI` to find higher-scoring final assignments.
* **Extensible**: Easily register your own custom components using the built-in factory system.

## Quick Start
The core idea of NeuPI is to train a neural network to act as a fast approximator for a complex PM query. The workflow is as follows:

1. **Load a Probabilistic Model (PM)**: The first step is to load your probabilistic model (e.g., a `Markov Network` from a `.uai` file) using an evaluator class like `MarkovNetwork`. This object serves as the scoring function (log-likelihood); its primary role is to take a complete assignment of variables and compute its log-likelihood, which is the score we want to maximize. To train the neural network, we will use the negative log-likelihood of the PM as the loss function.

2. **Define a Neural Solver**: Next, define a neural network architecture, such as the provided `MLP`, to act as the solver. This network will be trained to take an inference query (evidence variables, query variables, and evidence values) as input and produce a high-quality solution. You will also define a preprocessor, like `DiscreteEmbedder`, which transforms the raw query into a feature-rich tensor suitable for the neural network.

3. **Train the Solver**: With the PM evaluator and neural solver defined, use the `SelfSupervisedTrainer`. The trainer orchestrates the learning process. In each step, it asks the neural network for a solution, passes that solution to the PM evaluator to get a score, and uses this score (specifically, the negative of the log-likelihood) as the loss signal to update the network's weights via backpropagation.

4. **Perform Inference**: Once the model is trained, use an inference engine to solve new queries. The `SinglePassInferenceEngine` provides a fast solution by performing one forward pass. For higher accuracy, the `ITSELF_Engine`  can be used to perform test-time fine-tuning on each new query, leveraging the PM evaluator to refine the solution.

5. **Discretize the Output**: The neural network outputs continuous probabilities. The final step is to convert these into a discrete binary (0/1) assignment using a `Discretizer`. You can use a simple `ThresholdDiscretizer` for speed or more advanced methods like `KNearestDiscretizer` and `OAUAI` to find higher-scoring final assignments.

## Examples

Examples provided in the `examples` directory demonstrate:

 **Example 1:** Computing the negative log-likelihood (loss) of a solution to a MPE query on a Markov Network (Probabilistic Graphical Models) and Sum-Product Network (Probabilistic Circuits).

 **Example 2:** Training a neural solver to solve MPE queries on a Markov Network and Sum-Product Network.

 **Example 3:** Performing inference with a trained neural solver on a Markov Network and Sum-Product Network.

 **Example 4:** Discretizing the probabilities to binary assignments. Advanced discretizers include `KNearestDiscretizer` and `OAUAI`.

## Implemented Methods

The core components of NeuPI are directly linked to our published research. The following table maps the key classes in the library to the papers that introduce the primary concepts.

| Type          | Method / Component             | Primary Reference(s)                                                                                                                                                                                                                                                                                        | Description                                                                                                                                                  |
| :------------ | :----------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Trainer | `SelfSupervisedTrainer`        | [Arya et al., AAAI 2024](https://shivvrat.github.io/papers/AAAI_arya_2024_networkapproximatorsa.pdf), [Arya et al., NeurIPS 2024](https://shivvrat.github.io/papers/NeurIPS_12727_A_Neural_Network_Approac.pdf), [Arya et al., AISTATS 2025 (SINE)](https://shivvrat.github.io/papers/SINE%20Scalable%20MPE%20Inference.pdf)                                                                                                                                                                                                                                                                              | The core training loop for learning a neural solver by minimizing the negative log-likelihood provided by a PGM/PC evaluator (e.g., `MarkovNetwork`, `SumProductNetwork`).                              |
| Loss | `MarkovNetwork` and `SumProductNetwork`        | [Arya et al., AAAI 2024](https://shivvrat.github.io/papers/AAAI_arya_2024_networkapproximatorsa.pdf), [Arya et al., NeurIPS 2024](https://shivvrat.github.io/papers/NeurIPS_12727_A_Neural_Network_Approac.pdf), [Arya et al., AISTATS 2025 (SINE)](https://shivvrat.github.io/papers/SINE%20Scalable%20MPE%20Inference.pdf)                                                                                                                                                                                                                                                                              | This compute the negative log likelihood scores for the given Probabilistic Model which is used as the loss function to train the neural network.                              |
| Embedding | `DiscreteEmbedder`             | [Arya et al., AAAI 2024](https://shivvrat.github.io/papers/AAAI_arya_2024_networkapproximatorsa.pdf), [Arya et al., NeurIPS 2024](https://shivvrat.github.io/papers/NeurIPS_12727_A_Neural_Network_Approac.pdf)                                                                                                                                                                                                                                                                        | A feature engineering module that creates (discrete) input representations from variable assignments and bucket information (evidence, query, unobserved).       |
| Inference | `SinglePassInferenceEngine`    | [Arya et al., AAAI 2024](https://shivvrat.github.io/papers/AAAI_arya_2024_networkapproximatorsa.pdf), [Arya et al., AISTATS 2025 (SINE)](https://shivvrat.github.io/papers/SINE%20Scalable%20MPE%20Inference.pdf) | A standard inference pipeline involving a single forward pass of the neural network to produce an MPE/MMAP solution.                                        |
| Inference | `ITSELF_Engine`                | [Arya et al., NeurIPS 2024](https://shivvrat.github.io/papers/NeurIPS_12727_A_Neural_Network_Approac.pdf)                                                                                                                                                                                                                                                                              | Implements **I**nference **T**ime **S**elf-**S**upervised **L**earning **F**ine-tuning, our advanced inference engine that optimizes the model for each specific test instance to refine solution quality. |
| Discretizer | `ThresholdDiscretizer`          | [Arya et al., AAAI 2024](https://shivvrat.github.io/papers/AAAI_arya_2024_networkapproximatorsa.pdf), [Arya et al., NeurIPS 2024](https://shivvrat.github.io/papers/NeurIPS_12727_A_Neural_Network_Approac.pdf)                                                                                                                                                                                                                                                                       | A simple discretization method that uses a threshold to discretize the probabilities.            |
| Discretizer | `KNearestDiscretizer`          | [Arya et al., AISTATS 2025 (SINE)](https://shivvrat.github.io/papers/SINE%20Scalable%20MPE%20Inference.pdf)                                                                                                                                                                                                                                                                       | A sophisticated discretization method that performs beam search over the k-nearest binary vectors to find a high-quality discrete assignment.            |
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

```bibtex

@misc{aryaNeuPILibraryNeural2025,
	title = {{NeuPI}: {A} library for neural probabilistic inference},
	copyright = {MIT License},
	shorttitle = {{NeuPI}},
	url = {https://zenodo.org/doi/10.5281/zenodo.15873631},
	abstract = {NeuPI is a PyTorch-based library for solving inference tasks in Probabilistic Models using neural network solvers. It provides a modular framework for training neural models in a self-supervised fashion, where the Probabilistic Model itself provides the supervisory signal.},
	publisher = {Zenodo},
	author = {Arya, Shivvrat and Rahman, Tahrima and Gogate, Vibhav},
	doi = {10.5281/ZENODO.15873631},
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

* **ITSELF engine**, **GUIDE engine** and the general framework for MPE inference over Probabilistic Models:

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

* **Single-Pass Inference** and better embedding and discretization techniques for inference over probabilistic graphical models:

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
