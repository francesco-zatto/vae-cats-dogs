# Variational AutoEncoder (VAE) on MNIST and CIFAR-10

This project implements a **Variational AutoEncoder (VAE)** using TensorFlow and Keras. It includes model definition, training, and visualization for both the **MNIST** handwritten digits and the **CIFAR-10** natural images datasets.

## Features

- Builds a VAE architecture with custom encoder, decoder, and sampling layers.
- Trains the model on both MNIST and CIFAR-10 datasets.
- Visualizes latent space embeddings.
- Generates new samples by decoding from the latent space.

## File

- `vae_mnist.ipynb`: Jupyter notebook containing the full implementation, training procedures, and visualizations.

## Model Overview

- **Encoder**: Maps input images to latent space using mean and log variance vectors.
- **Sampling**: Performs stochastic sampling from latent space using the reparameterization trick.
- **Decoder**: Reconstructs input images from sampled latent vectors.
- **Loss Function**: Combines reconstruction loss (e.g., binary cross-entropy or mean squared error) with KL divergence.

## License

This project is open-source and free to use for educational and research purposes.

