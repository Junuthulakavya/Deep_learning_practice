# Deep_learning_practice
In this Iam going to explain you the different neural netorks in brief
<b>FNN:Feedforward Neural Network</b>
Feedforward Neural Network (FNN) is a type of artificial neural network in which information flows in a single direction—from the input layer through hidden layers to the output layer—without loops or feedback. It is mainly used for pattern recognition tasks like image and speech classification.
Input Layer: The input layer consists of neurons that receive the input data. Each neuron in the input layer represents a feature of the input data.
Hidden Layers: One or more hidden layers are placed between the input and output layers. These layers are responsible for learning the complex patterns in the data. Each neuron in a hidden layer applies a weighted sum of inputs followed by a non-linear activation function.
Output Layer: The output layer provides the final output of the network. The number of neurons in this layer corresponds to the number of classes in a classification problem or the number of outputs in a regression problem.<br><br>
<b>CNN:Convolutional Neural Network</b>
A Convolutional Neural Network is a type of neural network designed to work with image data. Instead of using Flatten like in FNNs (which ignores spatial structure), CNNs preserve the spatial hierarchy (pixels near each other are related) by using:<br>| Layer Type       | Purpose                                    |
| ---------------- | ------------------------------------------ |
| Conv2D      | Extracts features from local patches       |
| ReLU       | Applies non-linearity                      |
| MaxPooling2D | Reduces dimensions (downsampling)          |
| Flatten      | Converts 2D → 1D for classification        |
| Dense       | Fully connected layer for final prediction |<br><br>
<i>CNN Architecture</i>
Input (28x28 image)
 → Conv2D
 → ReLU
 → MaxPooling2D
 → Conv2D
 → ReLU
 → MaxPooling2D
 → Flatten
 → Dense
 → Output (10 classes for digits)
<br><br>
<b>RNN:Recurrent Neural Network<b>
An RNN is a type of neural network designed to handle sequential data by maintaining a memory of previous inputs.
Unlike FNN or CNN, RNNs process data step-by-step, remembering past information.
Great for tasks like:
Sentiment analysis
Text generation
Time-series forecasting
<i>basic architecture of RNN</i>
x₁ → [RNN Cell] → h₁ →
x₂ → [RNN Cell] → h₂ →
x₃ → [RNN Cell] → h₃ →
...
<b>layer by layer explanation</b>
 Layer-by-Layer Explanation:
1. Embedding Layer
🎯 Converts word indices into dense vectors of size 32
📐 Shape becomes (batch_size, 200, 32)
This helps the model understand the semantic meaning of words.

2. SimpleRNN Layer
🔁 Processes sequences one step at a time, maintaining a memory (hidden state)
📐 Outputs a single vector of size 64: (batch_size, 64)
Great for capturing the context of the entire sentence.

3. Dense Layer
🎯 A fully connected layer with sigmoid activation for binary classification
📐 Output: Probability of being Positive or Negative → (batch_size, 1)
<br>
<br>
<b>GAN:Generative Adversial Network</b>
A GAN is a type of deep learning model that consists of two neural networks — playing a game against each other:

Component	Role
🎨 Generator (G)	Tries to generate realistic fake data
🕵️ Discriminator (D)	Tries to detect if the data is real or fake

The generator learns to fool the discriminator.
The discriminator learns to catch the generator’s fakes.

Together, they get better — until the fake data looks realistic enough.
** GAN Workflow:**
Generator takes random noise and tries to generate fake images.

Discriminator looks at real vs. generated images and classifies them.

Both are updated based on how well they perform.

This process continues until the generated images become very realistic.
**AutoEncoders**
An Autoencoder is a neural network that learns to compress its input into a lower-dimensional form (encoding) and then reconstruct the original input from this encoding (decoding).

It has 3 main components:
Encoder – Compresses input data to a latent (lower-dimensional) representation

Latent Space – Bottleneck where compressed representation is held

Decoder – Reconstructs the input from the latent representation
**workflow**
Input (e.g. 28x28 image)
     ↓
[Encoder]
     ↓
Latent Representation (compressed)
     ↓
[Decoder]
     ↓
Reconstructed Output (same shape as input)

