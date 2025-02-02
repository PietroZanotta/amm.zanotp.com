---
title: "Post-Variational Quantum Neural Networks"
datePublished: Sun Feb 02 2025 23:54:19 GMT+0000 (Coordinated Universal Time)
cuid: cm6oa3nm9001b08lae5v04f0v
slug: pvqnn
cover: https://cdn.hashnode.com/res/hashnode/image/stock/unsplash/Gow8svoRZBA/upload/131f6fb153edfbe6d52e7787693477bb.jpeg
tags: quantum-computing, quantum-neural-networks

---

Variational quantum circuit are among the most promising methods for dealing with optimization problems, combinatorial optimization and quantum machine learning. However, despite their popularity, many of the ansatze upon which such circuit relies suffer of well documented barren plateau problem \[3\] as the quantum hardware noise or circuit depth increases. Moreover the training landscape doesn’t in generally correspond to any well-characterized optimization program, therefore making the investigation difficult. Because of that problem, multiple alternatives to variational quantum circuits have been studied. \[7\] for example, proposed to use classical combinations of quantum states to solve linear systems with near-term quantum computers and the idea of using combinations of quantum states and systematically generate ansatze has proven a viable alternative to variational solutions that can circumvent the barren plateau problem, and has found application in quantum eigen solvers \[4\], semidefinite programming \[6\] and simulations \[5\].

This blog post explores an alternative to variational quantum models, called post-variational quantum models and in particular post-variational quantum neural networks, a quantum machine learning model based on ensemble strategies which doesn’t rely on a single trainable circuit but on a classical combination of fixed circuits.

## Variational Circuits

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1738512468827/067bdc3e-c1b1-4f19-bffd-872477aa2889.png align="center")

Variational quantum circuits (see the picture above, coming from \[1\]) are hybrid methods referring to a large class of circuits operating on pure states and are also known as quantum neural networks when applied to machine learning tasks.

Such circuit operates in the following manner:

* encoding data \\(x\\) into a n-qubit quantum state \\(\rho(x) \in \mathcal M_{2^n\times 2^n}\\)
    
* a parametrized circuit \\(U(\theta) \\) (also called ansatz) is the applied on the encoded state, with parameters \\(\theta \in R^{d}\\), resulting in:
    

$$\rho(x, \theta) = U(\theta)\rho(x)U(\theta)^\dagger$$

* an estimation of the results of such circuits is then constructed with an observable \\(O\\):
    

$$E_\theta(x)= tr(O\rho(x, \theta))$$

* the parameters \\(\theta\\) are optimized using a gradient based optimization relying on the gradient of the variational quantum circuit (typically computed with [parameter shift rule](https://amm.zanotp.com/computing-gradients-of-quantum-circuits-using-parameter-shift-rule)).
    

The main challenges of variational quantum circuits are the following:

* despite some problem inspired ansatze exist, defining problem agnostic ansatze that are expressive enough to represent a useful function but don’t suffer of the barren plateau problem is a challenge and an open research direction
    
* implementing continuous parameterized rotations on real hardware is limited by the precision of control electronics.
    

## Post-Variational Circuits

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1738513556949/eeff5ecc-d681-440d-8bb5-35f1c65bc055.png align="center")

Because of the general difficulty and lack of training guarantees provided by variational algorithms, \[1\] has proposed a new approach, named “post-variational“. This approach (see the picture above, coming from \[1\]) basically replaces the parametrized quantum circuit of the variational approach with fixed circuits (parametrized only by the input data) and find an optimal combination of such the result of such circuits.

The idea is therefore to combine observable and ansatz in a single parametrized observable:

$$\mathcal D(\theta) = U(\theta)^\dagger OU (\theta)$$

Since any observable can be expressed as a linear combination of Hermitian matrices, one can express the observable \\(\mathcal D(\theta)\\) as linear combinations:

$$\mathcal D(\theta) = \sum_{i=1}^q a_i(\theta)\mathcal D_i$$

This comes from the fact that \\(U(\theta)\\) can be written as the product of unitary matrices:

\\[U(\theta) = \prod_{i=1}^s U_i(\theta_i)\\]

and because of Stone’s theorem, \\(U(\theta)\\) can be written as:

\\[U(\theta) = \prod_{i=1}^s W_ie^{j\theta_iH_i}V_i\\]

where \\(j\\) is the imaginary unit, \\(W_i\\) and \\(V_i\\) are fixed matrices and \\(H_i\\) are Hermitian matrices.

Thanks to the Baker–Campbell–Hausdorff identity one can represent \\(U(\theta)\\) as:

\\[\prod_{i=1}^sV_i^\dagger\left(\sum_{k=0}^\infty \frac{[(j\theta_iH_i)^k, W_i^\dagger O W_i]} {k!} \right)V_i=\prod_{i=1}^s\sum_{k=0}^\infty\frac{\theta_i^k}{k!}V_i^\dagger[(jH_i)^k, W_i^\dagger O W_i]V_i\\]

where \\([(X)^n, Y) = [X, \dots, [X,[X, Y]]]\\). Since \\(jH_i\\) is anti-Hermitian, \\([(j\theta_iH_i)^k, W_i^\dagger O W_i]\\) is Hermitian for all \\(i\\), which allows to rewrite \\(U(\theta)\\) as a weighted polynomial sum of Hermitian matrices against \\(\theta\\) which allows the following:

\\[\mathcal D(\theta) = U(\theta)^\dagger OU(\theta)\sum_{i=1}^q a_i(\theta)\mathcal D_i\\]

Moreover, since any Hermitian operator can be expressed in a basis of Pauli matrices:

\\[H \in M_{2\times2}(C)^{\otimes n} \in \text{span}({X; Y; Z; I}^{\otimes n})\\]

then \\(\mathcal D(\theta)\\) pertains to the same space and therefore at most \\(4^n\\) terms are necessary to express the optimal answer. Therefore, considering that a variational quantum circuit takes \\(\mathcal O(poly(s))\\)parameters to express the optimal solution, while the post-variational approach takes \\(\mathcal O(4^n)\\), the variational approach has an advantage, coming from the fact that it is to generate different observables on higher orders of \\(\theta\\), something that classical computers cannot achieve.

However, in order to get an approximate solution, one can restrict the number of Hermitian terms used in the post-variational approach to \\(\mathcal O(poly(s))\\), renouncing to some expressibility.

### Estimation of the parameters in the post-variational setting

The estimation in the post variational setting is:

\\[E_\theta = tr\left(\mathcal D(\theta) \rho(x)\right)= \sum_{i=1}^qa_i(\theta)tr\left(D_i\rho(x)\right)\\]

One can consider \\(\sum_{i=1}^qa_i(\theta)tr\left(D_i\rho(x)\right)\\) as a function \\(\mathcal H_\theta:\\) \\(R^q \rightarrow R\\) s.t.:

\\[E_\theta = \mathcal H_\theta\left(\left\{tr\left(D_i\rho(x)\right)\right\}_{i=1}^q\right)\\]

and exploiting the universal approximation theorem, the function \\(\mathcal H_\theta\\) can be approximated by a neural network.

### Design principles of post-variational quantum circuits

So far the only challenge pertaining to the post-variational design mentioned is the exponential amount of possible circuits. However the post-variational setting has another major challenge: the heuristic choice of fixed circuits and observables. the authors of \[1\] describe multiple strategies to decide the observables \\(\mathcal D_i\\).

#### Ansatz expansion

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1738521125836/24503657-d53a-422b-b87a-66cd552e9333.png align="center")

The first strategy outlined is to replace a problem agnostic parametrized ansatz \\(U(\theta)\\)coming from a variational quantum circuit with an ensemble of fixed ansatze \\(\{U_\alpha\}_{\alpha=1}^p\\). The authors use truncated Taylor polynomial expansions of the variational parameters to generate fixed ansatze for the model and use [parameter-shift rules](https://amm.zanotp.com/computing-gradients-of-quantum-circuits-using-parameter-shift-rule) to find derivatives of the trace-induced measurements of parameterized quantum circuits.

Therefore the full Taylor expansion of \\(U^\dagger(\theta)OU(\theta)\\) can be expressed a linear combination of \\(U^\dagger(\theta')OU(\theta')\\) where \\(\theta \in \{0,\pm\frac \pi 2\}^k\\). For a truncation of order \\(R\\), the number of circuit required is:

\\[\sum_{j=0}^R{k\choose j}2^j \in \mathcal O(2^Rk^R)\\]

which scales fast if a deep ansatz is chosen. To reduce the number of circuits required, one can adopt pruning techniques.

#### Observable construction

The observable construction strategy decomposes the parametrized observable \\(\mathcal D(\theta)\\) agains the basis of quantum observables s.t.:

\\[\mathcal D(\theta^*)\rightarrow \mathcal D(\alpha)= \sum_{P\in\{X; Y; Z; I\}^{\otimes n}}\alpha_P P\\]

The real problem of this strategy is that it scales exponentially with the number of qubits used in the system, therefore an heuristic selection is necessary. Considering all Pauli observables withing a locality \\(L\\) is considered a good heuristic, being most physical Hamiltonian local as well. If the target observable is \\(L\\)\-local, one can exploit the classical shadows method \[2\] to reduce the number of measurement required while obtaining the same additive error term. In the case that the observables are the complete set of L-local Paulis, the number of observables required is:

\\[\sum_{j=0}^L{n\choose j}3^j \in \mathcal O(3^ln^L)\\]

while if the classical shadow method us used, the number of random measurement of the circuit is:

\\[\mathcal O(3^LL\log n)\\]

#### Hybrid approach

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1738522315813/350cc2d4-3d2a-4e76-8a4e-1ae7ffc96f51.png align="center")

One might want to use ansatz circuits during the construction of observables, in order to increase the expressivity of the model. A strategy might therefore combine both the ansatz expansion strategy and the observable construction strategy.

The idea is that, instead of directly expanding \\(U(\theta)\\) in \\(\mathcal D(\theta) = U^\dagger (\theta)OU(\theta)\\), the ansatz is split into two unitaries:

\\[U(\theta)=U_B(\theta)U_A(\theta)\\]

and therefore:

\\[\mathcal D(\theta) =U_A^\dagger(\theta)U_B^\dagger(\theta)OU_B(\theta)U_A(\theta)\\]

Let \\(\mathcal D'(\theta) = U^\dagger_B(\theta) OU_B(\theta)\\), it can be decomposed into a linear combination of Paulis using the observable construction strategy. On the other hand the remaining Ansatz \\(U_A(\theta)\\) can be expanded using the ansatz expansion method. Last pruning techniques can be used to reduce the number of circuits.

#### Numerically comparing Post-Variational and Variational Quantum Neural Network

The following example demonstrates how to employ the post-variational quantum neural network on the classical machine learning task of image classification. The example comes from the [Pennylane documentation](https://pennylane.ai/qml/demos/tutorial_post-variational_quantum_neural_networks).

```python
import pennylane as qml
from pennylane import numpy as np
import jax
from jax import numpy as jnp
import optax
from itertools import combinations
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import matplotlib.colors
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)

# Load the digits dataset with features (X_digits) and labels (y_digits)
X_digits, y_digits = load_digits(return_X_y=True)

# Create a boolean mask to filter out only the samples where the label is 2 or 6
filter_mask = np.isin(y_digits, [2, 6])

# Apply the filter mask to the features and labels to keep only the selected digits
X_digits = X_digits[filter_mask]
y_digits = y_digits[filter_mask]

# Split the filtered dataset into training and testing sets with 10% of data reserved for testing
X_train, X_test, y_train, y_test = train_test_split(
    X_digits, y_digits, test_size=0.1, random_state=42
)

# Normalize the pixel values in the training and testing data
# Convert each image from a 1D array to an 8x8 2D array, normalize pixel values, and scale them
X_train = np.array([thing.reshape([8, 8]) / 16 * 2 * np.pi for thing in X_train])
X_test = np.array([thing.reshape([8, 8]) / 16 * 2 * np.pi for thing in X_test])

# Adjust the labels to be centered around 0 and scaled to be in the range -1 to 1
# The original labels (2 and 6) are mapped to -1 and 1 respectively
y_train = (y_train - 4) / 2
y_test = (y_test - 4) / 2
```

To visualize some of the digits:

```python
fig, axes = plt.subplots(nrows=2, ncols=3, layout="constrained")
for i in range(2):
    for j in range(3):
      axes[i][j].matshow(X_train[2*(2*j+i)])
      axes[i][j].axis('off')
fig.subplots_adjust(hspace=0.0)
fig.tight_layout()
plt.show()
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1738523659827/a6f57ed5-c67a-4f42-a5a0-543f5ab1e951.png align="center")

Now it’s time to train the QML models:

* first will embed our data through a series of rotation gates
    
* will then have an ansatz of rotation gates with parameters’ weights
    

```python
def feature_map(features):
    # Apply Hadamard gates to all qubits to create an equal superposition state
    for i in range(len(features[0])):
        qml.Hadamard(i)

    # Apply angle embeddings based on the feature values
    for i in range(len(features)):
        # For odd-indexed features, use Z-rotation in the angle embedding
        if i % 2:
            qml.AngleEmbedding(features=features[i], wires=range(8), rotation="Z")
        # For even-indexed features, use X-rotation in the angle embedding
        else:
            qml.AngleEmbedding(features=features[i], wires=range(8), rotation="X")

# Define the ansatz (quantum circuit ansatz) for parameterized quantum operations
def ansatz(params):
    # Apply RY rotations with the first set of parameters
    for i in range(8):
        qml.RY(params[i], wires=i)

    # Apply CNOT gates with adjacent qubits (cyclically connected) to create entanglement
    for i in range(8):
        qml.CNOT(wires=[(i - 1) % 8, (i) % 8])

    # Apply RY rotations with the second set of parameters
    for i in range(8):
        qml.RY(params[i + 8], wires=i)

    # Apply CNOT gates with qubits in reverse order (cyclically connected)
    # to create additional entanglement
    for i in range(8):
        qml.CNOT(wires=[(8 - 2 - i) % 8, (8 - i - 1) % 8])
```

We first test the performance of a shallow variational algorithm on the digits dataset:

```python
dev = qml.device("default.qubit", wires=8)


@qml.qnode(dev)
def circuit(params, features):
    feature_map(features)
    ansatz(params)
    return qml.expval(qml.PauliZ(0))


def variational_classifier(weights, bias, x):
    return circuit(weights, x) + bias


def square_loss(labels, predictions):
    return np.mean((labels - qml.math.stack(predictions)) ** 2)


def accuracy(labels, predictions):
    acc = sum([np.sign(l) == np.sign(p) for l, p in zip(labels, predictions)])
    acc = acc / len(labels)
    return acc


def cost(params, X, Y):
    predictions = [variational_classifier(params["weights"], params["bias"], x) for x in X]
    return square_loss(Y, predictions)


def acc(params, X, Y):
    predictions = [variational_classifier(params["weights"], params["bias"], x) for x in X]
    return accuracy(Y, predictions)


np.random.seed(0)
weights = 0.01 * np.random.randn(16)
bias = jnp.array(0.0)
params = {"weights": weights, "bias": bias}
opt = optax.adam(0.05)
batch_size = 7
num_batch = X_train.shape[0] // batch_size
opt_state = opt.init(params)
X_batched = X_train.reshape([-1, batch_size, 8, 8])
y_batched = y_train.reshape([-1, batch_size])


@jax.jit
def update_step_jit(i, args):
    params, opt_state, data, targets, batch_no = args
    _data = data[batch_no % num_batch]
    _targets = targets[batch_no % num_batch]
    _, grads = jax.value_and_grad(cost)(params, _data, _targets)
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return (params, opt_state, data, targets, batch_no + 1)


@jax.jit
def optimization_jit(params, data, targets):
    opt_state = opt.init(params)
    args = (params, opt_state, data, targets, 0)
    (params, opt_state, _, _, _) = jax.lax.fori_loop(0, 200, update_step_jit, args)
    return params


params = optimization_jit(params, X_batched, y_batched)
var_train_acc = acc(params, X_train, y_train)
var_test_acc = acc(params, X_test, y_test)

print("Training accuracy: ", var_train_acc)
print("Testing accuracy: ", var_test_acc)

# Training accuracy:  0.7484472049689441
# Testing accuracy:  0.6944444444444444
```

The observable construction heuristic:

```python
def local_pauli_group(qubits: int, locality: int):
    assert locality <= qubits, f"Locality must not exceed the number of qubits."
    return list(generate_paulis(0, 0, "", qubits, locality))

# This is a recursive generator function that constructs Pauli strings.
def generate_paulis(identities: int, paulis: int, output: str, qubits: int, locality: int):
    # Base case: if the output string's length matches the number of qubits, yield it.
    if len(output) == qubits:
        yield output
    else:
        # Recursive case: add an "I" (identity) to the output string.
        yield from generate_paulis(identities + 1, paulis, output + "I", qubits, locality)

        # If the number of Pauli operators used is less than the locality, add "X", "Y", or "Z"
        # systematically builds all possible Pauli strings that conform to the specified locality.
        if paulis < locality:
            yield from generate_paulis(identities, paulis + 1, output + "X", qubits, locality)
            yield from generate_paulis(identities, paulis + 1, output + "Y", qubits, locality)
            yield from generate_paulis(identities, paulis + 1, output + "Z", qubits, locality)
```

For each image sample, we measure the output of the quantum circuit using the \\(k\\)\-local observables sequence, and perform logistic regression on these outputs:

```python
# Initialize lists to store training and testing accuracies for different localities.
train_accuracies_O = []
test_accuracies_O = []

for locality in range(1, 4):
    print(str(locality) + "-local: ")

    # Define a quantum device with 8 qubits using the default simulator.
    dev = qml.device("default.qubit", wires=8)

    # Define a quantum node (qnode) with the quantum circuit that will be executed on the device.
    @qml.qnode(dev)
    def circuit(features):
        # Generate all possible Pauli strings for the given locality.
        measurements = local_pauli_group(8, locality)

        # Apply the feature map to encode classical data into quantum states.
        feature_map(features)

        # Measure the expectation values of the generated Pauli operators.
        return [qml.expval(qml.pauli.string_to_pauli_word(measurement)) for measurement in measurements]

    # Vectorize the quantum circuit function to apply it to multiple data points in parallel.
    vcircuit = jax.vmap(circuit)

    # Transform the training and testing datasets by applying the quantum circuit.
    new_X_train = np.asarray(vcircuit(jnp.array(X_train))).T
    new_X_test = np.asarray(vcircuit(jnp.array(X_test))).T

    # Train a Multilayer Perceptron (MLP) classifier on the transformed training data.
    clf = MLPClassifier(early_stopping=True).fit(new_X_train, y_train)

    # Print the log loss for the training data.
    print("Training loss: ", log_loss(y_train, clf.predict_proba(new_X_train)))

    # Print the log loss for the testing data.
    print("Testing loss: ", log_loss(y_test, clf.predict_proba(new_X_test)))

    # Calculate and store the training accuracy.
    acc = clf.score(new_X_train, y_train)
    train_accuracies_O.append(acc)
    print("Training accuracy: ", acc)

    # Calculate and store the testing accuracy.
    acc = clf.score(new_X_test, y_test)
    test_accuracies_O.append(acc)
    print("Testing accuracy: ", acc)
    print()

locality = ("1-local", "2-local", "3-local")
train_accuracies_O = [round(value, 2) for value in train_accuracies_O]
test_accuracies_O = [round(value, 2) for value in test_accuracies_O]
x = np.arange(3)
width = 0.25

# Create a bar plot to visualize the training and testing accuracies.
fig, ax = plt.subplots(layout="constrained")
# Training accuracy bars:
rects = ax.bar(x, train_accuracies_O, width, label="Training", color="#FF87EB")
# Testing accuracy bars:
rects = ax.bar(x + width, test_accuracies_O, width, label="Testing", color="#70CEFF")
ax.bar_label(rects, padding=3)
ax.set_xlabel("Locality")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy of different localities")
ax.set_xticks(x + width / 2, locality)
ax.legend(loc="upper left")
plt.show()
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1738524022642/fd73254e-3624-40bc-886e-c8229ded11c2.png align="center")

```python
1-local:
Training loss:  0.4592314401681531
Testing loss:  0.5045886276497531
Training accuracy:  0.8074534161490683
Testing accuracy:  0.7222222222222222

2-local:
Training loss:  0.43242776810519556
Testing loss:  0.5718358099121
Training accuracy:  0.860248447204969
Testing accuracy:  0.7222222222222222

3-local:
Training loss:  0.42526261814808347
Testing loss:  0.574942133390183
Training accuracy:  0.9316770186335404
Testing accuracy:  0.75
```

The ansatz expansion approach:

```python
def deriv_params(thetas: int, order: int):
    # This function generates parameter shift values for calculating derivatives
    # of a quantum circuit.
    # 'thetas' is the number of parameters in the circuit.
    # 'order' determines the order of the derivative to calculate (1st order, 2nd order, etc.).

    def generate_shifts(thetas: int, order: int):
        # Generate all possible combinations of parameters to shift for the given order.
        shift_pos = list(combinations(np.arange(thetas), order))

        # Initialize a 3D array to hold the shift values.
        # Shape: (number of combinations, 2^order, thetas)
        params = np.zeros((len(shift_pos), 2 ** order, thetas))

        # Iterate over each combination of parameter shifts.
        for i in range(len(shift_pos)):
            # Iterate over each possible binary shift pattern for the given order.
            for j in range(2 ** order):
                # Convert the index j to a binary string of length 'order'.
                for k, l in enumerate(f"{j:0{order}b}"):
                    # For each bit in the binary string:
                    if int(l) > 0:
                        # If the bit is 1, increment the corresponding parameter.
                        params[i][j][shift_pos[i][k]] += 1
                    else:
                        # If the bit is 0, decrement the corresponding parameter.
                        params[i][j][shift_pos[i][k]] -= 1

        # Reshape the parameters array to collapse the first two dimensions.
        params = np.reshape(params, (-1, thetas))
        return params

    # Start with a list containing a zero-shift array for all parameters.
    param_list = [np.zeros((1, thetas))]

    # Append the generated shift values for each order from 1 to the given order.
    for i in range(1, order + 1):
        param_list.append(generate_shifts(thetas, i))

    # Concatenate all the shift arrays along the first axis to create the final parameter array.
    params = np.concatenate(param_list, axis=0)

    # Scale the shift values by π/2.
    params *= np.pi / 2

    return params

n_wires = 8
dev = qml.device("default.qubit", wires=n_wires)

@jax.jit
@qml.qnode(dev, interface="jax")
def circuit(features, params, n_wires=8):
    feature_map(features)
    ansatz(params)
    return qml.expval(qml.PauliZ(0))
```

For each image sample, measure the outputs of each parameterised circuit for each feature, and feed the outputs into a multilayer perceptron:

```python
# Initialize lists to store training and testing accuracies for different derivative orders.
train_accuracies_AE = []
test_accuracies_AE = []

# Loop through different derivative orders (1st order, 2nd order, 3rd order).
for order in range(1, 4):
    print("Order number: " + str(order))

    # Generate the parameter shifts required for the given derivative order.
    to_measure = deriv_params(16, order)

    # Transform the training dataset by applying the quantum circuit with the
    # generated parameter shifts.
    new_X_train = []
    for thing in X_train:
        result = circuit(thing, to_measure.T)
        new_X_train.append(result)

    # Transform the testing dataset similarly.
    new_X_test = []
    for thing in X_test:
        result = circuit(thing, to_measure.T)
        new_X_test.append(result)

    # Train a Multilayer Perceptron (MLP) classifier on the transformed training data.
    clf = MLPClassifier(early_stopping=True).fit(new_X_train, y_train)

    # Print the log loss for the training data.
    print("Training loss: ", log_loss(y_train, clf.predict_proba(new_X_train)))

    # Print the log loss for the testing data.
    print("Testing loss: ", log_loss(y_test, clf.predict_proba(new_X_test)))

    # Calculate and store the training accuracy.
    acc = clf.score(new_X_train, y_train)
    train_accuracies_AE.append(acc)
    print("Training accuracy: ", acc)

    # Calculate and store the testing accuracy.
    acc = clf.score(new_X_test, y_test)
    test_accuracies_AE.append(acc)
    print("Testing accuracy: ", acc)
    print()

locality = ("1-order", "2-order", "3-order")
train_accuracies_AE = [round(value, 2) for value in train_accuracies_AE]
test_accuracies_AE = [round(value, 2) for value in test_accuracies_AE]
x = np.arange(3)
width = 0.25
fig, ax = plt.subplots(layout="constrained")
rects = ax.bar(x, train_accuracies_AE, width, label="Training", color="#FF87EB")
ax.bar_label(rects, padding=3)
rects = ax.bar(x + width, test_accuracies_AE, width, label="Testing", color="#70CEFF")
ax.bar_label(rects, padding=3)
ax.set_xlabel("Order")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy of different derivative orders")
ax.set_xticks(x + width / 2, locality)
ax.legend(loc="upper left")
plt.show()
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1738524176263/f17979db-ac09-4eb7-97a2-af007b24ceea.png align="center")

```python
Order number: 1
Training loss:  0.6917395840118673
Testing loss:  0.6898366117810784
Training accuracy:  0.5093167701863354
Testing accuracy:  0.5555555555555556

Order number: 2
Training loss:  0.6326009058014004
Testing loss:  0.6157803899808801
Training accuracy:  0.7018633540372671
Testing accuracy:  0.6666666666666666

Order number: 3
Training loss:  0.5815839249054562
Testing loss:  0.6016181640099203
Training accuracy:  0.7142857142857143
Testing accuracy:  0.6944444444444444
```

Regarding the hybrid strategy:

```python
# Initialize matrices to store training and testing accuracies for different
# combinations of locality and order.
train_accuracies = np.zeros([4, 4])
test_accuracies = np.zeros([4, 4])

# Loop through different derivative orders (1st to 3rd) and localities (1-local to 3-local).
for order in range(1, 4):
    for locality in range(1, 4):
        # Skip invalid combinations where locality + order exceeds 3 or equals 0.
        if locality + order > 3 or locality + order == 0:
            continue
        print("Locality: " + str(locality) + " Order: " + str(order))

        # Define a quantum device with 8 qubits using the default simulator.
        dev = qml.device("default.qubit", wires=8)

        # Generate the parameter shifts required for the given derivative order and transpose them.
        params = deriv_params(16, order).T

        # Define a quantum node (qnode) with the quantum circuit that will be executed on the device.
        @qml.qnode(dev)
        def circuit(features, params):
            # Generate the Pauli group for the given locality.
            measurements = local_pauli_group(8, locality)
            feature_map(features)
            ansatz(params)
            # Measure the expectation values of the generated Pauli operators.
            return [qml.expval(qml.pauli.string_to_pauli_word(measurement)) for measurement in measurements]

        # Vectorize the quantum circuit function to apply it to multiple data points in parallel.
        vcircuit = jax.vmap(circuit)

        # Transform the training dataset by applying the quantum circuit with the
        # generated parameter shifts.
        new_X_train = np.asarray(
            vcircuit(jnp.array(X_train), jnp.array([params for i in range(len(X_train))]))
        )
        # Reorder the axes and reshape the transformed data for input into the classifier.
        new_X_train = np.moveaxis(new_X_train, 0, -1).reshape(
            -1, len(local_pauli_group(8, locality)) * len(deriv_params(16, order))
        )

        # Transform the testing dataset similarly.
        new_X_test = np.asarray(
            vcircuit(jnp.array(X_test), jnp.array([params for i in range(len(X_test))]))
        )
        # Reorder the axes and reshape the transformed data for input into the classifier.
        new_X_test = np.moveaxis(new_X_test, 0, -1).reshape(
            -1, len(local_pauli_group(8, locality)) * len(deriv_params(16, order))
        )

        # Train a Multilayer Perceptron (MLP) classifier on the transformed training data.
        clf = MLPClassifier(early_stopping=True).fit(new_X_train, y_train)

        # Calculate and store the training and testing accuracies.
        train_accuracies[order][locality] = clf.score(new_X_train, y_train)
        test_accuracies[order][locality] = clf.score(new_X_test, y_test)

        print("Training loss: ", log_loss(y_train, clf.predict_proba(new_X_train)))
        print("Testing loss: ", log_loss(y_test, clf.predict_proba(new_X_test)))
        acc = clf.score(new_X_train, y_train)
        train_accuracies[locality][order] = acc
        print("Training accuracy: ", acc)
        acc = clf.score(new_X_test, y_test)
        test_accuracies[locality][order] = acc
        print("Testing accuracy: ", acc)
        print()

# Locality: 1 Order: 1
# Training loss:  0.29433122335335293
# Testing loss:  0.48158001426002656
# Training accuracy:  0.8944099378881988
# Testing accuracy:  0.7777777777777778

# Locality: 2 Order: 1
# Training loss:  0.32784353109905134
# Testing loss:  0.571967578071357
# Training accuracy:  0.8664596273291926
# Testing accuracy:  0.75

# Locality: 1 Order: 2
# Training loss:  0.20260000718215349
# Testing loss:  0.5550612230165831
# Training accuracy:  0.9409937888198758
# Testing accuracy:  0.75
```

Plotting all the post-variational strategies together:

```python
for locality in range(1, 4):
    train_accuracies[locality][0] = train_accuracies_O[locality - 1]
    test_accuracies[locality][0] = test_accuracies_O[locality - 1]
for order in range(1, 4):
    train_accuracies[0][order] = train_accuracies_AE[order - 1]
    test_accuracies[0][order] = test_accuracies_AE[order - 1]

train_accuracies[3][3] = var_train_acc
test_accuracies[3][3] = var_test_acc

cvals = [0, 0.5, 0.85, 0.95, 1]
colors = ["black", "#C756B2", "#FF87EB", "#ACE3FF", "#D5F0FD"]
norm = plt.Normalize(min(cvals), max(cvals))
tuples = list(zip(map(norm, cvals), colors))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)


locality = ["top qubit\n Pauli-Z", "1-local", "2-local", "3-local"]
order = ["0th Order", "1st Order", "2nd Order", "3rd Order"]

fig, axes = plt.subplots(nrows=1, ncols=2, layout="constrained")
im = axes[0].imshow(train_accuracies, cmap=cmap, origin="lower")

axes[0].set_yticks(np.arange(len(locality)), labels=locality)
axes[0].set_xticks(np.arange(len(order)), labels=order)
plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
for i in range(len(locality)):
    for j in range(len(order)):
        text = axes[0].text(
            j, i, np.round(train_accuracies[i, j], 2), ha="center", va="center", color="black"
        )
axes[0].text(3, 3, '\n\n(VQA)', ha="center", va="center", color="black")

axes[0].set_title("Training Accuracies")

locality = ["top qubit\n Pauli-Z", "1-local", "2-local", "3-local"]
order = ["0th Order", "1st Order", "2nd Order", "3rd Order"]

im = axes[1].imshow(test_accuracies, cmap=cmap, origin="lower")

axes[1].set_yticks(np.arange(len(locality)), labels=locality)
axes[1].set_xticks(np.arange(len(order)), labels=order)
plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
for i in range(len(locality)):
    for j in range(len(order)):
        text = axes[1].text(
            j, i, np.round(test_accuracies[i, j], 2), ha="center", va="center", color="black"
        )
axes[1].text(3, 3, '\n\n(VQA)', ha="center", va="center", color="black")

axes[1].set_title("Test Accuracies")
fig.tight_layout()
plt.show()
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1738524282437/d2386546-e2f4-4b3a-9298-bb603623a186.png align="center")

---

And that's it for this article. Thanks for reading.

For any question or suggestion related to what I covered in this article, please add it as a comment. For special needs, you can contact me [**here.**](http://amm.zanotp.com/contact)

## Sources:

1. [https://arxiv.org/pdf/2307.10560](https://arxiv.org/pdf/2307.10560#page=13&zoom=100,249,193)
    
2. [https://www.nature.com/articles/s41567-020-0932-7](https://www.nature.com/articles/s41567-020-0932-7)
    
3. [https://www.nature.com/articles/s41467-018-07090-4](https://www.nature.com/articles/s41467-018-07090-4)
    
4. [https://arxiv.org/pdf/2009.11001](https://arxiv.org/pdf/2009.11001)
    
5. [https://journals.aps.org/pra/abstract/10.1103/PhysRevA.104.042418](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.104.042418)
    
6. [https://journals.aps.org/pra/abstract/10.1103/PhysRevA.105.052445](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.105.052445)
    
7. [https://iopscience.iop.org/article/10.1088/1367-2630/ac325f](https://iopscience.iop.org/article/10.1088/1367-2630/ac325f)