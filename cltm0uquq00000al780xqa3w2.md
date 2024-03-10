---
title: "Quantum Blockchain"
seoTitle: "quantum computing"
seoDescription: "This article investigates the intersection between quantum technology and blockchain: quantum blockchain. "
datePublished: Sun Mar 10 2024 21:23:44 GMT+0000 (Coordinated Universal Time)
cuid: cltm0uquq00000al780xqa3w2
slug: quantum-blockchain
cover: https://cdn.hashnode.com/res/hashnode/image/stock/unsplash/qDG7XKJLKbs/upload/bc9d0dfed6abfa3457e9e876e362fb87.jpeg
tags: blockchain, quantum, quantum-computing, blockchain-technology, blockchain-security, quantum-cryptography, quantum-blockchain, quantum-money

---

In the landscape of technological innovation, two disruptive forces stand out: quantum computing and blockchain. While each has made significant strides independently, their convergence holds the promise of revolutionizing cryptography and reshaping the foundations of digital trust. At the heart of this synergy lies the concept of quantum blockchain, a novel blockchain model infused with quantum cryptographic principles.

Blockchain technology, epitomized by cryptocurrencies like Bitcoin and Ethereum, has redefined trust in digital transactions. Its decentralized ledger system offers immutable records, resistant to tampering and censorship, transforming industries beyond finance. Meanwhile, quantum computing, leveraging quantum mechanics, offers exponential computational power, poised to tackle problems deemed infeasible by classical computers.

While the two technologies seems unrelated, a profound connection exists between quantum computing and blockchain and this blog post intoduces you to quantum blockchain after a brief digression on fundamental concepts of quantum computing and quantum cryptography.

## Quantum computing

As you may know all the information a computer store and process are just interminable strings of 0s and 1s, the so-called bits. Quantum computing is a completely different computational paradigm, relying on quantum bits (also called qubits), which can exist in a superposition of states, enabling them to represent both 0 and 1 simultaneously.

While this may seems a logical contraddiction, according to postulates of quantum mechanics, the state of a system is described as a linear combination of all possible states until measured, when the state collapse and is deterministically defined. This superposition property allows qubits to hold exponentially more information than classical bits. Furthermore, qubits can exhibit another peculiar quantum behavior called entanglement, where the state of one qubit becomes correlated with the state of another qubit. The third ingredient that makes a quantum computer faster than a classical one is quantum interference. Quantum interference occurs when the probability amplitudes of different quantum states interfere constructively or destructively, resulting in the amplification or reduction of certain outcomes. In quantum computing, this interference allows for the manipulation and processing of information in a highly efficient manner.

These phenomena enables quantum computers to outperform classical computers in solving certain types of problems, particularly those that require extensive exploration of solution spaces. Thus, superposition, entanglement, and quantum interference collectively contribute to the computational power and speed of quantum computers, offering the potential for revolutionary advancements in various fields of science and technology.

## Quantum cryptography

One of the most fascinating and more important application of quantum technologies lies in quantum cryptography, a subset of quantum information science, that aims to utilize the principles of quantum mechanics to secure communication channels in a fundamentally different way than classical cryptographic methods.

In fact traditional cryptographic techniques rely on mathematical complexity, such as factorization or discrete logarithm problems, for securing data transmission. The idea here is to consider a problem that may take ages to a classical computer to solve due to its complexity and use this difficulty as the basis for encryption. Obviously having more and more powerful computers poses a threat to the security of these classical encryption methods. Moreover quantum computers will be able to efficiently solve these mathematical problems using algorithms like Shor's algorithm, rendering traditional encryption schemes obsolete.

Quantum cryptography, on the other hand, offers a solution that is fundamentally secure, regardless of the computational power of the adversary. By exploiting the properties of quantum mechanics, such as the superposition and entanglement of quantum states, quantum cryptography provides a means for two parties to communicate with absolute secrecy. Quantum Key Distribution (QKD), one of the most prominent applications of quantum cryptography, allows two parties to share a secret cryptographic key with the assurance that any attempt to intercept the key will be detected. This is achieved through the use of quantum states to encode the key information, making it impossible for an eavesdropper to gain knowledge of the key without disturbing the quantum states and revealing their presence.

![Quantum Key Distribution Technology | 24 Feb 2022](https://www.drishtiias.com/images/uploads/1645696513_Quantum_Key_Distribution_Work_Drishti_IAS_English.png align="left")

As such, quantum cryptography offers a level of security that is unparalleled by classical cryptographic methods, making it an essential tool for ensuring the confidentiality and integrity of sensitive information in the digital age.

## Quantum money

Before diving into quantum blockchain here I want to discuss another intriguing application of quantum technologies, somehow related to the blockchain and the cryptocurrencies: quantum money.

The concept of quantum money traces back to the early days of quantum information theory and cryptography, with theoretical proposals emerging in the 1970s and gaining momentum in subsequent decades. One of the pioneering works in this field was proposed by physicist Stephen Wiesner which was published in a [scientific journal](http://users.cms.caltech.edu/~vidick/teaching/120_qcrypto/wiesner.pdf) in 1983.

Wiesner's idea involved using quantum states to encode information on banknotes, making them effectively unforgeable due to the inherent properties of quantum mechanics. Specifically, Wiesner proposed a scheme where each banknote would contain a unique quantum state, which could not be precisely duplicated or measured without disturbing its state. This would make counterfeiting quantum money practically impossible, as any attempt to copy or measure the quantum state would inevitably alter it, thus revealing the counterfeit attempt.

![photo by NIST](https://www.nist.gov/sites/default/files/styles/480_x_480_limit/public/images/public_affairs/colloquia/011711_lr.jpg?itok=LWulsDZE align="center")

Despite the theoretical appeal of Wiesner's proposal, the practical implementation of quantum money remains a significant challenge. Generating and manipulating quantum states with the precision and reliability required for quantum money presents formidable technical hurdles. Additionally, quantum systems are inherently fragile and susceptible to environmental noise, which could compromise the security of quantum money schemes.

Therefore, similarly to cryptocurrencies, quantum money seeks to provide an unforgeable form of currency by exploiting the fundamental principles of quantum mechanics.

## Quantum blockchain

As you may already know, a blockchain functions as an immutable ledger where data is stored in the form of transactions, interconnected through a Merkle tree, and organized into blocks linked by hash functions. This network operates in a decentralized manner, with each node retaining a copy of the growing chain of blocks. Consensus protocols determine the addition of new blocks and establish agreement on the block sequence. Typically, the blockchain process begins with users broadcasting transactions, which are then verified and organized into a new block according to specific consensus rules, such as proof-of-work or proof-of-stake. Participants, often referred to as "miners" in systems like Bitcoin, compete to create the next block, with the successful miner being rewarded. The longest chain of blocks is considered definitive, providing a basis for consensus.

One of the main features of blockchain is that if any block within the chain is altered, it invalidates all subsequent blocks. Consequently, nodes in the blockchain network reject the tampered version and continue to work on the version supported by the majority.

Moreover access control in blockchains relies on public-key cryptography, where users safeguard private keys as passwords and use public keys as account identifiers. Transactions are authenticated using signatures generated with private keys, which are verified by network nodes against the corresponding public keys. Once you are familiar with the above information, you are ready to explore quantum blockchains.

Quantum blockchain typically refers to a variety of protocols, including classical blockchains with quantum-resistant cryptography, hybrid blockchains leveraging Quantum Key Distribution networks (just hybrid blockchains from now on), and fully quantum blockchains operating in the realm of quantum computing.

Hybrid blockchains aim to tackle the fact that public-key cryptography is not quantum resistant (we already mentioned Shor’s algorithm), therefore substitutig publik-key cryptography with the already mentioned Quantum Key Distribution.

Quantum blockchains on the other hand are more variegate and substitute some core block of a classical blockchain with a quantum counterpart. For example, [Rajan, D., & Visser, M. (2019)](https://arxiv.org/pdf/1804.05979.pdf), whose quantum blockchain is regarded as a pioneering theoretical work, replaces the functionality of time-stamped blocks and hash functions linking them with a temporal entangled state, which offers a fairly interesting advantage: since the sensitivity towards tampering is significantly amplified, meaning that the full local copy of the blockchain is destroyed if one tampers with a single block (due to entanglement) while on a classical blockchain only the blocks following the compromised block are changed , which leaves it open to vulnerabilities. However let's dive a little more in the formulation of both the blockchain and the network as proposed in [Rajan, D., & Visser, M. (2019)](https://arxiv.org/pdf/1804.05979.pdf).

### Blockchain

This subsection explores the implementation of a quantum version of a block and a blockchain, utilizing temporally entangled states (a concept in quantum mechanics where the quantum states of multiple particles become correlated over time, rather than in space).

Entanglement, essentially the inseparability of distinct states, forms the basis for capturing the chain-like structure. Consequently, the blockchain can be viewed as an entangled quantum state, with a block's timestamp emerging from the immediate absorption of the first qubit of a block.

Constructing the blockchain from a series of entangled states involves amalgamating the blocks into a specialized entangled state known as a Greenberger–Horne–Zeilinger (GHZ) state.

### Network

After establishing the blockchain, additional components are necessary for a functional blockchain system, notably a protocol for disseminating the blockchain's state to all network nodes. Since the blockchain's state is quantum in nature, a quantum channel must replace the classical one, with digital signatures implemented through Quantum Key Distribution (QKD) protocols.

Similar to classical blockchain systems, each node in a quantum blockchain setup must possess a copy of the blockchain, and new blocks must undergo verification before integration into each node's blockchain.

### Conclusion

Quantum blockchain is still an area of research and is an author’s opinion that, given the rise of classical blockchains and the realistic development of a global quantum network, quantum blockchain can potentially open the door to a new research frontier in quantum information science as well as new business possibilities.

Thanks for reading. This article does not want to be exhaustive on the topic and is no more than an introduction to quantum blockchain. To go further there are resources online and the source section below is a good starting point.

Sources:

* [Weisner, S. (1983) Conjugate Coding. ACM SIGACT News, 15, 78-88](http://users.cms.caltech.edu/~vidick/teaching/120_qcrypto/wiesner.pdf)
    
* [Rajan, D., & Visser, M. (2019)](https://arxiv.org/pdf/1804.05979.pdf)
    
* [Ringbauer, M.; Costa, F.; Goggin, M.E.; White, A.G.; Fedrizzi, A. Multi-time quantum correlations with no spatial analog’. NPJ Quantum Inf. 2018, 4 , 37.](https://www.nature.com/articles/s41534-018-0086-y)