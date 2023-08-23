---
title: "Blockchain and randomness"
seoTitle: "How to generate randomness on the blockchain: hacks and best practices"
seoDescription: "Getting random numbers on the blockchain used to be a headache for may years. This article covers the evolution of prngs on the blockchain and some hacks"
datePublished: Wed Aug 23 2023 12:17:29 GMT+0000 (Coordinated Universal Time)
cuid: cllnp9w3000070amcaqlr4e6z
slug: blockchain-and-randomness
cover: https://cdn.hashnode.com/res/hashnode/image/stock/unsplash/T9rKvI3N0NM/upload/28c0ced23ce653a91b9b9bde743215c0.jpeg
tags: randomness, blockchain, blockchain-technology, blockchain-development, blockchain-security

---

Getting random numbers on the blockchain used to be a headache for those who wanted to use truly random numbers in a dapp or protocol, and the lotteries that used these pseudo-random numbers were easily hacked by fast and malicious agents. However, the idea of a blockchain whose dapps can function without random numbers was (and is) out of the question, and obtaining random numbers has become easy and relatively secure.

The reason for this difficulty is that the calculations must be deterministic in order to be replayed in a decentralized manner, and any data that can serve as random sources is also available to an attacker.

In this article, I'll review the solutions that blockchain engineers have developed in the past to address this problem and their weaknesses and conclude with the simplest and most commonly used method currently available.

## Pseudo-randomness from unknowable at the time of transacting information

One of the first sources of entropy that blockchain engineers used was the block timestamp, a global variable that represents the timestamp of the current block in which the contract is executed. This timestamp is a Unix timestamp that indicates the number of seconds that have elapsed since January 1, 1970 (UTC) and provides information about when the block was mined.

The problem with block timestamps is that miners have the ability to influence them as long as the timestamp doesn't precede that of the parent block. Although timestamps are usually quite accurate, there is a potential problem if a miner benefits from inaccurate timestamps. In such cases, the miner could use his mining power to create blocks with incorrect timestamps and thus manipulate the results of the random function to his advantage.

For example, imagine a lottery in which a random bidder is selected from a set of bidders by a function that uses the timestamp of a block as the source of the randomness:  a miner may enter the lottery and then modify the timestamp value to increase his chances of winning.

While these attacks may sound anachronistic, they are not beyond the realm of possibility. In fact, Feathercoin was the victim of a time-warp attack in 2013. In it, a group of miners exploited a vulnerability in Feathercoin's mining algorithm that allowed them to manipulate the timestamps of blocks, resulting in the rapid creation of new blocks. The attack undeniably caused significant damage to Feathercoin's value and reputation.

Still, one might think that using the block hash as a source of entropy or other block information that is generally unknown at the time of the transaction is a good idea. However, similar implementations have a major problem: they rely on publicly available information, which means that malicious actors can increase the probability of winning the lottery with an attack similar to the time-warp attack. This is because these quantities can be read and manipulated by any other transaction within the same mining block if the attacker is also a miner.

Even using a sophisticated combination of all information unknown at the time of the transaction is not a good idea: it makes the attack much more difficult, but does not make the protocol as secure as other methods do.

## Randomness from off-chain data: oracles and APIs

I hope you have been convinced that using on-chain information is not a good practice when security is a crucial feature. What can we do to get an unpredictable random number for our lottery?

We can turn our attention to off-chain data, i.e. use the data that an API or oracle provides. For example, if we have an API that provides the temperature in a particular city, we can use it to calculate the remainder when dividing the number of tipsters and use the result as a random number. The temperature in a particular city changes frequently, and if the API's answer is updated frequently, the likelihood of a malicious agent guessing the number is very low.

Although this is a better solution than using on-chain data, it is not the best available because we centralize our random source and the smart contract is useless if the API is corrupted.

Moreover, no one would trust the lottery contract, since it can be assumed that the API is programmed to always return the same set of values and the protocol is no longer trustless.

Despite these drawbacks, oracles and APIs have been widely used to obtain data outside the chain, and are sometimes still used. It's worth noting that combining the results of different APIs and oracles can result in almost unpredictable output, which can be a good deal for small dapps or protocols that don't rely entirely on randomness. The reputation of the data provider is also important in this case.

The most important attack on APIs and oracles is so-called oracle manipulation, in which vulnerabilities in a blockchain oracle are exploited to make it report inaccurate information about events outside the chain. This attack is often part of a broader attack on a protocol, as malicious actors can cause a protocol’s smart contracts to execute based on false input or in a way that is advantageous to them.

## Verifiable random functions (VRFs)

Steering clear of intricate mathematics,Verifiable Random Functions (VRFs ) can be described as public key pseudorandom functions. Put simply, these functions produce outputs that appear pseudorandom based on a given seed and mimic the behavior of true random outputs (if you want to dig deeper, read [this](https://amm.zanotp.com/an-introduction-to-prngs-with-python-and-r) article). The real power of VRFs is their ability to prove the correctness of their output calculations. The possessor of the secret key is the only one able to compute the output of the function (i.e., the random output) along with a corresponding proof for any input value. Conversely, anyone else who has the proof and the corresponding public key can verify the exact computation of this output. However, this information is not sufficient to derive the secret key.

One of the most commonly used VRFs is the Chainlink VRF, which relies on a decentralized oracle network (i.e., a set of oracles that receive data from multiple reliable sources) to enhance existing blockchains by providing verified off-chain data.

Chainlink VRF enables the generation of random numbers within smart contracts, enabling blockchain developers to create improved user experiences by incorporating unpredictable outcomes into their blockchain-powered applications. In addition, Chainlink VRF is immune to tampering, whether done by node operators, users, or malicious entities.

### [To](http://entities.To) go further

To be an outstanding blockchain developer it's not necessary to know everything about VRFs, however for the curious ones I suggest [Micali, Rabin, Vadhan (1999)](https://dash.harvard.edu/bitstream/handle/1/5028196/Vadhan_VerifRandomFunction.pdf) and the [Chainlink VRF docs](https://docs.chain.link/vrf/v2/introduction).

---

And that's it for this article.

Thanks for reading.

For any question or suggestion related to what I covered in this article, please add it as a comment. For special needs, you can contact me [**here**](http://amm.zanotp.com/contact).