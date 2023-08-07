---
title: "An introduction to PRNGs with Python and R"
seoTitle: "An introduction to PRNGs with Python and R"
seoDescription: "How can a computer, a perfectly deterministic machine, produce random numbers? Let's dive into random number generators with Python and R."
datePublished: Sun Jan 29 2023 23:00:39 GMT+0000 (Coordinated Universal Time)
cuid: cldhzjjc9003rlsnv8lpn7v7f
slug: an-introduction-to-prngs-with-python-and-r
cover: https://cdn.hashnode.com/res/hashnode/image/stock/unsplash/aaSTQ-wY5DQ/upload/2a831240ef9f9415c1cb90dc01525477.jpeg
ogImage: https://cdn.hashnode.com/res/hashnode/image/upload/v1675064557839/4ed24b84-0e05-4227-ba6f-624745354fa7.jpeg
tags: statistics, randomness, cryptography, random-numbers, prngs

---

> Life's most important questions are, for the most part, nothing but probability problems.
> 
> Pierre-Simon de Laplace

# Introduction

Imagine this scenario: you and your brother want to go to the cinema. Two movies are played: Interstellar (the one you want to see) and A Clockwork Orange (that your brother wants to see).

The classic solution to this problem is flipping a coin, but since we are not unimaginative people (or we don't have a coin) we may want to find a more elegant solution.

Thus let's define a program that decides what to see in R and Python. The program will generate a number between 0 and 1. If this number is closer to 0 we watch Interstellar. Otherwise, A Clockwork Orange is chosen.

```python
import random as rd

x = rd.uniform(0, 1)

if x < .5:
	print("Interstellar")
else:
	print("A Clockwork Orange")
```

We now do the same in R:

```r
x <- runif(1)

ifelse(x < .5, "Interstellar", "A Clockwork Orange")
```

Fair enough, but there is something paradoxical in the previous examples: a computer, a perfectly *deterministic machine*, is creating something *randomly*.

Fair enough, but there is something paradoxical in the previous examples: a computer, a perfectly *deterministic machine*, is creating something *randomly*.

In this article, I want to introduce you to **pseudorandom number generators** and their application.

## Determinism versus randomness

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1674834068845/a7d1ce92-4841-42cc-9cc3-18101b50b441.jpeg align="center")

Above I wrote "computer, a perfectly *deterministic machine*", but what does it mean to be deterministic?

In brief, computers are deterministic because they follow a set of instructions, or a program, in a predictable manner, i.e. given some inputs they return always the same output. The paradox lies in the fact that in the above example, `x = rd.uniform(0, 1)` and `x <- runif(1)` return a different value every time the line is compiled.

Are `x = rd.uniform(0, 1)` and `x <- runif(1)` exceptions to the deterministic property of computers?

The answer is no, and in a minute I'll explain the reasons behind that.

## What is randomness

Before diving into PRNGs we need to define **randomicity**. We usually call random a sequence of numbers with the following trait:

* **lack of pattern**: a random sequence should not have any discernible structure;
    
* **independence**: the numbers in a random sequence should not be affected by one another;
    
* **unpredictability**: a random sequence of numbers should not be able to be predicted or reconstructed.
    

It's important to notice that randomicity is a complex concept and it's hard to quantify it precisely. Therefore it's common to use statistical tests to evaluate the randomness of a sequence of numbers, but this is beyond the scope of this article.

Random number generators are mathematical algorithms or mechanical devices that produce a sequence that follows the above properties.

As you may suppose, there are two types of random number generators:

* **true random number generators** (TRNGs from now on)
    
* **pseudorandom number generators** (PRNGs from now on)
    

In this article, I'll just cover PRNGs but be aware that TRNGs exist and have important applications in many fields such as gaming, gambling and cryptography.

# Pseudorandom number generators (PRNGs)

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1674834995478/6742e995-0396-4aa0-827b-2c3363249970.jpeg align="center")

As the name suggests, pseudorandom number generators are a type of software used to generate a sequence of numbers that *mimic* the properties of truly random numbers. The algorithm takes an initial input (the **seed**) which produces a sequence. The **seed** is what *determines* the sequence of numbers, for example, if we set the `1234` seed, compiling multiple times the following lines of code, `x` remains the same. In Python this is:

```python
import random as rd

rd.seed(1234)

x = rd.uniform(0, 1)

print(x)
```

The same is true for the following R code:

```r
set.seed(1234)

x <- runif(1)

print(x)
```

## Properties

The goodness of a PRNG is given by its properties. The most important properties are:

* **periodicity**: PRNGs will generate a sequence of numbers that repeats itself after a certain number of iterations, known as the *period*. A PRNG with a long period is more desirable than one with a shorter period;
    
* **uniformity**: PRNGs generate numbers that are distributed uniformly across the range of possible values;
    
* **independence**: the numbers generated by a PRNG should be independent of one another;
    
* **randomness**: the numbers generated by a PRNG should not have any discernible patterns;
    
* **seed-ability**: PRNGs should be able to be seeded with an initial value in order to produce a different sequence.
    

## Two PRNGs algorithms

In this section, I want to present the most known PRNGs algorithms to practically show how PRNGs look like. I will present the **middle square algorithm** and the **linear congruential generator algorithms**.

### Middle square algorithm

Proposed by von Neumann, the middle square algorithm takes a **seed** that is squared and its midterm is fetched as the random number. Let's discuss an example and then implement it in Python and R.

| seed | square | random number |
| --- | --- | --- |
| 12 | 0**14**4 | 14 |
| 33 | 1**08**9 | 08 |
| 24 | 0**57**6 | 57 |
| 66 | 4**35**6 | 35 |

Usually, the algorithm is repeated more than once, i.e the random number becomes the new seed and is then squared and its midterm becomes the random number and so on.

Here is an implementation in Python:

```python
import numpy as np

def middle_square_algo(seed):
    
    # first of all we square the seed
    square = str(np.square(seed))
    
    # then we need to take the mid-term, we have two possibilities
    # the square may have an even number of digits:
    if len(square) % 2 == 0:
        half_str = int(len(square) / 2)
    
    # the number has an odd number of digits:
    else:
        half_str = int(len(square) / 2 - .5)
        
        
    mid = square[half_str - 1 : half_str + 1]
    return int(mid)

# finally the testing:

print(middle_square_algo(12))

#> 14
```

And here is the R code:

```r
middle_square_algo <- function(seed){
  
  # first of all we square the seed
  square <- seed^2
  
  # we now need to get the number of digits of square
  len <- nchar(square)
  
  # we have two possible scenarios
  # len is even:
  if(len %% 2 == 0){
    
    half_square <- len / 2
    
  # len is odd:  
  } else{
    
    half_square <- len / 2 + .5
    
  }
  square <- as.character(square)
  mid <- substr(square, half_square, half_square + 1)
  
  return(as.double(mid))
}

# finally the testing:
print(middle_square_algo(33))

#> 8
```

Assuming now that we want to loop more than one time the algorithm, the Python code is:

```python
def middle_square_algo_deep(seed, deep):
    
    # we just need to repeat what we did before but more than one time
    
    for rep in range(deep):
        seed = int(middle_square_algo(seed))
    return seed

# finally the testing:      
middle_square_algo_deep(33, 3)

#> 9
```

And similarly, the R code is:

```r
middle_square_algo_deep <- function(seed, deep=2){
  
  # we just need to repeat what we did before but more than one time
  
  for( rep in 1:deep){
    seed <- middle_square_algo(seed)
  }
  
  return(seed)
}

# finally the testing:
middle_square_algo_deep(33, 3)

#> 9
```

The most important weakness of this algorithm is that it needs an appropriate starting seed. In fact, some seeds have a really short period.

For example the seed `50` has a really short period (1) as shown in the following lines of code:

```r
middle_square_algo_deep(50, 1)
#> 50
 
middle_square_algo_deep(50, 2)
#> 50
 
middle_square_algo_deep(50, 3)
#> 50
 
middle_square_algo_deep(50, 4)
#> 50
```

### Linear congruential generators

The linear congruential generators (LCGs) are a family of PRNGs and are probably the most used approach to generate pseudorandom numbers. The algorithms are defined by a linear congruential equation as the following one:

$$x_{n+1} = ax_n + b \space \space mod(y)$$

where \\(a\\), \\(b\\) and \\(c\\) are positive integers and we also need a **seed**.

Let's now consider (and then implement) a particular LCG: the **Lagged Fibonacci Generator** (LFG).

$$x\_{n+1} = a\_1 x\_{n-1} + a\_2 x\_{n-j} + b \space \space mod(y)$$

We just need to provide LFG from \\(x\_1\\) to \\(x\_{max(i, j)+1}\\) and it will generate a pseudorandom sequence of numbers.

Let me make an example to clear your mind. Let the following equation be our LFG:

$$x\_{n+1} = x\_{n-3} + x\_{n-5} \space \space mod(10)$$

and let's say we want to generate a sequence of random numbers between 1 and 9 from the initial seed \[4, 2, 9, 5, 5\].

The sequence starts from \\(x\_6\\) (you can easily prove that the values before \\(x_6\\) don't exist by cause of \\(max(i, j) = 5\\)).

Thus the sequence is:

$$x\_{6} = x\_{3} + x\_{1} \space \space mod(10) = 9 + 4 \space \space mod(10) = 3$$

$$x\_{7} = x\_{4} + x\_{2} \space \space mod(10) = 5 + 2 \space \space mod(10) = 7$$

$$x\_{8} = x\_{5} + x\_{3} \space \space mod(10) = 5 + 9 \space \space mod(10) = 4$$

and so on.

We now implement the LFG in Python and R. In Python the algorithm is something like this:

```python
def lagged_fib_gen(seed, i, j, mod, length, a_1 = 1, a_2 = 1, c = 0):
    l_f = seed

        # we suppose that i < j

    for rep in range(max([i, j]) + 1, length + 1):
                
        x = (a_1 * l_f[rep - i - 1] + a_2 * l_f[rep - j - 1]) % 10
        l_f.append(x)

    return l_f

# finally the testing:
lagged_fib_gen([4, 2, 9, 5, 5], 3, 5, 10, 10)

#> [4, 2, 9, 5, 5, 3, 7, 4, 8, 2]
```

In R the algorithm is:

```r
lagged_fib_gen <- function(seed, i, j, mod, length, a_1 = 1, a_2 = 1, c = 0){
  
  l_f <- seed
  
  for(rep in (max(c(i, j))+1):length){
    
    x <- (a_1 * l_f[rep - i] + a_2 * l_f[rep - j]) %% mod
    l_f[rep] <- x
  }
  return(l_f)
}

# finally the testing:
lagged_fib_gen(c(4, 2, 9, 5, 5), 3, 5, 10, 10)

#> 4 2 9 5 5 3 7 4 8 2
```

As for the **middle square algorithm**, the efficiency of LGCs depends on the chosen parameters.

# Applications of PRNGs

Now we know what PRNGs are, but what are they used for? Well they have many applications, some examples include:

* cryptography: random numbers are used to generate encryption keys (the PRNGs used in cryptography are much more complex than the two I showed before);
    
* modelling: many scientific simulations use random numbers to represent uncertainty;
    
* gaming: random numbers are used to make games less predictable and complex (e.g. biomes generation in Minecraft);
    
* randomized algorithms: some algorithms use randomness to solve problems more efficiently (e.g. the famous Randomized Hill Climbing algorithm).
    

# To go further

As you may imagine, the world of PRNGs is quite vast and complex and has applications in almost every field of science. This article doesn't want to be exhaustive on the topic and is no more than a gentle introduction to PRNGs. To go further there are many resources online but [The Art of Computer Programming - Seminumerical Algorithms](https://seriouscomputerist.atariverse.com/media/pdf/book/Art%20of%20Computer%20Programming%20-%20Volume%202%20(Seminumerical%20Algorithms).pdf) by D. Knuth and [this](https://cran.r-project.org/web/packages/randtoolbox/vignettes/fullpres.pdf) CRAN vignette are great starting points.

Thanks for reading.

For any question or suggestion related to what I covered in this article, please add it as a comment. For special needs, you can contact me [here](http://www.zanotp.com/contact).