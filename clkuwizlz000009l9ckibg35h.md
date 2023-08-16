---
title: "Testing smart contracts: unit tests and invariant tests"
seoTitle: "Advanced testing tecniques"
seoDescription: "Testing is a crucial part of a blockchain engeneer life as it helps ensuring the security, functionality, and reliability of smart contracts"
datePublished: Thu Aug 03 2023 08:35:11 GMT+0000 (Coordinated Universal Time)
cuid: clkuwizlz000009l9ckibg35h
slug: testing-smart-contracts
cover: https://cdn.hashnode.com/res/hashnode/image/stock/unsplash/FnA5pAzqhMM/upload/64dc6ede535c99c4686ca6f1df72f553.jpeg
tags: security, testing, solidity, smart-contracts, foundry

---

Testing plays a vital role in ensuring the security, functionality, and reliability of smart contracts and being able to write some goods test can save not only a lot of time but also a lot of money. In this article, we will discuss two types of testing methodologies: unit tests and invariant tests.

Note that I assume a basic knowledge of the Solidity language and Foundry framework, however, even someone without this knowledge should be able to follow along.

## Blockchain 101: smart contracts

In simple words, smart contracts are like digital agreements that automatically execute and enforce themselves when certain conditions are met. We can liken smart contracts to vending machines: once they receive the right inputs, they automatically execute the agreement.

For example, to decentralize a lottery, we will write a function that takes in input the amount paid by a particular player and a function that, once a particular condition is met (e.g. if the number of players is 10 or if one day has passed), it generates a pseudo-random number between 0 and the number of players and pays the amount to the selected winner.

These digital agreements can be used for various purposes, such as transferring money, buying and selling assets, or even voting in elections. Since smart contracts run on the blockchain, they are tamper-resistant and transparent (or at least they should be). Nevertheless, not all blockchain developers pay attention to the contract doing what it is supposed to do, and in fact, the number of hacked or tampered smart contracts is surprisingly high.

To prevent this, it is important to develop a comprehensive testing strategy that includes both unit tests and invariant tests.

## Set up

First of all, we need to set up the foundry environment:

```solidity
forge init
```

Then we need a contract to test. The following lines of code implement a lottery as the one described before. Note that for simplicity the winner should be the first player (`players[0]`) that joins the lottery (pseudo-random numbers on the blockchain is a big theme for an upcoming article) and that the lottery ends once there are at least 5 participants and the owner of the lottery calls the function `endTheLottery`.

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

contract Lottery {
    error Lottery__notEnoughEthSent(uint256 amount);
    error Lottery__notTheOwner(address sender);
    error Lottery__notEnoughtPlayers();
    error Lottery__invalidTransaction();

    uint256 immutable i_lotteryPriceInEth;
    address owner;
    address[] players;
    address winner;

    modifier onlyOwner() {
        if (msg.sender != owner) revert Lottery__notTheOwner(msg.sender);
        _;
    }

    modifier moreThanFivePlayers() {
        if (players.length < 5) revert Lottery__notEnoughtPlayers();
        _;
    }

    constructor(uint256 lotteryPriceInEth) {
        i_lotteryPriceInEth = lotteryPriceInEth;
        owner = msg.sender;
    }

    function joinLottery() public payable {
        if (msg.value < i_lotteryPriceInEth)
            revert Lottery__notEnoughEthSent(msg.value);
        players.push(msg.sender);
    }

    function endTheLottery() public onlyOwner moreThanFivePlayers {
        if (players.length % 2 == 0) {
            winner = players[1];
        }
        if (players.length % 2 == 1) {
            winner = players[0];
        }
        (bool success, bytes memory data) = payable(winner).call{
            value: address(this).balance
        }("");
        if (!success) revert Lottery__invalidTransaction();
    }

    function transferOwnership(address newOwner) public {
        owner = newOwner;
    }

    function getNumberOfPlayer() public view returns (uint256) {
        return players.length;
    }

    function getPlayer(uint256 index) public view returns (address) {
        return players[index];
    }

    function getWinner() public view returns (address) {
        return winner;
    }
}
```

In this first version of the contract, a couple of things are not correct and by testing we should be able to spot them.

## Unit tests

Unit tests are deterministic tests, i.e. they produce deterministic results, are easy to debut and are used to assert particular behaviors of the contract. Before we write any unit test we need a deployer script as the following one:

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import {Script} from "../lib/forge-std/src/Script.sol";
import {Lottery} from "src/Lottery.sol";

contract DeployLottery is Script {
    Lottery lottery;

    function run() public returns (Lottery) {
        vm.startBroadcast();
        lottery = new Lottery(1 ether);
        vm.stopBroadcast();
        return lottery;
    }
}
```

Suppose now we want to assess that the modifier `onlyOwner` is doing his job (which is to prevent addresses different from the owner to call the `endLotteryFunction`), what we need to do is:

* deploy the contract;
    
* transfer the ownership of the contract calling the `transferOwnership` function;
    
* prank an address (different from the owner) and try to call the `endTheLottery` function;
    
* assert that the contract trows the `Lottery__notTheOwner` error.
    

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import {Test} from "lib/forge-std/src/Test.sol";
import {Lottery} from "src/Lottery.sol";
import {DeployLottery} from "script/DeployLottery.s.sol";

contract LotteryTest is Test {
    address player0 = makeAddr("Alice");
    address player1 = makeAddr("Bob");
    address player2 = makeAddr("Carl");
    address player3 = makeAddr("David");
    address player4 = makeAddr("Eleonor");
    address owner = makeAddr("Owner");
    uint256 public constant BALANCE = 100 ether;

    Lottery lottery;

    function setUp() public {
        DeployLottery deployer = new DeployLottery();
        vm.deal(player0, BALANCE);
        vm.deal(player1, BALANCE);
        vm.deal(player2, BALANCE);
        vm.deal(player3, BALANCE);
        vm.deal(player4, BALANCE);
        vm.deal(owner, BALANCE);
        lottery = deployer.run();
        lottery.transferOwnership(owner);
    }

    function testOnlyOwnerCanEndTheLottery() public {
        vm.expectRevert(
            abi.encodeWithSelector(
                Lottery.Lottery__notTheOwner.selector,
                player0
            )
        );

        vm.startPrank(player0); // not the owner
        lottery.endTheLottery();
        vm.stopPrank();
    }
}
```

Since `player0` is the caller of `endTheLottery`, the contract trows the `Lottery__notTheOwner` error, as expected:

```bash
Running 1 test for test/LotteryTest.t.sol:LotteryTest
[PASS] testOnlyOwnerCanEndTheLottery() (gas: 13825)
Test result: ok. 1 passed; 0 failed; 0 skipped; finished in 1.22ms
```

Another classical use of unit tests is for asserting a particular relation between two variables. For example, let's assert that the number of players is five in the following script:

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import {Test} from "lib/forge-std/src/Test.sol";
import {Lottery} from "src/Lottery.sol";
import {DeployLottery} from "script/DeployLottery.s.sol";

contract LotteryTest is Test {
    address player0 = makeAddr("Alice");
    address player1 = makeAddr("Bob");
    address player2 = makeAddr("Carl");
    address player3 = makeAddr("David");
    address player4 = makeAddr("Eleanor");
    address owner = makeAddr("Owner");
    uint256 public constant BALANCE = 100 ether;

    Lottery lottery;

    function setUp() public {
        DeployLottery deployer = new DeployLottery();
        vm.deal(player0, BALANCE);
        vm.deal(player1, BALANCE);
        vm.deal(player2, BALANCE);
        vm.deal(player3, BALANCE);
        vm.deal(player4, BALANCE);
        vm.deal(owner, BALANCE);
        lottery = deployer.run();
        lottery.transferOwnership(owner);
    }

    function testOnlyOwnerCanEndTheLottery() public {
        vm.expectRevert(
            abi.encodeWithSelector(
                Lottery.Lottery__notTheOwner.selector,
                player0
            )
        );

        vm.startPrank(player0); // not the owner
        lottery.endTheLottery();
        vm.stopPrank();
    }

    function testAssertNumberOfPlayers() public {
        vm.startPrank(player0);
        lottery.joinLottery{value: 1 ether}();
        vm.stopPrank();
        vm.startPrank(player1);
        lottery.joinLottery{value: 1 ether}();
        vm.stopPrank();
        vm.startPrank(player2);
        lottery.joinLottery{value: 1 ether}();
        vm.stopPrank();
        vm.startPrank(player3);
        lottery.joinLottery{value: 1 ether}();
        vm.stopPrank();
        vm.startPrank(player4);
        lottery.joinLottery{value: 1 ether}();
        vm.stopPrank();

        uint256 expectedNumberOfPlayers = 5;
        uint256 numberOfPlayers = lottery.getNumberOfPlayer();

        assertEq(expectedNumberOfPlayers, numberOfPlayers);
    }
}
```

As expected, `assertEq(expectedNumberOfPlayers, numberOfPlayers);` is true and the test passed:

```bash
Running 2 tests for test/LotteryTest.t.sol:LotteryTest
[PASS] testAssertNumberOfPlayers() (gas: 192928)
[PASS] testOnlyOwnerCanEndTheLottery() (gas: 13880)
Test result: ok. 2 passed; 0 failed; 0 skipped; finished in 2.34ms
```

Note that these are only two simple cases and we haven't tested edge cases (for example the contract has undesired behavior after the first lottery concludes as the address array is never reinitialized).

As we saw, unit tests are particularly powerful tests in particular when the contract is quite simple. If the contract has some complex functions or inherits from other contracts we may want to conduct a different type of test: the invariant tests.

## Invariant tests

Invariant tests are a form of stochastic testing, meaning the results may vary across test runs (unless the same seed is set). In other words, performing an invariant test means supplying random data to the contract functions trying to individuate some unexpected behavior.

The part of my reader that *actually* read the contract may have noticed that the function `endTheLottery` does something undesired. In fact, if the length of `players` is an even number (`%` is the modulus operator), the contract behaves correctly (remember that for simplicity we want the first to join the lottery to be the winner), but if the number is odd the winner is `player[1]` (i.e. the second one who joined the lottery).

It appears that the victory of `player[0]` should be an invariant property of the contract. Since many contracts have at least an invariant property and testing these properties with unit tests may be difficult or impossible (especially for complex contracts), knowing how to perform invariant tests is the *conditio sine qua non* to be a proficient blockchain engineer.

Note that there are two types of invariant tests:

* stateless invariant tests: tests where the states of the test are independent of one other.;
    
* stateful invariant tests: tests where the state of the next run is affected by all the previous states;
    

We can in fact find the undesired behaviour of `endTheLottery` just by performing the following test:

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import {Test} from "lib/forge-std/src/Test.sol";
import {Lottery} from "src/Lottery.sol";
import {DeployLottery} from "script/DeployLottery.s.sol";
import {StdInvariant} from "lib/forge-std/src/StdInvariant.sol";

contract LotteryTest is Test {
    address player0 = makeAddr("Alice");
    address player1 = makeAddr("Bob");
    address player2 = makeAddr("Carl");
    address player3 = makeAddr("David");
    address player4 = makeAddr("Eleonor");
    address owner = makeAddr("Owner");
    uint256 public constant BALANCE = 100 ether;

    Lottery lottery;

    function setUp() public {
        DeployLottery deployer = new DeployLottery();
        vm.deal(player0, BALANCE);
        vm.deal(player1, BALANCE);
        vm.deal(player2, BALANCE);
        vm.deal(player3, BALANCE);
        vm.deal(player4, BALANCE);
        vm.deal(owner, BALANCE);
        lottery = deployer.run();
        lottery.transferOwnership(owner);
        targetContract(address(lottery));
    }

    function testOnlyOwnerCanEndTheLottery() public {
        vm.expectRevert(
            abi.encodeWithSelector(
                Lottery.Lottery__notTheOwner.selector,
                player0
            )
        );

        vm.startPrank(player0); // not the owner
        lottery.endTheLottery();
        vm.stopPrank();
    }

    function testAssertNumberOfPlayers() public {
        vm.startPrank(player0);
        lottery.joinLottery{value: 1 ether}();
        vm.stopPrank();
        vm.startPrank(player1);
        lottery.joinLottery{value: 1 ether}();
        vm.stopPrank();
        vm.startPrank(player2);
        lottery.joinLottery{value: 1 ether}();
        vm.stopPrank();
        vm.startPrank(player3);
        lottery.joinLottery{value: 1 ether}();
        vm.stopPrank();
        vm.startPrank(player4);
        lottery.joinLottery{value: 1 ether}();
        vm.stopPrank();

        uint256 expectedNumberOfPlayers = 5;
        uint256 numberOfPlayers = lottery.getNumberOfPlayer();

        assertEq(expectedNumberOfPlayers, numberOfPlayers);
    }

    function testFuzz_WinnerIsAlwaysPlayers0(uint96 numPlayers) public {
        vm.startPrank(player0);
        for (uint256 i = 5; i < numPlayers; i++) {
            lottery.joinLottery{value: 1 ether}();
        }
        vm.stopPrank();

        vm.startPrank(owner);
        lottery.endTheLottery();
        vm.stopPrank();

        address expectedWinner = lottery.getPlayer(0);
        assertEq(
            lottery.getPlayer(0),
            expectedWinner
        );
    }
}
```

The test fails (as expected) and it returns the following logs to notify that there is at leas a situation in which `endTheLottery` has unexpected behavior:

```solidity
Test result: FAILED. 2 passed; 1 failed; 0 skipped; finished in 4.96ms

Failing tests:
Encountered 1 failing test in test/LotteryTest.t.sol:LotteryTest
[FAIL. Reason: Lottery__notEnoughtPlayers() Counterexample: calldata=0x515cecbc0000000000000000000000000000000000000000000000000000000000000000, args=[0]] testFuzz_WinnerIsAlwaysPlayers0(uint96) (runs: 0, μ: 0, ~: 0)

Encountered a total of 1 failing tests, 2 tests succeeded
```

In fact `0x515cecbc0000000000000000000000000000000000000000000000000000000000000000` is the hexadecimal representation of an even number. Adding an even number to 5 (the starting point of the for loop) results in an odd number, which triggers the second if statement in the `endLottery` function and thus the winner is `players[1]`.

## To go further

To learn more about testing Solidity contract with the Foundry framework and discover advanced testing techniques consult the [Foundry docs](https://book.getfoundry.sh/forge/tests).

---

---

And that's it for this article.

Thanks for reading.

For any question or suggestion related to what I covered in this article, please add it as a comment. For special needs, you can contact me [**here**](http://amm.zanotp.com/contact).