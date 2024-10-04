# Best Practices for Assigning Validator Hotkeys

In the Compute Subnet, validators play a critical role in ensuring the performance and security of the network. However, miners must assign executors carefully to the validators to maximize incentives. This guide explains the best strategy for assigning validator hotkeys based on stake distribution within the network.

## Why Validator Hotkey Assignment Matters

You will **not receive any rewards** if your executors are not assigned to validators that control a **majority of the stake** in the network. Therefore, it’s crucial to understand how stake distribution works and how to assign your executors effectively.

## Step-by-Step Strategy for Assigning Validator Hotkeys

### 1. Check the Validator Stakes

The first step is to determine how much stake each validator controls in the network. You can find the current stake distribution of all validators by visiting:

[**TaoMarketCap Subnet 51 Validators**](https://taomarketcap.com/subnets/51/validators)

This page lists each validator and their respective stake, which is essential for making decisions about hotkey assignments.

### 2. Assign Executors to Cover at Least 50% of the Stake

To begin, you need to ensure that your executors are covering **at least 50%** of the total network stake. This guarantees that your executors will be actively validated and you’ll receive rewards.

#### Example:

Suppose you have **100 executors** (GPUs) and the stake distribution of the validators is as follows:

| Validator | Stake (%) |
|-----------|-----------|
| Validator 1 | 50% |
| Validator 2 | 25% |
| Validator 3 | 15% |
| Validator 4 | 5% |
| Validator 5 | 1% |

- To cover 50% of the total stake, assign **enough executors** to cover **Validator 1** (50% stake).
- In this case, assign at least **one executor** to **Validator 1** because they control 50% of the network stake.

### 3. Stake-Weighted Assignment for Remaining Executors

Once you’ve ensured that you’re covering at least 50% of the network stake, the remaining executors should be assigned in a **stake-weighted** fashion to maximize rewards.

#### Continuing the Example:

You have **99 remaining executors** to assign to validators. Here's the distribution of executors you should follow based on the stake:

- **Validator 1 (50% stake)**: Assign **50% of executors** to Validator 1.
    - Assign 50 executors.
- **Validator 2 (25% stake)**: Assign **25% of executors** to Validator 2.
    - Assign 25 executors.
- **Validator 3 (15% stake)**: Assign **15% of executors** to Validator 3.
    - Assign 15 executors.
- **Validator 4 (5% stake)**: Assign **5% of executors** to Validator 4.
    - Assign 5 executors.
- **Validator 5 (1% stake)**: Assign **1% of executors** to Validator 5.
    - Assign 1 executor.

### 4. Adjust Based on Network Dynamics

The stake of validators can change over time. Make sure to periodically check the **validator stakes** on [TaoMarketCap](https://taomarketcap.com/subnets/51/validators) and **reassign your executors** as needed to maintain optimal rewards. If a validator’s stake increases significantly, you may want to adjust your assignments accordingly.

## Summary of the Best Strategy

- **Step 1**: Check the validator stakes on [TaoMarketCap](https://taomarketcap.com/subnets/51/validators).
- **Step 2**: Ensure your executors are covering at least **50% of the total network stake**.
- **Step 3**: Use a **stake-weighted** strategy to assign your remaining executors, matching the proportion of the stake each validator controls.
- **Step 4**: Periodically recheck the stake distribution and adjust assignments as needed.

By following this strategy, you’ll ensure that your executors are assigned to validators in the most efficient way possible, maximizing your chances of receiving rewards.

## Additional Resources

- [TaoMarketCap Subnet 51 Validators](https://taomarketcap.com/subnets/51/validators)
- [Compute Subnet Miner README](README.md)

