---
name: usaco
description: “Solve algorithmic programming problems by extracting constraints, selecting appropriate algorithms, implementing efficient solutions, and validating correctness against input-output examples.“
---

# Algorithmic Programming Solver

This skill enables an agent to solve **algorithmic programming
problems** that require designing efficient algorithms and implementing
correct code.

These problems typically include:

-   structured problem statements
-   strict input/output formats
-   constraints on input size
-   hidden test cases used for evaluation

The skill guides the agent through a **structured reasoning workflow**
to ensure both correctness and efficiency.

------------------------------------------------------------------------

# When to Use This Skill

Use this skill when tasks involve:

-   algorithm design
-   competitive programming problems
-   structured input/output coding tasks
-   problems requiring efficient data structures or algorithms

Typical indicators include:

-   constraints on N (e.g., N ≤ 10⁵)
-   example input/output pairs
-   tasks requiring optimized algorithms

------------------------------------------------------------------------

# Expected Input

Tasks typically contain:

  Component             Meaning
  --------------------- ----------------------------
  Problem description   Defines the task
  Input format          Structure of provided data
  Output format         Expected output
  Constraints           Limits on input size
  Example               Sample input/output

Example structure:

Input: N a1 a2 ... aN

Output: single integer representing the answer

------------------------------------------------------------------------

# Core Principle

Algorithmic problems must be solved with **correct and efficient
algorithms**.

The agent must:

1.  analyze constraints
2.  determine feasible time complexity
3.  select appropriate algorithms
4.  implement correct code
5.  validate the solution

Brute-force solutions should only be used when constraints allow.

------------------------------------------------------------------------

# Reasoning Workflow (Executable Checklist)

## Step 1 --- Parse the Problem

Extract key elements:

-   variables
-   input size
-   objective
-   constraints

Questions to answer:

-   What needs to be computed?
-   What are the input limits?
-   Are multiple test cases possible?

Example extracted information:

N ≤ 100000\
Need to compute maximum value

------------------------------------------------------------------------

## Step 2 --- Determine Complexity Requirements

Use constraints to estimate acceptable complexity.

  N        Typical feasible complexity
  -------- -----------------------------
  ≤ 10     brute force
  ≤ 1000   O(N²)
  ≤ 10⁵    O(N log N)
  ≤ 10⁶    O(N)

Avoid algorithms that exceed feasible limits.

------------------------------------------------------------------------

## Step 3 --- Identify Problem Category

Classify the problem to guide algorithm selection.

Common categories:

### Graph problems

Possible tools:

-   BFS
-   DFS
-   Dijkstra
-   Union-Find
-   Topological sort

### Dynamic Programming

Indicators:

-   optimal substructure
-   sequential decisions
-   overlapping states

### Greedy algorithms

Indicators:

-   locally optimal choices lead to global optimum
-   sorting-based strategies

### Range queries

Possible structures:

-   prefix sums
-   segment trees
-   binary indexed trees

------------------------------------------------------------------------

## Step 4 --- Design the Algorithm

Before coding, define:

-   data structures
-   algorithm steps
-   edge cases

Example outline:

Sort elements\
Use two pointers\
Maintain running sum

Confirm that time complexity fits the constraints.

------------------------------------------------------------------------

## Step 5 --- Implement the Solution

Write clear and deterministic code.

Guidelines:

-   follow input/output format strictly
-   avoid unnecessary overhead
-   ensure memory usage fits constraints

Preferred languages often include:

-   Python
-   C++

------------------------------------------------------------------------

## Step 6 --- Validate the Solution

Test the implementation using provided examples.

Checklist:

-   example input produces expected output
-   edge cases handled
-   no infinite loops
-   memory and time limits respected

If errors occur:

-   check off-by-one errors
-   verify algorithm assumptions
-   reconsider complexity

------------------------------------------------------------------------

# Common Pitfalls

### Ignoring constraints

Selecting an algorithm that is too slow.

### Input parsing errors

Incorrectly handling the input format.

### Edge cases

Common cases include:

-   smallest possible input
-   largest constraint values
-   duplicate or special values

------------------------------------------------------------------------

# Output Expectations

A correct solution should:

1.  correctly interpret the problem
2.  choose an algorithm consistent with constraints
3.  produce valid executable code
4.  generate correct outputs for test cases
