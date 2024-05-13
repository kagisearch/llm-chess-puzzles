# How well can LLMs solve chess puzzles?

The goal of this project is to establish the relative reasoning capabilities of different large language models through a unique and hopefully unbiased benchmark.

Chess puzzles are a very challenging problem for most humans, let alone an LLM given a textual description of the entire board in just a few characters.

While we initially thought this would lead to interesting conclusions, further refinement may be necessary to ensure clarity and effectiveness of the benchmark.

## Methodology

Each LLM is given 1000 chess puzzles in FEN notation and its job is to predict the best next move.

Here is an example prompt:


**You are a very strong chess engine. The chess board is in the following state (FEN): 'q3k1nr/1pp1nQpp/3p4/1P2p3/4P3/B1PP1b2/B5PP/5K2 b k - 0 17'. 
What is the best move for white?**

## Results

Each LLM is given the same 1000 chess puzzles to solve. See `puzzles.csv`. Benchmarked on Mar 25, 2024. Last Update: May 13, 2024 (added gpt-4o)


| Model                    | Solved | Solved % | Illegal Moves | Illegal Moves % | Adjusted Elo |
|--------------------------|--------|----------|---------------|-----------------|--------------|
| gpt-4o                   | 501    | 50.1%    | 127           | 12.7%           | 1790         | 
| gpt-4-turbo-preview      | 229    | 22.9%    | 163           | 16.3%           | 1144         |
| gpt-4                    | 195    | 19.5%    | 183           | 18.3%           | 1047         |
| claude-3-opus-20240229   | 72     | 7.2%     | 464           | 46.4%           | 521          |
| claude-3-haiku-20240307  | 38     | 3.8%     | 590           | 59.0%           | 363          |
| claude-3-sonnet-20240229 | 23     | 2.3%     | 663           | 66.3%           | 286          |
| gpt-3.5-turbo            | 23     | 2.3%     | 683           | 68.3%           | 269          |
| claude-instant-1.2       | 10     | 1.0%     | 707           | 70.7%           | 245          |
| mistral-large-latest     | 4      | 0.4%     | 813           | 81.3%           | 149          |
| mixtral-8x7b             | 9      | 0.9%     | 832           | 83.2%           | 136          |
| gemini-1.5-pro-latest*   | FAIL   | -        | -             | -               | -            |

* gemini-1.5-pro-latest failed to comply with instructions to output just the move, no matter what how we changed the prompt

The best LLM was able to predict the best move in the position 229 out of 1000 times. It is quite remarkable that a language model is able to not just internalize the correct board state based on the notation, but also to find the best move.

The count of illegal moves made is included, as it represents a complete failure of the model to internalize the board state and rules of the game. 

The adjusted Elo is an attempt to calculate the equivalent 'chess Elo' of an LLM, adjusted for illegal move attempts. Take it with a grain of salt, it is mostly for comparative purposes between LLMs.

## Commentary

We can see very large differences in the models' ability to solve these problems. Some of the new models, like Claude and Mistral Large, lag behind the GPT-4 family of models significantly more than in mainstream reasoning benchmarks.

It is hard not to be impressed by the performance of the best model. However, we wanted to verify whether the model is actually capable of reasoning by building a simulation for a much simpler game - Connect 4 (see 'llmc4.py').

When asked to play Connect 4, all LLMs fail to do so, even at most basic level. This should not be the case, as the rules of the game are simpler and widely available.

The only conclusion is that this failure is due to the lack of historical records of played games in the training data.

This implies that it cannot be argued that these models are able to 'reason' in any sense of the word, but merely output a variation of what they have seen during training.

Also it means that the utility of this benchmark to compare models in the future will be influenced by this type of data finding its way into the model training data, and that even chess puzzles can become a dedicated part of it. 

## Usage

Start the benchmark:
```
pip install chess pyllms
python llmchess.py
```

This will assume you have OpenAI, Mistral and Anthropic API keys set in the environment, per [pyllms](https://github.com/kagisearch/pyllms) instructions.

The output will contain the index and ELO of the puzzle that is solved together with current score information:
```
Puzzle 505 (1556) solved. Try it: https://lichess.org/odoNOk41/black#49 Score: -351 Elo: 818 adjusted:243
Puzzle 708 (1561) solved. Try it: https://lichess.org/xC286h6k/black#33 Score: -493 Elo: 813 adjusted:241
```

Repository contains 1000 chess puzzles used to reproduce the above results. Puzzles are sourced at random from the [Lichess puzzle database](https://database.lichess.org/#puzzles). There is code available to create a different sample.

Repository also contains code to allow two LLM to play chess against each other (with output to stdout).

Additionaly, `llmc4.py` will have two models play Connect 4.

```
python llmc4.py
```

