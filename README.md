# DCIE: ARC Prize 2026, ARC-AGI-3

This is my code submission for the ARC-AGI-3 track of ARC Prize 2026, and the versions that came after it.

| | |
|---|---|
| Kaggle notebook | [netphantom17/ssm-farseer-arc3-official-gateway-runner](https://www.kaggle.com/code/netphantom17/ssm-farseer-arc3-official-gateway-runner) |
| Submission | 57913002, official score 0.00 |
| Paper | [*Likely and Allowed*](https://ivybe1337.github.io/Survival-Mechanics/), DOI [10.5281/zenodo.24093939](https://doi.org/10.5281/zenodo.24093939) |

## What was submitted

The submitted notebook is the foundation of the agent: the scaffold, the two online models and the navigation layer, joined to the official `ARC-AGI-3-Agents` framework and played through Kaggle's gateway. Nothing in it is pretrained. Both models start from random weights and learn only from what they see during play. The official score of that run is 0.00.

## The agent

DCIE runs two small recursive mixture-of-experts networks in PyTorch, side by side, and trains both inside every action step.

**The causal model** (`CausalCore`) predicts what an action does: the next frame's features, whether the level progresses, and whether anything changes. It learns from every transition it witnesses.

**The admissibility model** (`SurvivalCore`) predicts what an action risks: a game over, or a move that changes nothing. It keys its evidence on the state fingerprint, the recent action history and the proposed action.

Every proposed action carries one of three statuses:

| Status | Meaning |
|---|---|
| A | witnessed: this action was taken here and the game went on |
| X | certified invalid by the official action contract |
| U | unknown |

Observed risk lowers an action's score. It never turns an action into X. Only the action contract can exclude.

Candidate actions come from FARSEER navigation and a world-state policy (`razael_chi/`), plus clicks on salient coordinates. Each candidate is scored by a short counterfactual rollout through both models: progress and change from the causal side, risk and stagnation from the admissibility side. The agent takes the best.

## Terms

| Code | Paper |
|---|---|
| Survival State Mechanics (SSM) | the joint state of prediction and admissibility |
| causal model, transition forecast | predictive state ε |
| survival model, survival morph evidence | evidence about the admissibility rule V |
| A / X / U | statuses A / X / U |

## Versions

```
notebook/        the submitted notebook and its metadata
dcie/            version 1, exactly as submitted (dcie_cognition.py, my_agent.py)
dcie_fast/       same scoring rule, batched: one forward pass per candidate set
dcie_v2/         version 1 plus fair scheduling: the official swarm runs every game
                 as a thread in one process, so v2 caps how many games compute at
                 once and starts each game's clock when it starts playing
razael_chi/      the navigation and world-state policy package the notebook embeds
agents_init.py   the agent registration the framework expects
```

## Running

The agent plugs into the ARC Prize Foundation's [`ARC-AGI-3-Agents`](https://github.com/arcprize/ARC-AGI-3-Agents) framework:

1. Copy `dcie/my_agent.py` to `agents/templates/my_agent.py`.
2. Replace `agents/__init__.py` with `agents_init.py`.
3. Put `dcie/dcie_cognition.py` and `razael_chi/` on the Python path.
4. Run `uv run main.py --agent=myagent`.

The last cell of the notebook does exactly this against Kaggle's gateway.

Environment knobs: `DCIE_MAX_ACTIONS` (default 500), `DCIE_GAME_SECONDS` (default 900), `DCIE_SEED` (default 17), and for v2 `DCIE_CONCURRENT_GAMES` (default 2).

## License

My code in this repository is released under [MIT-0](LICENSE). The `ARC-AGI-3-Agents` framework and the official ARC-AGI-3 wheels belong to the ARC Prize Foundation and are not included here.

Joshua Matthew Carter
