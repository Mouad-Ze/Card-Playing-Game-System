# Distributed Rami Game

A distributed multiplayer card game implementation demonstrating distributed systems concepts including shared state, consensus, consistency, and fault tolerance.

## Prerequisites

- Python 3.7 or higher
- Network connectivity between nodes (for distributed setup)

## Project Structure

```
code/
├── game/
│   └── rami.py          # Game logic (cards, rules, validation)
├── network/
│   ├── node.py          # Distributed node implementation
│   └── protocol.py      # Message protocol definitions
├── run_node.py          # Main entry point for running a node
└── logs/                # Log files (created automatically)
```

## Running the System

### Option 1: Running Locally (For Testing/Development)

You can run all three nodes on the same machine for testing. Open **three separate terminal windows**:

**Terminal 1 (P1):**
```bash
python run_node.py P1 8001
```

**Terminal 2 (P2):**
```bash
python run_node.py P2 8002
```

**Terminal 3 (P3):**
```bash
python run_node.py P3 8003
```

**Important:** Start P1 first, as it initiates the game. Wait a few seconds between starting each node to allow connections to establish.

### Option 2: Running on Separate Machines/VMs (For Project Requirements)

For the actual project demonstration, each node must run on a separate machine or virtual machine with its own IP address.

#### Step 1: Update IP Addresses in `run_node.py`

Edit `run_node.py` and update the `all_nodes` dictionary with the actual IP addresses of all machines:

```python
all_nodes = {
    "P1": ("192.168.1.10", 8001),  # Replace with Machine 1's actual IP
    "P2": ("192.168.1.11", 8002),  # Replace with Machine 2's actual IP
    "P3": ("192.168.1.12", 8003),  # Replace with Machine 3's actual IP
}
```

**Important:** Update this on ALL three machines with the same IP addresses.

#### Step 2: Run on Each Machine

On **Machine 1** (IP: 192.168.1.10):
```bash
python run_node.py P1 8001 192.168.1.10
```

On **Machine 2** (IP: 192.168.1.11):
```bash
python run_node.py P2 8002 192.168.1.11
```

On **Machine 3** (IP: 192.168.1.12):
```bash
python run_node.py P3 8003 192.168.1.12
```

The third argument is the machine's own IP address. The node will automatically bind to `0.0.0.0` (all interfaces) when a non-localhost IP is provided.

## Game Rules & Flow

Once all nodes are running, the game will automatically start. P1 initiates the first game. The rules follow the traditional 13-card Rami variant used in this project and are organized below for easy reference.

### 1. Introduction

Rami is a 2–4 player card game. The goal is to be the first player to arrange all 13 cards in your hand into valid combinations featuring at least one Tirsi (set) and one Suivi (run) that do not require Jokers.

### 2. The Deck

- A full 108-card deck is used: two standard 52-card decks plus four Jokers.
- Ranks: Ace, 2, 3, 4, 5, 6, 7, 8, 9, 10, Jack, Queen, King.
- The Ace can be low (1, before a 2) or high (14, after a King) but cannot wrap around from King to 2 in a single combination.

### 3. Objective

Form all 13 cards into valid combinations such that:

- There is at least one Tirsi (same-rank set of 3–4 cards, all different suits).
- There is at least one Suivi (3+ consecutive cards of the same suit).
- Both of the required Tirsi and Suivi must be “Free” (contain no Jokers).

### 4. Setup

1. Randomly choose a dealer (the system automates this via consensus when nodes start).
2. The dealer shuffles the 108-card deck.
3. The dealer deals 13 cards to each player, starting with themselves and continuing counter‑clockwise.
4. The remaining cards form the face-down draw pile. A visible discard pile (LIFO stack) is formed during play.

### 5. Gameplay (Turn Structure)

Play proceeds counter-clockwise, enforced in software via the token ring. On a turn the current player must:

1. **Draw**: Take the top card from either the draw pile or the discard pile. The CLI shows the top of both piles before you choose when you type `draw`.
2. **Discard**: Place one card from your hand face-up on the discard pile (which always exposes its top card to other players).

### 6. Valid Combinations

- **Tirsi (Set)**: 3 or 4 cards of identical rank, all different suits, with at most one Joker.
  - Example: `7H, 7D, 7S`
- **Suivi (Run)**: 3+ consecutive cards of the same suit, with at most one Joker.
  - Valid low-Ace example: `AH, 2H, 3H`
  - Valid high-Ace example: `QD, KD, AD`
  - Invalid wrap example: `KS, AS, 2S`
- A card cannot belong to multiple combos simultaneously.

### 7. Jokers

- Jokers can substitute for any missing card within a Tirsi or Suivi, but only one Joker is allowed per combination.
- Examples:
  - Tirsi: `JOKER, 10C, 10D`
  - Suivi: `8S, JOKER, 10S` (Joker represents 9S)

### 8. Declaring “I Won”

To win:

1. Draw as normal (from deck or discard).
2. Before discarding, declare `win` (CLI command described below) and provide the arrangement.
3. Lay down all 13 cards in valid combinations (the system validates them).
4. After validation, discard the final card automatically to reach zero cards.

Requirements recap:

- Exactly 13 cards arranged.
- At least one Free Tirsi and one Free Suivi.
- Remaining combos can include Jokers as needed (max one per combo).

### Available Commands (CLI)

- `draw` – Show both pile tops, then choose `deck`/`discard` or cancel before any card is drawn.
- `piles` – Peek at the pile tops without triggering a draw.
- `discard <CARD>` – Discard a card (e.g., `discard 7H`, `discard AD`, `discard JOKER`).
- `win <GROUPS>` – Declare win by specifying groups separated by `|` and comma-separated cards inside each group.
  - Example: `win 7H,7D,7S | AH,2H,3H | 10C,10D,JOKER`
- `hand` – Show your current hand.
- `state` – Show the replicated game state summary (current player, phase, pile tops, etc.).
- `token` – Display who currently holds the action token.
- `quit` – Exit the node process.

### Card Format

- Ranks: `A, 2-10, J, Q, K`
- Suits: `H (Hearts), D (Diamonds), C (Clubs), S (Spades)`
- Joker literal: `JOKER`

Examples: `7H`, `AD`, `JOKER`

## Logging

All important events are logged to files in the `logs/` directory:
- `logs/node_P1.log`
- `logs/node_P2.log`
- `logs/node_P3.log`

Logs include:
- Node startup/shutdown
- Connection events
- Game actions (draw, discard, win)
- Consensus decisions
- Token transfers
- Node failures

## Troubleshooting

### "Failed to connect" errors
- Ensure all nodes are running
- Check that IP addresses are correct
- Verify firewall settings allow connections
- Try starting nodes in order (P1, then P2, then P3)

### "Cannot draw: no token"
- Wait for your turn (check with `token` command)
- The token rotates automatically after each action

### Connection drops
- Check network connectivity
- Verify nodes are still running
- Check log files for error messages

### Game state inconsistencies
- All nodes should have the same seed (default: 42)
- Check that all nodes received the same actions (check logs)
- Restart all nodes if state becomes corrupted

## Testing the System

### Test 1: Basic Gameplay
1. Start all three nodes
2. Each player draws and discards cards
3. Verify game state is consistent across nodes

### Test 2: Win Declaration
1. Arrange cards to form valid winning combinations
2. Declare win using the `win` command
3. Verify all nodes recognize the win

### Test 3: Fault Tolerance
1. Start all nodes
2. Kill one node (Ctrl+C)
3. Verify remaining nodes detect the failure
4. Verify token rotation skips the dead node

### Test 4: Scaling (4+ nodes)
To test with more nodes, modify `run_node.py`:
```python
all_player_ids=("P1", "P2", "P3", "P4")  # Add P4
all_nodes = {
    "P1": ("127.0.0.1", 8001),
    "P2": ("127.0.0.1", 8002),
    "P3": ("127.0.0.1", 8003),
    "P4": ("127.0.0.1", 8004),  # Add P4
}
```

## Architecture Notes

- **Consensus**: Two-phase commit protocol for all game actions
- **State Replication**: Each node maintains a local copy of game state
- **Token Ring**: Ensures only one player acts at a time
- **Heartbeat**: Detects node failures
- **Logging**: All events logged for debugging and demonstration

## Project Requirements Compliance

✅ 3 separate nodes with own IP addresses  
✅ Communication via message passing (TCP sockets)  
✅ Each node communicates with ≥2 other nodes  
✅ Shared distributed state  
✅ Data consistency and synchronization  
✅ Consensus protocol  
✅ Fault tolerance (heartbeat + dead node detection)  
✅ Logging of all important events  

### How the Prototype Meets the Course Specification

- **Shared Global State (a)**: Every node maintains an identical `RamiGame` replica (hands, piles, turn order). Messages propose deterministic actions so state evolves consistently everywhere.
- **Consistency & Synchronization (b)**: Actions are ordered via a two-phase commit protocol. A player can only act with the token, ensuring serialized turns and synchronized application of draws/discards/wins.
- **Consensus (c)**: Each move (draw, discard, declare win) is committed only after the quorum of live nodes votes YES. This demonstrates distributed consensus on state transitions without a permanent central server.
- **Fault Tolerance (d)**: Heartbeats detect failed nodes, automatically mark them dead, and rotate the token past them so the game can continue. Logs capture events for debugging after failures.
- **Scalability Discussion (e)**: Adding players requires moderate changes (e.g., expanding `all_player_ids`). The protocol already generalizes to more nodes; documentation notes how to extend to 4+ players and how network setup must evolve.
- **Node Requirements**: Each node is a standalone process (can run on separate VMs) with its own IP, TCP server, and outbound sockets to the others. All important events are logged under `logs/`.

