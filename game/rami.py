# game/rami_game.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import random


SUITS = ["H", "D", "C", "S"]
RANKS = list(range(1, 14))  


@dataclass(frozen=True)
class Card:
    rank: Optional[int] = None  
    suit: Optional[str] = None  

    @property
    def is_joker(self) -> bool:
        return self.rank is None and self.suit is None

    def __str__(self) -> str:
        if self.is_joker:
            return "JOKER"
        rank_str = {
            1: "A", 11: "J", 12: "Q", 13: "K"
        }.get(self.rank, str(self.rank))
        return f"{rank_str}{self.suit}"

    def __repr__(self) -> str:
        return str(self)


class Deck:
    """
    2x standard 52-card decks + 4 jokers = 108 cards,
    which is typical for Rami variants.
    """
    def __init__(self, seed: Optional[int] = None):
        self.cards: List[Card] = []
        for _ in range(2):
            for s in SUITS:
                for r in RANKS:
                    self.cards.append(Card(rank=r, suit=s))
        for _ in range(4):
            self.cards.append(Card(rank=None, suit=None))

        if seed is not None:
            random.seed(seed)
        random.shuffle(self.cards)

    def draw(self) -> Optional[Card]:
        if not self.cards:
            return None
        return self.cards.pop()

    def peek_top(self) -> Optional[Card]:
        if not self.cards:
            return None
        return self.cards[-1]

    def __len__(self) -> int:
        return len(self.cards)




def count_jokers(cards: List[Card]) -> int:
    return sum(1 for c in cards if c.is_joker)


def non_jokers(cards: List[Card]) -> List[Card]:
    return [c for c in cards if not c.is_joker]


def is_tirsi(cards: List[Card]) -> bool:
    """
    Tirsi (set): 3 or 4 cards of same rank, all different suits.
    At most one joker. No pure-joker sets.
    """
    if not (3 <= len(cards) <= 4):
        return False
    n_jokers = count_jokers(cards)
    if n_jokers > 1:
        return False
    real = non_jokers(cards)
    if not real:
        return False
    ranks = {c.rank for c in real}
    if len(ranks) != 1:
        return False
    suits = [c.suit for c in real]
    if len(suits) != len(set(suits)):
        return False
    return True


def is_free_tirsi(cards: List[Card]) -> bool:
    return is_tirsi(cards) and count_jokers(cards) == 0


def _try_consecutive(ranks: List[int], allow_gap: bool) -> bool:
    if not ranks:
        return False
    gaps_used = 0
    for a, b in zip(ranks, ranks[1:]):
        diff = b - a
        if diff == 1:
            continue
        elif diff == 2 and allow_gap and gaps_used == 0:
            gaps_used += 1
        else:
            return False
    return True


def is_suivi(cards: List[Card]) -> bool:
    """
    Suivi (run): 3+ cards, same suit, consecutive ranks.
    At most one joker. Ace can be low (A,2,3) or high (Q,K,A),
    but no wrap (K,A,2).
    """
    if len(cards) < 3:
        return False
    n_jokers = count_jokers(cards)
    if n_jokers > 1:
        return False

    real = non_jokers(cards)
    if not real:
        return False

    suits = {c.suit for c in real}
    if len(suits) != 1:
        return False

    ranks = [c.rank for c in real]
    if len(ranks) != len(set(ranks)):
        return False
    ranks.sort()

    has_ace = 1 in ranks

    def check_variant(use_ace_high: bool) -> bool:
        transformed = []
        for r in ranks:
            transformed.append(14 if (r == 1 and use_ace_high) else r)
        transformed.sort()
        return _try_consecutive(transformed, allow_gap=(n_jokers == 1))

    if not has_ace:
        return check_variant(False)
    return check_variant(False) or check_variant(True)


def is_free_suivi(cards: List[Card]) -> bool:
    return is_suivi(cards) and count_jokers(cards) == 0


def validate_arrangement(
    hand: List[Card],
    groups: List[List[Card]],
) -> Tuple[bool, str]:
    """
    Validate a 13-card 'I won' arrangement:
    - Exactly 13 cards used (from a 14-card hand after drawing)
    - Each group is Tirsi or Suivi
    - At least one free Tirsi (no jokers)
    - At least one free Suivi (no jokers)
    """
    used = [c for g in groups for c in g]
    if len(used) != 13:
        return False, "Arrangement must use exactly 13 cards."

    # Check that all used cards are in the hand
    hand_copy = hand.copy()
    for c in used:
        if c not in hand_copy:
            return False, f"Card {c} is not in hand."
        hand_copy.remove(c)
    
    # After using 13 cards, exactly 1 card should remain (to be discarded)
    if len(hand_copy) != 1:
        return False, "Arrangement must use exactly 13 cards, leaving 1 to discard."

    has_free_t = False
    has_free_s = False

    for g in groups:
        if len(g) < 3:
            return False, f"Group {g} is too short."
        if is_tirsi(g):
            if count_jokers(g) == 0:
                has_free_t = True
        elif is_suivi(g):
            if count_jokers(g) == 0:
                has_free_s = True
        else:
            return False, f"Group {g} is neither valid Tirsi nor Suivi."

    if not has_free_t:
        return False, "Need at least one Tirsi without Joker."
    if not has_free_s:
        return False, "Need at least one Suivi without Joker."

    return True, "Valid winning arrangement."



@dataclass
class Player:
    player_id: str
    hand: List[Card] = field(default_factory=list)

    def draw(self, card: Card):
        self.hand.append(card)

    def remove_card(self, card: Card):
        self.hand.remove(card)


class RamiGame:
    """
    Pure game logic, no networking.
    """

    def __init__(self, player_ids: List[str], seed: Optional[int] = None):
        if not (2 <= len(player_ids) <= 4):
            raise ValueError("Rami supports 2–4 players.")

        self.deck = Deck(seed=seed)
        self.discard_pile: List[Card] = []
        self.players: Dict[str, Player] = {
            pid: Player(player_id=pid) for pid in player_ids
        }
        self.turn_order = player_ids.copy()
        self.current_player_index = 0
        self.phase = "INIT"  

        self._deal_initial_hands()


    @property
    def current_player_id(self) -> str:
        return self.turn_order[self.current_player_index]

    def next_player(self):
        self.current_player_index = (self.current_player_index + 1) % len(self.turn_order)
        self.phase = "AWAIT_DRAW"

    # ----- Setup -----

    def _deal_initial_hands(self):
        for _ in range(13):
            for pid in self.turn_order:
                c = self.deck.draw()
                if c is None:
                    raise RuntimeError("Deck ran out while dealing.")
                self.players[pid].draw(c)
        self.phase = "AWAIT_DRAW"

    # ----- Actions -----

    def draw_card(self, player_id: str, source: str) -> Card:
        if player_id != self.current_player_id:
            raise RuntimeError(f"Not this player's turn. Current: {self.current_player_id}, Requested: {player_id}")
        if self.phase != "AWAIT_DRAW":
            raise RuntimeError(f"Player has already drawn. Phase: {self.phase}")
        if source == "deck":
            if len(self.deck) == 0:
                raise RuntimeError("Deck is empty.")
            card = self.deck.draw()
        elif source == "discard":
            if not self.discard_pile:
                raise RuntimeError("Discard pile is empty.")
            # Pop from the end (top of discard pile)
            card = self.discard_pile.pop()
        else:
            raise ValueError("source must be 'deck' or 'discard'.")

        # Add card to player's hand
        self.players[player_id].draw(card)
        self.phase = "AWAIT_DISCARD_OR_WIN"
        return card

    def discard_card(self, player_id: str, card: Card):
        if player_id != self.current_player_id:
            raise RuntimeError("Not this player's turn.")
        if self.phase not in ("AWAIT_DISCARD_OR_WIN", "WIN_DECLARED"):
            raise RuntimeError("Must draw before discarding.")

        player = self.players[player_id]
        if card not in player.hand:
            raise RuntimeError("Player does not have this card.")

        player.remove_card(card)
        self.discard_pile.append(card)
        
        # After win declaration, don't advance to next player - game is over
        if self.phase != "WIN_DECLARED":
            self.next_player()
        else:
            self.phase = "GAME_OVER"

    # ----- Winning -----

    def can_declare_win(self, player_id: str, groups: List[List[Card]]) -> Tuple[bool, str]:
        if player_id != self.current_player_id:
            return False, "Not this player's turn."
        if self.phase not in ("AWAIT_DISCARD_OR_WIN",):
            return False, "Player must draw before declaring win."
        player = self.players[player_id]
        # After drawing, player has 14 cards. They arrange 13 and discard 1.
        if len(player.hand) != 14:
            return False, "Must have exactly 14 cards (after drawing) to declare win."
        return validate_arrangement(player.hand, groups)

    def declare_win(self, player_id: str, groups: List[List[Card]]) -> Tuple[bool, str, Card]:
        """
        Declare win and return the card that should be discarded.
        Returns: (success, message, card_to_discard)
        """
        ok, msg = self.can_declare_win(player_id, groups)
        if not ok:
            return False, msg, None
        
        # Find the card that wasn't used in the arrangement (the one to discard)
        # Use the same logic as validate_arrangement to handle duplicates correctly
        used = [c for g in groups for c in g]
        player = self.players[player_id]
        hand_copy = player.hand.copy()
        for c in used:
            if c in hand_copy:
                hand_copy.remove(c)
        
        if len(hand_copy) != 1:
            return False, "Expected exactly one card remaining to discard.", None
        
        self.phase = "WIN_DECLARED"
        return True, msg, hand_copy[0]

    # ----- Debug / summary -----

    def hand_as_strings(self, player_id: str) -> List[str]:
        return [str(c) for c in self.players[player_id].hand]

    def game_state_summary(self) -> Dict:
        deck_top_card = self.deck.peek_top()
        return {
            "current_player": self.current_player_id,
            "phase": self.phase,
            "deck_size": len(self.deck),
            "deck_top": str(deck_top_card) if deck_top_card else None,
            "discard_top": str(self.discard_pile[-1]) if self.discard_pile else None,
            "hands_sizes": {pid: len(p.hand) for pid, p in self.players.items()},
        }
