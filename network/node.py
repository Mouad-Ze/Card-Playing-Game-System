# net/node.py

import socket
import threading
import time
import random
import logging
import os
from typing import Dict, List, Tuple

from game.rami import RamiGame
from .protocol import (
    encode_msg, decode_msg,
    card_to_str, str_to_card,
    HELLO, STATE_SNAPSHOT, HEARTBEAT,
    TOKEN_ANNOUNCE, PLAYER_QUIT,
    ACTION_PROPOSE, ACTION_VOTE, ACTION_COMMIT, ACTION_ABORT,
    DEALER_SELECTED, NEW_GAME,
    WIN_DECISION,
)


class RamiNode:
    def __init__(
        self,
        player_id: str,
        host: str,
        port: int,
        peers: List[Tuple[str, int]],
        all_player_ids=("P1", "P2", "P3"),
        seed=42,
    ):
        self.player_id = player_id
        self.host = host
        self.port = port
        self.peers_info = peers

        # Player list (static)
        self.all_player_ids = list(all_player_ids)
        self.n_players = len(self.all_player_ids)

        # Distributed state
        self.dealer = None
        self.turn_order = list(self.all_player_ids)
        self.token_holder = None

        # Local replicated game engine
        self.game = None
        self.seed = seed

        # Networking
        self.server_socket = None
        self.connections = []
        self.connections_lock = threading.Lock()

        # Consensus tracking
        self.current_action_id = 0
        self.pending_votes: Dict[int, Dict[str, bool]] = {}
        self.actions_by_id: Dict[int, dict] = {}
        # Use composite key (action_id, player_id) to track applied actions
        # This prevents collisions when different players use the same action ID
        self.applied_actions = set()  # Set of (action_id, player_id) tuples

        # Heartbeat tracking
        now = time.time()
        self.last_heartbeat = {pid: now for pid in self.all_player_ids}
        self.alive_players = set(self.all_player_ids)
        self.heartbeat_interval = 2
        self.heartbeat_timeout = 15  # Increased to 15 seconds to be more tolerant of network delays

        self.running = True

        # Setup logging
        self._setup_logging()


    def _setup_logging(self):
        """Setup file-based logging for this node."""
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        logger = logging.getLogger(f'Node_{self.player_id}')
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        logger.handlers = []
        
        # File handler
        file_handler = logging.FileHandler(
            f'{log_dir}/node_{self.player_id}.log',
            mode='a'
        )
        file_handler.setLevel(logging.INFO)
        
        # Console handler (also print to stdout)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        self.logger = logger
        self.logger.info(f"Node {self.player_id} logging initialized")


    def start(self):
        threading.Thread(target=self._run_server, daemon=True).start()
        time.sleep(0.5)
        self._connect_to_peers()

        # Announce presence
        self._broadcast({"type": HELLO, "sender": self.player_id, "payload": {}})

        # P1 initiates the FIRST game
        if self.player_id == "P1":
            self._start_new_game()

        threading.Thread(target=self._send_heartbeats_loop, daemon=True).start()
        threading.Thread(target=self._check_liveness_loop, daemon=True).start()

        self.logger.info(f"Node started at {self.host}:{self.port}")
        print(f"[{self.player_id}] Node started at {self.host}:{self.port}")


    def _start_new_game(self):
        self.logger.info("Starting NEW GAME")
        print(f"[{self.player_id}] Starting NEW GAME.")

        dealer = random.choice(self.all_player_ids)
        # Randomize seed for each new game
        new_seed = random.randint(1, 1000000)
        self.logger.info(f"Randomly selected dealer = {dealer}, seed = {new_seed}")
        print(f"[{self.player_id}] Randomly selected dealer = {dealer}, seed = {new_seed}")

        self._broadcast({
            "type": DEALER_SELECTED,
            "sender": self.player_id,
            "payload": {"dealer": dealer, "seed": new_seed}
        })

        # Apply locally too
        self._apply_dealer(dealer, new_seed)

        # Create the game using new turn order
        self._reset_game_state()
        
        # P1 (who initiates the game) should always broadcast the token holder
        # to ensure all nodes receive it, regardless of who the dealer is
        if self.token_holder:
            self._broadcast_token_holder()
            self.logger.info(f"Broadcasted initial token holder: {self.token_holder}")

    def _apply_dealer(self, dealer, seed=None):
        """Apply the dealer to THIS node and derive turn order."""
        self.dealer = dealer
        if seed is not None:
            self.seed = seed
        idx = self.all_player_ids.index(dealer)

        # Turn order: dealer → next → next …
        self.turn_order = [
            self.all_player_ids[(idx + i) % self.n_players]
            for i in range(self.n_players)
        ]

        # Dealer holds the initial token (first in turn order)
        self.token_holder = self.turn_order[0]

        self.logger.info(f"New dealer = {dealer}, turn order = {self.turn_order}, seed = {self.seed}, token_holder = {self.token_holder}")
        print(f"[{self.player_id}] New dealer = {dealer}")
        print(f"[{self.player_id}] New turn order = {self.turn_order}")
        print(f"[{self.player_id}] Initial token holder = {self.token_holder}")

        # Always broadcast token holder when dealer is applied
        # The player who initiated the game (P1) should broadcast
        # But also, if this node is the dealer, it should broadcast too
        # To ensure all nodes get the token announcement, we'll broadcast from whoever receives DEALER_SELECTED
        # Actually, let's have the dealer always broadcast the token
        if self.player_id == dealer:
            self._broadcast_token_holder()
        # Sync game's current player after reset
        self._sync_game_current_player()

    def _reset_game_state(self):
        """Start a fresh RamiGame replica using the new turn order."""
        self.logger.info("Resetting local game state")
        print(f"[{self.player_id}] Resetting local game state.")
        self.game = RamiGame(self.turn_order, seed=self.seed)
        # Sync game's current player with token holder after reset
        self._sync_game_current_player()


    def try_draw(self, source="deck"):
        if not self._i_have_token():
            self.logger.warning("Cannot draw: no token")
            print(f"[{self.player_id}] Cannot draw: no token.")
            return
        
        # Ensure game state is synced before proposing draw
        if self.game:
            if self.game.current_player_id != self.player_id:
                self.logger.warning(f"Game current player ({self.game.current_player_id}) doesn't match token holder ({self.player_id}), syncing...")
                self._sync_game_current_player(force=True)
            if self.game.phase != "AWAIT_DRAW":
                self.logger.warning(f"Game phase is {self.game.phase}, expected AWAIT_DRAW. Current player: {self.game.current_player_id}, Token holder: {self.token_holder}")
                # Try to sync
                self._sync_game_current_player(force=True)
        
        aid = self._next_action_id()
        action = {
            "action_id": aid,
            "kind": "DRAW",
            "player": self.player_id,
            "source": source,
        }
        self.logger.info(f"Proposing DRAW action {aid} from {source} (game current: {self.game.current_player_id if self.game else 'None'}, phase: {self.game.phase if self.game else 'None'})")
        self._propose_action(action)

    def try_discard(self, card_str: str):
        if not self._i_have_token():
            self.logger.warning("Cannot discard: no token")
            print(f"[{self.player_id}] Cannot discard: no token.")
            return
        aid = self._next_action_id()
        action = {
            "action_id": aid,
            "kind": "DISCARD",
            "player": self.player_id,
            "card": card_str,
        }
        self.logger.info(f"Proposing DISCARD action {aid}: {card_str}")
        self._propose_action(action)

    def try_declare_win(self, groups: List[List[str]]):
        """
        Declare win with the given groups.
        groups: List of groups, where each group is a list of card strings.
        """
        if not self._i_have_token():
            self.logger.warning("Cannot declare win: no token")
            print(f"[{self.player_id}] Cannot declare win: no token.")
            return
        aid = self._next_action_id()
        action = {
            "action_id": aid,
            "kind": "DECLARE_WIN",
            "player": self.player_id,
            "groups": groups,
        }
        self.logger.info(f"Proposing DECLARE_WIN action {aid} with {len(groups)} groups")
        self._propose_action(action)


    def _run_server(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.logger.info(f"Listening on {self.host}:{self.port}")
        print(f"[{self.player_id}] Listening on {self.host}:{self.port}")

        while self.running:
            conn, addr = self.server_socket.accept()
            with self.connections_lock:
                self.connections.append(conn)
            threading.Thread(
                target=self._handle_connection,
                args=(conn,),
                daemon=True
            ).start()

    def _connect_to_peers(self):
        for h, p in self.peers_info:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((h, p))
                with self.connections_lock:
                    self.connections.append(sock)
                threading.Thread(
                    target=self._handle_connection,
                    args=(sock,),
                    daemon=True
                ).start()
                self.logger.info(f"Connected to peer {h}:{p}")
                print(f"[{self.player_id}] Connected to peer {h}:{p}")
            except Exception as e:
                self.logger.error(f"Failed to connect to {h}:{p}: {e}")
                print(f"[{self.player_id}] Failed to connect {h}:{p}")

    def _handle_connection(self, conn):
        buffer = b""
        while self.running:
            try:
                data = conn.recv(4096)
                if not data:
                    break
                buffer += data
                while b"\n" in buffer:
                    raw, buffer = buffer.split(b"\n", 1)
                    msg = decode_msg(raw)
                    self._handle_message(msg)
            except:
                break

        with self.connections_lock:
            if conn in self.connections:
                self.connections.remove(conn)
        conn.close()

    def _handle_message(self, msg):
        mtype = msg["type"]
        sender = msg["sender"]
        payload = msg["payload"]

        # Update heartbeat timestamp for ANY message from a peer
        # This ensures we know the node is alive if it's sending any messages
        if sender in self.all_player_ids and sender != self.player_id:
            self.last_heartbeat[sender] = time.time()
            # If the node was previously marked as dead, mark it as alive again
            if sender not in self.alive_players:
                self.logger.info(f"Node {sender} is alive again (received message)")
                self.alive_players.add(sender)

        if mtype == HEARTBEAT:
            # Heartbeat messages are handled above (updating last_heartbeat)
            return

        if mtype == DEALER_SELECTED:
            dealer = payload["dealer"]
            seed = payload.get("seed", random.randint(1, 1000000))
            self._apply_dealer(dealer, seed)
            self._reset_game_state()
            # If this node is the dealer, also broadcast token to ensure all nodes get it
            # (P1 already broadcast, but this provides redundancy)
            if self.player_id == dealer and self.token_holder:
                self._broadcast_token_holder()
            return

        if mtype == TOKEN_ANNOUNCE:
            new_token_holder = payload["token_holder"]
            old_token_holder = self.token_holder
            sender = msg.get("sender", "unknown")
            
            # Always accept token announcements - they represent the authoritative state
            # Log the change for debugging
            if new_token_holder != old_token_holder:
                self.logger.info(f"Token holder changed to: {new_token_holder} (was: {old_token_holder}, from: {sender})")
                print(f"[{self.player_id}] Token holder now: {new_token_holder} (from {sender})")
            else:
                self.logger.debug(f"Token holder unchanged: {new_token_holder} (from {sender})")
            
            # Always update token holder to match the announcement (authoritative source)
            self.token_holder = new_token_holder
            
            # Force sync game's current player with token holder
            # This ensures the game state matches the token holder
            # Important: sync after setting token_holder so sync can use the new value
            self._sync_game_current_player(force=True)
            return

        if mtype == ACTION_PROPOSE:
            self._on_action_propose(payload)
            return

        if mtype == ACTION_VOTE:
            self._on_action_vote(sender, payload)
            return

        if mtype == ACTION_COMMIT:
            self._on_action_commit(payload)
            return

        if mtype == ACTION_ABORT:
            self.logger.warning("Action aborted")
            print(f"[{self.player_id}] Action aborted")
            self._rotate_token()
            return

        if mtype == PLAYER_QUIT:
            quitter = payload["player"]
            self.logger.info(f"Player {quitter} announced quit.")
            print(f"[{self.player_id}] Player {quitter} quit the game.")
            self._mark_player_dead(quitter, reason="player quit")
            return

        if mtype == WIN_DECISION:
            winner = payload["winner"]
            self.logger.info(f"GAME OVER - Winner: {winner}")
            print(f"[{self.player_id}] Game Over. Winner = {winner}")

            # Start a NEW GAME after win
            if self.player_id == self.token_holder:
                self._start_new_game()

            return


    def _send_heartbeats_loop(self):
        while self.running:
            self._broadcast({
                "type": HEARTBEAT,
                "sender": self.player_id,
                "payload": {}
            })
            time.sleep(self.heartbeat_interval)

    def _check_liveness_loop(self):
        while self.running:
            now = time.time()
            for pid in self.all_player_ids:
                if pid == self.player_id:
                    continue
                time_since_last = now - self.last_heartbeat[pid]
                if time_since_last > self.heartbeat_timeout:
                    self.logger.warning(f"Node {pid} heartbeat timeout: {time_since_last:.1f}s since last message (timeout: {self.heartbeat_timeout}s)")
                    self._mark_player_dead(pid, reason="heartbeat timeout")
            time.sleep(self.heartbeat_interval)


    def _i_have_token(self):
        return self.token_holder == self.player_id

    def _rotate_token(self):
        """Rotate token to next player in turn order. Only call this when you have the token."""
        if not self.turn_order:
            return

        # Use game's turn order if available to ensure consistency
        if self.game and hasattr(self.game, 'turn_order') and self.game.turn_order:
            turn_order = self.game.turn_order
        else:
            turn_order = self.turn_order

        current = self.token_holder
        if current not in turn_order:
            self.logger.error(f"Current token holder {current} not in turn order {turn_order}")
            return

        idx = turn_order.index(current)
        # Find next alive player in turn order
        new_holder = None
        for _ in range(len(turn_order)):
            idx = (idx + 1) % len(turn_order)
            candidate = turn_order[idx]
            if candidate in self.alive_players:
                new_holder = candidate
                break

        if new_holder is None:
            # Fallback: use first alive player
            for candidate in turn_order:
                if candidate in self.alive_players:
                    new_holder = candidate
                    break

        if new_holder:
            self.token_holder = new_holder
            self._broadcast_token_holder()
            # Sync game's current player with token holder
            self._sync_game_current_player()

    def _broadcast_token_holder(self):
        self._broadcast({
            "type": TOKEN_ANNOUNCE,
            "sender": self.player_id,
            "payload": {"token_holder": self.token_holder}
        })
        self.logger.info(f"Token rotated to: {self.token_holder}")
        print(f"[{self.player_id}] Broadcast token holder = {self.token_holder}")

    def _sync_game_current_player(self, force=False):
        """Sync the game's current player index with the token holder.
        Only sync if the game's current player doesn't match the token holder.
        Don't override if game is in a specific phase (like AWAIT_DISCARD_OR_WIN).
        
        Args:
            force: If True, force sync even if current player matches (used when receiving TOKEN_ANNOUNCE)
        """
        if self.game is None or self.token_holder is None:
            return
        if self.token_holder not in self.game.turn_order:
            return
        
        # Find the index of the token holder in the game's turn order
        target_idx = self.game.turn_order.index(self.token_holder)
        current_idx = self.game.current_player_index
        
        # If game's current player already matches token holder and phase is correct, no need to sync
        if not force and current_idx == target_idx and self.game.phase == "AWAIT_DRAW":
            return
        
        # When forcing (e.g., receiving TOKEN_ANNOUNCE), sync if needed
        # This ensures the game state matches the token holder
        # BUT: Don't override AWAIT_DISCARD_OR_WIN phase - player is mid-turn
        if force:
            old_idx = self.game.current_player_index
            old_phase = self.game.phase
            # Only update if the index actually needs to change
            if old_idx != target_idx:
                self.game.current_player_index = target_idx
                # Only set phase to AWAIT_DRAW if we're not in a mid-turn phase
                # Don't override AWAIT_DISCARD_OR_WIN - player has drawn and needs to discard
                if old_phase not in ("AWAIT_DISCARD_OR_WIN", "WIN_DECLARED", "GAME_OVER"):
                    self.game.phase = "AWAIT_DRAW"
                    self.logger.info(f"Force synced game: current player {old_idx}->{target_idx} ({self.game.current_player_id}), phase {old_phase}->AWAIT_DRAW")
                else:
                    self.logger.info(f"Force synced game: current player {old_idx}->{target_idx} ({self.game.current_player_id}), phase unchanged ({old_phase})")
            elif old_phase != "AWAIT_DRAW" and old_phase not in ("AWAIT_DISCARD_OR_WIN", "WIN_DECLARED", "GAME_OVER"):
                # Index is correct but phase is wrong, just fix the phase (but not if mid-turn)
                self.game.phase = "AWAIT_DRAW"
                self.logger.info(f"Force synced game phase: {old_phase}->AWAIT_DRAW (current player already correct: {self.game.current_player_id})")
            # If both index and phase are correct, or we're in a mid-turn phase, no need to do anything
        elif self.game.phase == "AWAIT_DRAW":
            # Normal sync when already in AWAIT_DRAW phase
            old_idx = self.game.current_player_index
            self.game.current_player_index = target_idx
            self.logger.info(f"Synced game current player to {self.token_holder} (index {old_idx} -> {target_idx})")
        elif self.game.phase == "INIT":
            # During initialization, sync and set phase
            self.game.current_player_index = target_idx
            self.game.phase = "AWAIT_DRAW"
            self.logger.info(f"Synced game current player to {self.token_holder} (index {target_idx}) during INIT")
        else:
            # Game is in mid-turn phase, log but don't sync
            self.logger.warning(f"Cannot sync: game phase is {self.game.phase}, current player is {self.game.current_player_id}, token holder is {self.token_holder}")

    def _mark_player_dead(self, pid, reason=""):
        if pid not in self.alive_players:
            return
        self.alive_players.remove(pid)
        self.logger.warning(f"Node {pid} removed from alive set ({reason}).")
        print(f"[{self.player_id}] Node {pid} removed ({reason}).")
        self.last_heartbeat[pid] = 0
        if self.token_holder == pid:
            self._rotate_token()


    def _next_action_id(self):
        self.current_action_id += 1
        return self.current_action_id

    def _propose_action(self, action):
        aid = action["action_id"]
        self.actions_by_id[aid] = action
        self.pending_votes[aid] = {self.player_id: True}

        self._broadcast({
            "type": ACTION_PROPOSE,
            "sender": self.player_id,
            "payload": action
        })

    def _on_action_propose(self, action):
        aid = action["action_id"]
        self.actions_by_id[aid] = action
        # Before validating, ensure game state is synced if this is our action
        # This helps when we just received the token
        # BUT: Don't sync for DISCARD actions - player is in AWAIT_DISCARD_OR_WIN phase
        if action.get("player") == self.player_id and self.token_holder == self.player_id:
            action_kind = action.get("kind")
            if action_kind != "DISCARD" and self.game:
                if self.game.current_player_id != self.player_id or self.game.phase != "AWAIT_DRAW":
                    self.logger.info(f"Syncing game state before voting on own {action_kind} action")
                    self._sync_game_current_player(force=True)
        vote = self._validate_action(action)

        if aid not in self.pending_votes:
            self.pending_votes[aid] = {}
        self.pending_votes[aid][self.player_id] = vote

        self._broadcast({
            "type": ACTION_VOTE,
            "sender": self.player_id,
            "payload": {"action_id": aid, "vote": vote}
        })

    def _on_action_vote(self, sender, payload):
        aid = payload["action_id"]
        vote = payload["vote"]

        if aid not in self.pending_votes:
            self.pending_votes[aid] = {}
        self.pending_votes[aid][sender] = vote

        if not self._i_have_token():
            return

        # All votes received
        if set(self.pending_votes[aid].keys()) != self.alive_players:
            return

        yes = sum(1 for v in self.pending_votes[aid].values() if v)
        action = self.actions_by_id[aid]
        if yes >= 2:
            self.logger.info(f"COMMITTED action {aid}: {action['kind']} by {action['player']}")
            print(f"[{self.player_id}] COMMITTED action {aid}")
            self._broadcast({
                "type": ACTION_COMMIT,
                "sender": self.player_id,
                "payload": {"action": action}
            })
            self._apply_action(action)
            # Only rotate token after DISCARD actions
            # DRAW actions keep the token so player can discard
            # DECLARE_WIN ends the game, so no token rotation needed
            if self.game and self.game.phase != "GAME_OVER":
                if action['kind'] == 'DISCARD':
                    # After discard, the game's next_player() was called in discard_card()
                    # Set token holder to match the game's new current player
                    # This ensures sequential order based on game's turn order
                    new_token_holder = self.game.current_player_id
                    if new_token_holder and new_token_holder in self.alive_players:
                        # Always update and broadcast, even if it's the same
                        # This ensures all nodes have the same view
                        old_token_holder = self.token_holder
                        self.token_holder = new_token_holder
                        self._broadcast_token_holder()
                        if new_token_holder != old_token_holder:
                            self.logger.info(f"Token passed to next player: {old_token_holder} -> {new_token_holder} (after DISCARD by {action['player']}, game current_player_index={self.game.current_player_index})")
                        else:
                            self.logger.warning(f"Token holder unchanged after DISCARD: {new_token_holder} (this should not happen)")
                    else:
                        self.logger.error(f"Cannot set token to {new_token_holder} - not in alive players or invalid. Alive players: {self.alive_players}, game current_player_id: {self.game.current_player_id}")
                # For DRAW actions, don't rotate - player keeps token to discard
        else:
            self.logger.warning(f"ABORTED action {aid}: insufficient votes ({yes}/{len(self.alive_players)})")
            print(f"[{self.player_id}] ABORTED action {aid}")
            self._broadcast({
                "type": ACTION_ABORT,
                "sender": self.player_id,
                "payload": {"action_id": aid}
            })
            self._rotate_token()

    def _on_action_commit(self, payload):
        action = payload["action"]
        # Sync game state before applying action to ensure we have the correct state
        # This is especially important for DRAW actions from discard pile
        if action.get('kind') == 'DRAW' and self.token_holder:
            # Before applying a DRAW action, ensure our game state matches the token holder
            if self.game and self.game.current_player_id != self.token_holder:
                self.logger.info(f"[{self.player_id}] Syncing game state before DRAW: current={self.game.current_player_id}, token={self.token_holder}")
                self._sync_game_current_player(force=True)
        # Only apply the action, don't rotate token here
        # Token rotation should only happen in _on_action_vote by the token holder
        self._apply_action(action)
        # After applying a DISCARD action, sync game state if we received the token
        # This ensures the game's current player matches the token holder
        # Note: We sync regardless of whether we have the token, because the token might
        # have been passed to us and we need to ensure our game state is correct
        if action.get('kind') == 'DISCARD' and self.token_holder:
            # The discard action called next_player(), so the game's current player should
            # match the token holder. If not, sync it.
            if self.game and self.game.current_player_id != self.token_holder:
                self.logger.info(f"Syncing game state after DISCARD: current={self.game.current_player_id}, token={self.token_holder}")
                self._sync_game_current_player(force=True)


    def _validate_action(self, action):
        if self.game is None:
            return False

        g = self.game
        kind = action["kind"]
        player = action["player"]

        # Before validating, ensure game state is synced with token holder
        # This is critical when the token was just passed to this player
        # BUT: Don't sync phase for DISCARD actions - player is in AWAIT_DISCARD_OR_WIN phase
        if self.token_holder and kind != "DISCARD":
            needs_sync = False
            if self.token_holder == player and (g.current_player_id != player or g.phase != "AWAIT_DRAW"):
                needs_sync = True
            elif self.token_holder == self.player_id and player == self.player_id and (g.current_player_id != self.player_id or g.phase != "AWAIT_DRAW"):
                needs_sync = True
            
            if needs_sync:
                self.logger.info(f"Syncing game state before validation: current={g.current_player_id}, token={self.token_holder}, phase={g.phase}, action_player={player}")
                self._sync_game_current_player(force=True)
        elif self.token_holder and kind == "DISCARD":
            # For DISCARD actions, only sync current player index if needed, NOT the phase
            # The player is in AWAIT_DISCARD_OR_WIN phase and should stay there
            if g.current_player_id != player:
                self.logger.info(f"Syncing current player before DISCARD validation: current={g.current_player_id}, token={self.token_holder}, action_player={player}, phase={g.phase}")
                # Only sync the player index, not the phase
                if self.token_holder in g.turn_order:
                    target_idx = g.turn_order.index(self.token_holder)
                    if g.current_player_index != target_idx:
                        old_phase = g.phase  # Preserve phase
                        g.current_player_index = target_idx
                        # Ensure phase is preserved
                        if g.phase != old_phase:
                            g.phase = old_phase
                            self.logger.warning(f"Phase was changed during sync, restored to {old_phase}")
                        self.logger.info(f"Synced current player index to {target_idx} ({g.current_player_id}) for DISCARD validation, phase preserved: {g.phase}")
            # Log if phase is wrong for debugging
            if g.phase != "AWAIT_DISCARD_OR_WIN" and g.phase != "WIN_DECLARED":
                self.logger.warning(f"DISCARD validation: phase is {g.phase}, expected AWAIT_DISCARD_OR_WIN or WIN_DECLARED")

        if g.current_player_id != player:
            self.logger.warning(f"Validation failed: current player is {g.current_player_id}, action player is {player}")
            return False

        if kind == "DRAW":
            if g.phase != "AWAIT_DRAW":
                self.logger.warning(f"Validation failed: phase is {g.phase}, expected AWAIT_DRAW")
                return False
            src = action["source"]
            if src == "deck":
                if len(g.deck) == 0:
                    self.logger.warning(f"Validation failed: deck is empty")
                    return False
            elif src == "discard":
                # Check discard pile directly, not through game_state_summary
                # This is more reliable as it checks the actual state
                if not g.discard_pile or len(g.discard_pile) == 0:
                    self.logger.warning(f"Validation failed: discard pile is empty (size: {len(g.discard_pile)})")
                    return False
            return True

        if kind == "DISCARD":
            # Check phase - must be AWAIT_DISCARD_OR_WIN or WIN_DECLARED
            if g.phase not in ("AWAIT_DISCARD_OR_WIN", "WIN_DECLARED"):
                self.logger.warning(f"DISCARD validation failed: phase is {g.phase}, expected AWAIT_DISCARD_OR_WIN or WIN_DECLARED")
                return False
            card = str_to_card(action["card"])
            if card not in g.players[player].hand:
                self.logger.warning(f"DISCARD validation failed: card {card} not in {player}'s hand")
                return False
            return True

        if kind == "DECLARE_WIN":
            if g.phase != "AWAIT_DISCARD_OR_WIN":
                return False
            # Parse groups from action
            groups_str = action.get("groups", [])
            try:
                groups = [[str_to_card(cs) for cs in group] for group in groups_str]
                ok, _ = g.can_declare_win(player, groups)
                return ok
            except Exception as e:
                self.logger.error(f"Win validation error: {e}")
                print(f"[{self.player_id}] Win validation error: {e}")
                return False

        return False

    def _apply_action(self, action):
        if self.game is None:
            self.logger.warning(f"Cannot apply action {action.get('action_id')}: game is None")
            return

        g = self.game
        kind = action["kind"]
        action_id = action.get("action_id")
        player = action["player"]  # Extract player first, before using it in action_key

        if action_id is not None:
            # Use composite key (action_id, player_id) to check if action was already applied
            action_key = (action_id, player)
            if action_key in self.applied_actions:
                self.logger.info(f"[{self.player_id}] Action {action_id} by {player} already applied, skipping")
                return
            self.logger.info(f"[{self.player_id}] Applying action {action_id}: {kind} by {player}")

        if kind == "DRAW":
            src = action["source"]
            try:
                # Log hand size and discard pile state before draw
                hand_size_before = len(g.players[player].hand)
                discard_size_before = len(g.discard_pile) if src == "discard" else 0
                discard_top_before = str(g.discard_pile[-1]) if (src == "discard" and g.discard_pile) else None
                
                # Log game state before draw for debugging
                self.logger.info(f"[{self.player_id}] Before DRAW: player={player}, current_player={g.current_player_id}, phase={g.phase}, hand_size={hand_size_before}, discard_size={discard_size_before}, discard_top={discard_top_before}")
                print(f"[{self.player_id}] Before DRAW from {src}: hand_size={hand_size_before}, discard_size={discard_size_before}, discard_top={discard_top_before}")
                
                # Before calling draw_card, ensure the game state is correct
                # If the current player doesn't match, sync it
                if g.current_player_id != player:
                    self.logger.warning(f"[{self.player_id}] Current player mismatch before draw: game.current={g.current_player_id}, action.player={player}. Syncing...")
                    self._sync_game_current_player(force=True)
                    # After sync, check again
                    if g.current_player_id != player:
                        self.logger.error(f"[{self.player_id}] Still mismatched after sync: game.current={g.current_player_id}, action.player={player}")
                        raise RuntimeError(f"Cannot apply DRAW: current player mismatch after sync")
                
                # Ensure phase is correct
                if g.phase != "AWAIT_DRAW":
                    self.logger.warning(f"[{self.player_id}] Phase mismatch before draw: phase={g.phase}, expected AWAIT_DRAW. Syncing...")
                    self._sync_game_current_player(force=True)
                    # After sync, check again
                    if g.phase != "AWAIT_DRAW":
                        self.logger.error(f"[{self.player_id}] Still wrong phase after sync: phase={g.phase}")
                        raise RuntimeError(f"Cannot apply DRAW: phase mismatch after sync")
                
                # Check if discard pile is empty before attempting draw
                if src == "discard":
                    if not g.discard_pile or len(g.discard_pile) == 0:
                        self.logger.error(f"[{self.player_id}] Cannot draw from discard: discard pile is empty! Discard pile size: {len(g.discard_pile)}")
                        print(f"[{self.player_id}] ERROR: Cannot draw from discard - discard pile is empty!")
                        raise RuntimeError(f"Cannot apply DRAW: discard pile is empty on this node")
                    # Log what card we're about to draw
                    top_card = g.discard_pile[-1]
                    self.logger.info(f"[{self.player_id}] About to draw from discard pile: top card is {top_card}, pile size is {len(g.discard_pile)}")
                    print(f"[{self.player_id}] About to draw {top_card} from discard pile (size: {len(g.discard_pile)})")
                
                # Before calling draw_card, ensure the game state is correct
                # If the current player doesn't match, sync it
                if g.current_player_id != player:
                    self.logger.warning(f"[{self.player_id}] Current player mismatch before draw: game.current={g.current_player_id}, action.player={player}. Syncing...")
                    self._sync_game_current_player(force=True)
                    # After sync, check again
                    if g.current_player_id != player:
                        self.logger.error(f"[{self.player_id}] Still mismatched after sync: game.current={g.current_player_id}, action.player={player}")
                        raise RuntimeError(f"Cannot apply DRAW: current player mismatch after sync")
                
                # Ensure phase is correct
                if g.phase != "AWAIT_DRAW":
                    self.logger.warning(f"[{self.player_id}] Phase mismatch before draw: phase={g.phase}, expected AWAIT_DRAW. Syncing...")
                    self._sync_game_current_player(force=True)
                    # After sync, check again
                    if g.phase != "AWAIT_DRAW":
                        self.logger.error(f"[{self.player_id}] Still wrong phase after sync: phase={g.phase}")
                        raise RuntimeError(f"Cannot apply DRAW: phase mismatch after sync")
                
                # For discard draws, ensure discard pile is not empty
                if src == "discard":
                    if not g.discard_pile or len(g.discard_pile) == 0:
                        self.logger.error(f"[{self.player_id}] Discard pile is empty on this node! This is a sync issue - action was proposed, so discard pile should exist.")
                        raise RuntimeError(f"Cannot apply DRAW from discard: discard pile is empty on this node")
                
                # Call draw_card - this should pop from discard pile and add to hand
                self.logger.info(f"[{self.player_id}] Calling g.draw_card({player}, {src})")
                card = g.draw_card(player, src)
                self.logger.info(f"[{self.player_id}] draw_card returned: {card}")
                
                hand_size_after = len(g.players[player].hand)
                discard_size_after = len(g.discard_pile) if src == "discard" else 0
                
                self.logger.info(f"[{self.player_id}] Applied DRAW: {player} drew from {src} -> {card} (hand: {hand_size_before} -> {hand_size_after}, discard: {discard_size_before} -> {discard_size_after})")
                print(f"[{self.player_id}] Applied DRAW: {player} drew {card} from {src} (hand: {hand_size_before} -> {hand_size_after}, discard: {discard_size_before} -> {discard_size_after})")
                
                # For discard draws, verify the card was actually removed from discard pile
                if src == "discard":
                    if discard_size_after != discard_size_before - 1:
                        self.logger.error(f"[{self.player_id}] CRITICAL: Discard pile size mismatch! Before: {discard_size_before}, After: {discard_size_after}, Expected: {discard_size_before - 1}")
                        print(f"[{self.player_id}] CRITICAL ERROR: Discard pile size did not decrease! Before: {discard_size_before}, After: {discard_size_after}")
                    if card in g.discard_pile:
                        self.logger.error(f"[{self.player_id}] CRITICAL: Card {card} is still in discard pile after drawing!")
                        print(f"[{self.player_id}] CRITICAL ERROR: Card {card} is still in discard pile!")
                    else:
                        self.logger.info(f"[{self.player_id}] Verified: Card {card} was removed from discard pile")
                
                if hand_size_after != hand_size_before + 1:
                    self.logger.error(f"ERROR: Hand size did not increase correctly after draw! Expected {hand_size_before + 1}, got {hand_size_after}")
                    print(f"[{self.player_id}] ERROR: Hand size did not increase correctly after draw!")
                
                if src == "discard" and discard_size_after != discard_size_before - 1:
                    self.logger.error(f"ERROR: Discard pile size did not decrease correctly! Expected {discard_size_before - 1}, got {discard_size_after}")
                    print(f"[{self.player_id}] ERROR: Discard pile size did not decrease correctly!")
                
                # Verify card is in hand
                if card not in g.players[player].hand:
                    self.logger.error(f"ERROR: Card {card} not found in {player}'s hand after draw from {src}!")
                    print(f"[{self.player_id}] ERROR: Card {card} not found in {player}'s hand after draw!")
                    # Try to add it manually as a fallback
                    g.players[player].draw(card)
                    self.logger.warning(f"Manually added card {card} to {player}'s hand as fallback")
                    # Verify it's now in hand
                    if card in g.players[player].hand:
                        self.logger.info(f"Successfully added card {card} to hand via fallback")
                        print(f"[{self.player_id}] Successfully added card {card} to hand via fallback")
                else:
                    self.logger.info(f"[{self.player_id}] Verified: Card {card} is in {player}'s hand after draw from {src}")
                
                # Mark action as applied only after successful completion
                if action_id is not None:
                    action_key = (action_id, player)
                    self.applied_actions.add(action_key)
                    self.logger.info(f"[{self.player_id}] Successfully applied action {action_id} by {player}")
                    
            except Exception as e:
                # Don't mark as applied if it failed
                self.logger.error(f"[{self.player_id}] Failed to apply action {action_id}: {e}")
                self.logger.error(f"[{self.player_id}] DRAW apply error: {e}", exc_info=True)
                print(f"[{self.player_id}] DRAW apply error: {e}")
                # Log game state for debugging
                if self.game:
                    discard_info = f"discard_pile_size={len(self.game.discard_pile)}"
                    if self.game.discard_pile:
                        discard_info += f", discard_top={self.game.discard_pile[-1]}"
                    self.logger.error(f"[{self.player_id}] Game state: current_player={self.game.current_player_id}, phase={self.game.phase}, {discard_info}")
                    print(f"[{self.player_id}] Game state: current_player={self.game.current_player_id}, phase={self.game.phase}, {discard_info}")
                # If it's a discard draw and the error is about empty pile, check if it's a sync issue
                if src == "discard" and "empty" in str(e).lower():
                    self.logger.error(f"[{self.player_id}] Discard pile appears empty on this node. This might be a synchronization issue.")
                    print(f"[{self.player_id}] WARNING: Discard pile appears empty - possible sync issue between nodes")

        elif kind == "DISCARD":
            card = str_to_card(action["card"])
            try:
                # Log discard pile state before discard
                discard_size_before = len(g.discard_pile)
                discard_top_before = str(g.discard_pile[-1]) if g.discard_pile else None
                hand_size_before = len(g.players[player].hand)
                
                g.discard_card(player, card)
                
                # Log discard pile state after discard
                discard_size_after = len(g.discard_pile)
                discard_top_after = str(g.discard_pile[-1]) if g.discard_pile else None
                hand_size_after = len(g.players[player].hand)
                
                self.logger.info(f"Applied DISCARD: {player} discarded {card} (hand: {hand_size_before} -> {hand_size_after}, discard: {discard_size_before} -> {discard_size_after}, top: {discard_top_before} -> {discard_top_after})")
                print(f"[{self.player_id}] Applied DISCARD: {player} discarded {card} (discard pile: {discard_size_before} -> {discard_size_after} cards)")
                
                # Verify card is in discard pile
                if card not in g.discard_pile:
                    self.logger.error(f"ERROR: Card {card} not found in discard pile after discard!")
                    print(f"[{self.player_id}] ERROR: Card {card} not found in discard pile after discard!")
                else:
                    self.logger.info(f"Verified: Card {card} is in discard pile (position: {g.discard_pile.index(card) if card in g.discard_pile else 'not found'})")
                    
                # Verify discard pile size increased
                if discard_size_after != discard_size_before + 1:
                    self.logger.error(f"ERROR: Discard pile size did not increase correctly! Expected {discard_size_before + 1}, got {discard_size_after}")
                    print(f"[{self.player_id}] ERROR: Discard pile size did not increase correctly!")
                
                # Mark action as applied only after successful completion
                if action_id is not None:
                    action_key = (action_id, player)
                    self.applied_actions.add(action_key)
                    self.logger.info(f"[{self.player_id}] Successfully applied DISCARD action {action_id} by {player}")
                    
            except Exception as e:
                # Don't mark as applied if it failed
                if action_id is not None:
                    self.logger.error(f"[{self.player_id}] Failed to apply DISCARD action {action_id} by {player}: {e}")
                self.logger.error(f"DISCARD apply error: {e}", exc_info=True)
                print(f"[{self.player_id}] DISCARD apply error: {e}")
                # Log game state for debugging
                if self.game:
                    self.logger.error(f"Game state: current_player={self.game.current_player_id}, phase={self.game.phase}, discard_pile_size={len(self.game.discard_pile)}")
                    print(f"[{self.player_id}] Game state: current_player={self.game.current_player_id}, phase={self.game.phase}, discard_pile_size={len(self.game.discard_pile)}")

        elif kind == "DECLARE_WIN":
            groups_str = action.get("groups", [])
            try:
                groups = [[str_to_card(cs) for cs in group] for group in groups_str]
                success, msg, card_to_discard = g.declare_win(player, groups)
                if success:
                    self.logger.info(f"WIN DECLARED by {player}: {msg}")
                    print(f"[{self.player_id}] {player} declared WIN! {msg}")
                    # Automatically discard the final card
                    if card_to_discard:
                        g.discard_card(player, card_to_discard)
                        self.logger.info(f"{player} discarded final card: {card_to_discard}")
                        print(f"[{self.player_id}] {player} discarded final card: {card_to_discard}")
                    # Broadcast win decision
                    if self._i_have_token():
                        self._broadcast({
                            "type": WIN_DECISION,
                            "sender": self.player_id,
                            "payload": {"winner": player}
                        })
                else:
                    self.logger.warning(f"Win declaration failed: {msg}")
                    print(f"[{self.player_id}] Win declaration failed: {msg}")
            except Exception as e:
                self.logger.error(f"DECLARE_WIN apply error: {e}")
                print(f"[{self.player_id}] DECLARE_WIN apply error: {e}")


    def _broadcast(self, msg):
        enc = encode_msg(msg)
        with self.connections_lock:
            for c in list(self.connections):
                try:
                    c.sendall(enc)
                except:
                    self.connections.remove(c)

    def announce_quit(self):
        self.logger.info("Announcing quit.")
        print(f"[{self.player_id}] Announcing quit.")
        self._broadcast({
            "type": PLAYER_QUIT,
            "sender": self.player_id,
            "payload": {"player": self.player_id}
        })
        self._mark_player_dead(self.player_id, reason="self quit")
        self.shutdown()

    def shutdown(self):
        if not self.running:
            return
        self.running = False
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
            self.server_socket = None
        with self.connections_lock:
            for c in list(self.connections):
                try:
                    c.close()
                except:
                    pass
            self.connections.clear()
