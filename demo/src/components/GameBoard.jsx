import { useState, useEffect } from "react";
import PlayerHand from "./PlayerHand";
import PlayerMeldGroup from "./PlayerMeldGroup";
import OpponentHand from "./OpponentHand";
import Card from "./Card";
import { DragDropContext } from "@hello-pangea/dnd";
import { createDeck } from "../logic/deck";
import { aiTakeTurn } from "../logic/ai";
import { canDeclareWin } from "../logic/declareWin";

export default function GameBoard() {
  const [players, setPlayers] = useState([]);
  const [drawPile, setDrawPile] = useState([]);
  const [discard, setDiscard] = useState([]);
  const [current, setCurrent] = useState(0);
  const [mustDiscard, setMustDiscard] = useState(false);
  const [melds, setMelds] = useState([]);

  useEffect(() => {
    startGame();
  }, []);

  const startGame = () => {
    let deck = createDeck();
    const ps = [
      { name: "You", hand: deck.splice(0, 13), human: true },
      { name: "Bot A", hand: deck.splice(0, 13) },
      { name: "Bot B", hand: deck.splice(0, 13) },
    ];

    const top = deck.pop();
    setDiscard([top]);
    setPlayers(ps);
    setDrawPile(deck);
    setCurrent(0);
    setMustDiscard(false);
    setMelds([]);
  };

  const onDraw = (fromDiscard) => {
    if (mustDiscard) return;
    const ps = [...players];
    const me = ps[current];
    const taken = fromDiscard ? discard.pop() : drawPile.pop();
    me.hand.push(taken);
    setPlayers(ps);
    setMustDiscard(true);
  };

  const onDiscard = (card) => {
    if (players[current].name !== "You" || !mustDiscard) return;
    const ps = [...players];
    const me = ps[current];
    me.hand = me.hand.filter(c => c.id !== card.id);
    discard.push(card);
    setPlayers(ps);
    setDiscard([...discard]);
    nextTurn();
  };

  const nextTurn = () => {
    setMustDiscard(false);
    setCurrent((c) => (c + 1) % 3);
  };

  // AI turn
  useEffect(() => {
    const p = players[current];
    if (!p || p.human) return;
    const timeout = setTimeout(() => {
      const { drawPile: dp, discard: dc, player: ai } =
        aiTakeTurn(p, drawPile, discard);
      const ps = [...players];
      ps[current] = ai;
      setPlayers(ps);
      setDrawPile(dp);
      setDiscard([...dc]);
      nextTurn();
    }, 1200);
    return () => clearTimeout(timeout);
  }, [current]);

  // Drag & Drop handler
 const onDragEnd = (result) => {
  if (!result.destination) return;

  const source = result.source;
  const dest = result.destination;
  const ps = [...players];
  const me = ps[0];

  let newHand = [...me.hand];
  let newMelds = [...melds];

  // 🟦 CASE 1: Rearranging inside the hand
  if (source.droppableId === "player-hand" && dest.droppableId === "player-hand") {
    const [moved] = newHand.splice(source.index, 1);
    newHand.splice(dest.index, 0, moved);
    ps[0].hand = newHand;
    setPlayers(ps);
    return;
  }

  // 🟥 CASE 2: Moving between hand and melds OR between melds
  const getCard = () => {
    if (source.droppableId === "player-hand") {
      return newHand.splice(source.index, 1)[0];
    } else {
      const gIndex = parseInt(source.droppableId.split('-')[1]);
      return newMelds[gIndex].splice(source.index, 1)[0];
    }
  };

  const placeCard = (card) => {
    if (dest.droppableId === "player-hand") {
      newHand.splice(dest.index, 0, card);
    } else {
      const gIndex = parseInt(dest.droppableId.split('-')[1]);
      newMelds[gIndex].splice(dest.index, 0, card);
    }
  };

  const moved = getCard();
  placeCard(moved);

  ps[0].hand = newHand;
  setPlayers(ps);
  setMelds(newMelds);
};


  const me = players[0];

  return (
    <DragDropContext onDragEnd={onDragEnd}>
      <h2 style={{ color: "gold" }}> Rami</h2>

      {players.slice(1).map(p => (
        <OpponentHand key={p.name} name={p.name} count={p.hand.length} />
      ))}

      <h3>Draw</h3>
<div className="draw-section">
  <div className="deck-area">
    <Card faceDown={true} />
    <button 
      className="btn btn-deck"
      disabled={mustDiscard || current !== 0} 
      onClick={() => onDraw(false)}
    >
      Draw from Deck
    </button>
  </div>
  
  <div className="discard-area">
    <Card card={discard[discard.length - 1]} />
    <button 
      className="btn btn-discard"
      disabled={mustDiscard || current !== 0} 
      onClick={() => onDraw(true)}
    >
      Draw from Discard
    </button>
  </div>
</div>
      <h3 style={{ marginTop: 20 }}>Your Hand</h3>
      {me && (
        <PlayerHand
          hand={me.hand}
          setHand={(newHand) => {
            const ps = [...players];
            ps[0].hand = newHand;
            setPlayers(ps);
          }}
          onDiscard={onDiscard}
        />
      )}

      <h3 style={{ marginTop: 20, color: "gold" }}>Your Melds</h3>
      <button
        style={{
          padding: "6px 10px",
          background: "#222",
          color: "gold",
          border: "1px solid gold",
          borderRadius: 8,
          marginBottom: 10,
          cursor: "pointer"
        }}
        onClick={() => setMelds([[], ...melds])}
      >
        New Goup
      </button>

      {melds.map((group, gIndex) => (
        <PlayerMeldGroup
  key={gIndex}
  group={group}
  gIndex={gIndex}
  onRemove={() => {
    // Put cards back into hand when deleting meld
    const ps = [...players];
    const me = ps[0];

    me.hand = [...me.hand, ...group];

    // Remove meld
    const newMelds = [...melds];
    newMelds.splice(gIndex, 1);

    setPlayers(ps);
    setMelds(newMelds);
  }}
/>

      ))}

      {current === 0 && (
        <button
          style={{
            marginTop: 10,
            padding: "10px 18px",
            background: "gold",
            border: "2px solid #543",
            fontWeight: "bold",
            cursor: "pointer",
            borderRadius: 8
          }}
          onClick={() => {
            const result = canDeclareWin(melds);

if (result.success) {
  alert(result.reason);
  startGame();
} else {
  alert(result.reason);
}

          }}
        >
          Declare Win
        </button>
      )}
    </DragDropContext>
  );
}
