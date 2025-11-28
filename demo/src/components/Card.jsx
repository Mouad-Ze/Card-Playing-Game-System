export default function Card({ card, onClick, faceDown }) {
  if (!card) {
    return (
      <img
        src="/cards/back.png"
        alt="card-back"
        style={{ width: 72, opacity: 0.5 }}
      />
    );
  }

  let src;

  if (faceDown) src = "/cards/back.png";
  else if (card.rank === "black_joker" || card.rank === "red_joker")
    src = `/cards/${card.rank}.png`;
  else src = `/cards/${card.rank}_of_${card.suit}.png`;

  return (
    <img
      className={faceDown ? "" : "card"}
      src={src}
      alt={card.rank}
      onClick={onClick}
      style={{ width: 72, filter: faceDown ? "brightness(.7)" : "none" }}
    />
  );
}
