import Card from "./Card";

export default function OpponentHand({ name, count }) {
  return (
    <div style={{ padding: 10, border: "1px solid gray", marginBottom: 4 }}>
      {name}: {count} cards
      <div style={{ display: "flex", gap: 4, marginTop: 4 }}>
        {Array.from({ length: count }).map((_, i) => (
          <Card key={i} faceDown card={{}} />
        ))}
      </div>
    </div>
  );
}
