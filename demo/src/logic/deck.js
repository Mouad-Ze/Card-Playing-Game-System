const suits = ["clubs", "diamonds", "hearts", "spades"];
const ranks = ["ace","2","3","4","5","6","7","8","9","10","jack","queen","king"];

export const createDeck = () => {
  let deck = [];
  for (let d = 0; d < 2; d++) {
    for (let s of suits) {
      for (let r of ranks) {
        deck.push({
          id: `${r}_of_${s}-${d}`,
          rank: r,
          suit: s,
        });
      }
    }
  }
  for (let j = 0; j < 4; j++) {
    deck.push({
      id: `joker-${j}`,
      rank: j % 2 === 0 ? "black_joker" : "red_joker",
      suit: null,
    });
  }
  return shuffle(deck);
};

export const shuffle = arr => [...arr].sort(() => Math.random() - 0.5);
