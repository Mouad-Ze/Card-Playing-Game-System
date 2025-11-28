export const aiTakeTurn = (player, drawPile, discard) => {
  const drawn = drawPile.pop();
  player.hand.push(drawn);
  const idx = Math.floor(Math.random() * player.hand.length);
  const discarded = player.hand.splice(idx, 1)[0];
  discard.push(discarded);
  return { drawPile, discard, player };
};
