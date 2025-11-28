const rankOrder = ["ace","2","3","4","5","6","7","8","9","10","jack","queen","king"];
const rIndex = (rank) => rankOrder.indexOf(rank);

function isTirsi(group) {
  if (group.length < 3 || group.length > 4) return false;
  let jokers = group.filter(c => c.rank.includes("joker"));
  if (jokers.length > 1) return false;
  let real = group.filter(c => !c.rank.includes("joker"));
  let ranks = [...new Set(real.map(c => c.rank))];
  if (ranks.length !== 1) return false;
  let suits = real.map(c => c.suit);
  if ([...new Set(suits)].length !== suits.length) return false;
  return true;
}

function isFreeTirsi(group) {
  return isTirsi(group) && !group.some(c => c.rank.includes("joker"));
}

function isSuivi(group) {
  if (group.length < 3) return false;
  let jokers = group.filter(c => c.rank.includes("joker"));
  if (jokers.length > 1) return false;
  let real = group.filter(c => !c.rank.includes("joker"));
  let suits = [...new Set(real.map(c => c.suit))];
  if (suits.length !== 1) return false;
  let sorted = [...real].sort((a,b)=>rIndex(a.rank)-rIndex(b.rank));
  let gaps = 0;
  for (let i = 0; i < sorted.length - 1; i++) {
    let diff = rIndex(sorted[i+1].rank) - rIndex(sorted[i].rank);
    if (diff === 1) continue;
    if (diff > 2) return false;
    gaps++;
    if (gaps > jokers.length) return false;
  }
  // ✔ Only forbid wrap K-A-2
  const values = sorted.map(c => rIndex(c.rank));
  if (values.includes(0) && values.includes(12) && values.includes(1)) {
    return false;
  }
  return true;
}

function isFreeSuivi(group) {
  return isSuivi(group) && !group.some(c => c.rank.includes("joker"));
}

export function canDeclareWin(groups) {
  let total = groups.reduce((s,g)=>s+g.length,0);
  if (total !== 13)
    return { success: false, reason: ` You must use all 13 cards (you used ${total}).` };

  let invalidGroups = groups.filter(g => !isTirsi(g) && !isSuivi(g));
  if (invalidGroups.length > 0)
    return { success: false, reason: ` One or more groups are not valid Tirsi or Suivi.` };

  if (!groups.some(isFreeTirsi))
    return { success: false, reason: ` You must have at least 1 FREE Tirsi (no Joker).` };

  if (!groups.some(isFreeSuivi))
    return { success: false, reason: ` You must have at least 1 FREE Suivi (no Joker).` };

  return { success: true, reason: ` Valid Rami! You used all 13 cards with at least one Free Tirsi and one Free Suivi.` };
}

