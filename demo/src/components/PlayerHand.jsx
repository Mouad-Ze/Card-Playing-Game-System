import Card from "./Card";
import { Droppable, Draggable } from "@hello-pangea/dnd";

export default function PlayerHand({ hand, setHand, onDiscard }) {
  return (
    <div className="hand-area">
      <Droppable droppableId="player-hand" direction="horizontal">
        {provided => (
          <div
            ref={provided.innerRef}
            style={{ display: "flex", gap: 8 }}
            {...provided.droppableProps}
          >
            {hand.map((card, index) => (
              <Draggable draggableId={card.id} index={index} key={card.id}>
                {provided2 => (
                  <div
                    ref={provided2.innerRef}
                    {...provided2.draggableProps}
                    {...provided2.dragHandleProps}
                  >
                    <Card card={card} onClick={() => onDiscard(card)} />
                  </div>
                )}
              </Draggable>
            ))}
            {provided.placeholder}
          </div>
        )}
      </Droppable>
    </div>
  );
}
