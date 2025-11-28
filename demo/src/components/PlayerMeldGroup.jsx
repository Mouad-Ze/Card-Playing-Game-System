import { Droppable, Draggable } from "@hello-pangea/dnd";
import Card from "./Card";

export default function PlayerMeldGroup({ group, gIndex, onRemove }) {
  return (
    <div style={{
      marginBottom: 10,
      padding: 10,
      border: "2px dashed gold",
      borderRadius: 10,
      background: "rgba(255,255,255,0.08)",
      position: "relative"
    }}>
      <h4 style={{ color: "white", margin: 0 }}>Meld {gIndex + 1}</h4>

      {/* ❌ Close / Delete Button */}
      <button
        onClick={onRemove}
        style={{
          position: "absolute",
          right: 8,
          top: 8,
          background: "#c00",
          color: "white",
          border: "none",
          borderRadius: "50%",
          width: 22,
          height: 22,
          cursor: "pointer",
          fontWeight: "bold",
          lineHeight: "18px"
        }}
      >
        ✕
      </button>

      <Droppable droppableId={`meld-${gIndex}`} direction="horizontal">
        {provided => (
          <div
            ref={provided.innerRef}
            style={{ display: "flex", gap: 8, minHeight: 80, marginTop: 10 }}
            {...provided.droppableProps}
          >
            {group.map((card, index) => (
              <Draggable draggableId={card.id} index={index} key={card.id}>
                {provided2 => (
                  <div
                    ref={provided2.innerRef}
                    {...provided2.draggableProps}
                    {...provided2.dragHandleProps}
                  >
                    <Card card={card} />
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
