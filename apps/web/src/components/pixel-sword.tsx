// Pixel-art sword in the style of a classic adventure blade. B=blade, E=edge light, G=guard, H=hilt, J=gem.
const ROWS = [
  ".......E.......",
  "......EBB......",
  "......EBB......",
  "......EBB......",
  "......EBB......",
  "......EBB......",
  "......EBB......",
  "......EBB......",
  "......EBB......",
  "......EBB......",
  "......EBB......",
  "......EBB......",
  "......EBB......",
  "..G...EBB...G..",
  ".GGG.GGJGG.GGG.",
  "GGGGGGJJJGGGGGG",
  ".GG...GJG...GG.",
  "......HHH......",
  "......HGH......",
  "......HHH......",
  "......HGH......",
  ".......G.......",
];
const FILL: Record<string, string> = {
  B: "var(--steel)",
  E: "var(--steel-hi)",
  G: "var(--moss)",
  H: "var(--bark)",
  J: "var(--honey)",
};

export function PixelSword() {
  return (
    <div className="sword-float relative" aria-hidden>
      <svg
        width={90}
        height={132}
        viewBox="0 0 15 22"
        shapeRendering="crispEdges"
      >
        {ROWS.flatMap((r, y) =>
          [...r].map((c, x) =>
            c === "." ? null : (
              <rect
                key={`${x}-${y}`}
                x={x}
                y={y}
                width={1}
                height={1}
                fill={FILL[c]}
              />
            )
          )
        )}
        <rect
          className="sword-glint"
          x={6}
          y={0}
          width={3}
          height={2}
          fill="var(--steel-hi)"
          opacity={0.9}
        />
      </svg>
      <div className="sword-shadow bg-muted mx-auto mt-3 h-1.5 w-12" />
    </div>
  );
}
