// HO as a small pixel sprout-bot. It floats, blinks, sways its leaf and runs a
// scan beam down its body, so the discovery card reads as something alive and
// working. Drawn on the shared 16x16 grid; colour comes from the theme tokens.
import { useId } from "react";

const BODY = [
  "................",
  "................",
  "................",
  "...BBBBBBBBBB...",
  "..BBBBBBBBBBBB..",
  "..BBBBBBBBBBBB..",
  "..BBBBBBBBBBBB..",
  "..BBBBBBBBBBBB..",
  "..BBBBBBBBBBBB..",
  "..BBBBBBBBBBBB..",
  "..BBBBBBBBBBBB..",
  "...BBBBBBBBBB...",
  "....BBBBBBBB....",
  "....BB....BB....",
  "....BB....BB....",
  "................",
];

function cells(rows: string[], char: string): [number, number][] {
  const out: [number, number][] = [];
  for (const [y, row] of rows.entries()) {
    for (const [x, c] of [...row].entries()) {
      if (c === char) {
        out.push([x, y]);
      }
    }
  }
  return out;
}

export function PipelineMascot() {
  const clip = useId();
  const body = cells(BODY, "B");

  return (
    <div className="ho-bot relative shrink-0" aria-hidden>
      <svg
        viewBox="-1 -2 18 20"
        shapeRendering="crispEdges"
        className="h-16 w-16 sm:h-20 sm:w-20"
      >
        <defs>
          <clipPath id={clip}>
            <rect x={2} y={8} width={12} height={6} />
          </clipPath>
        </defs>

        {/* contact shadow */}
        <ellipse
          className="ho-bot-shadow fill-bark/40"
          cx={8}
          cy={15.4}
          rx={4.4}
          ry={0.7}
        />

        {/* body */}
        {body.map(([x, y]) => (
          <rect
            key={`${x}-${y}`}
            x={x}
            y={y}
            width={1}
            height={1}
            className="fill-bark"
          />
        ))}

        {/* leaf sprout */}
        <g className="ho-bot-leaf">
          <rect x={7} y={1} width={2} height={4} className="fill-moss" />
          <rect x={4} y={2} width={3} height={2} className="fill-moss" />
          <rect x={9} y={2} width={3} height={2} className="fill-moss" />
          <rect x={4} y={2} width={1} height={1} className="fill-moss/70" />
        </g>

        {/* eyes */}
        <g className="ho-bot-eyes">
          <rect x={5} y={6} width={2} height={2} className="fill-background" />
          <rect x={9} y={6} width={2} height={2} className="fill-background" />
        </g>

        {/* honey core, pulsing like a little engine */}
        <rect
          x={7}
          y={10}
          width={2}
          height={2}
          className="ho-bot-core fill-honey"
        />

        {/* scan beam sweeping the body */}
        <g clipPath={`url(#${clip})`}>
          <rect
            className="ho-bot-scan fill-honey"
            x={2}
            y={8}
            width={12}
            height={1}
          />
        </g>

        {/* drifting data motes */}
        <rect
          className="ho-bot-mote fill-honey/80"
          x={1}
          y={6}
          width={1}
          height={1}
          style={{ animationDelay: "0s" }}
        />
        <rect
          className="ho-bot-mote fill-moss/80"
          x={14}
          y={9}
          width={1}
          height={1}
          style={{ animationDelay: "1.2s" }}
        />
        <rect
          className="ho-bot-mote fill-honey/70"
          x={13}
          y={4}
          width={1}
          height={1}
          style={{ animationDelay: "2.1s" }}
        />
      </svg>
    </div>
  );
}
