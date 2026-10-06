// In-house pixel icon set: every glyph is drawn on a shared 8x8 grid.
interface P {
  className?: string;
  size?: number;
  style?: React.CSSProperties;
}

function grid(rows: string[]) {
  const cells: [number, number][] = [];
  for (const [y, r] of rows.entries()) {
    for (const [x, c] of [...r].entries()) {
      if (c === "#") {
        cells.push([x, y]);
      }
    }
  }
  return cells;
}

function Pix({ rows, className, style, size = 16 }: P & { rows: string[] }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 8 8"
      shapeRendering="crispEdges"
      className={className}
      style={style}
      fill="currentColor"
      aria-hidden
    >
      {grid(rows).map(([x, y]) => (
        <rect key={`${x}-${y}`} x={x} y={y} width={1} height={1} />
      ))}
    </svg>
  );
}

export const LeafIcon = (p: P) => (
  <Pix
    {...p}
    rows={[
      "....###.",
      "..#####.",
      ".######.",
      ".#####..",
      ".####...",
      "..##....",
      ".#......",
      "#.......",
    ]}
  />
);
export const SparkleIcon = (p: P) => (
  <Pix
    {...p}
    rows={[
      "...#....",
      "...#....",
      "..###...",
      "#######.",
      "..###...",
      "...#....",
      "...#....",
      "........",
    ]}
  />
);
export const SendIcon = (p: P) => (
  <Pix
    {...p}
    rows={[
      "#.......",
      "###.....",
      "#####...",
      "#######.",
      "#####...",
      "###.....",
      "#.......",
      "........",
    ]}
  />
);
export const SearchIcon = (p: P) => (
  <Pix
    {...p}
    rows={[
      ".####...",
      "#....#..",
      "#....#..",
      "#....#..",
      ".####...",
      ".....##.",
      "......##",
      "........",
    ]}
  />
);
export const PinIcon = (p: P) => (
  <Pix
    {...p}
    rows={[
      "..###...",
      ".#####..",
      ".##.##..",
      ".#####..",
      "..###...",
      "..###...",
      "...#....",
      "........",
    ]}
  />
);
export const ClockIcon = (p: P) => (
  <Pix
    {...p}
    rows={[
      ".####...",
      "#..#.#..",
      "#..#.#..",
      "#..###..",
      "#....#..",
      ".####...",
      "........",
      "........",
    ]}
  />
);
export const ArrowIcon = (p: P) => (
  <Pix
    {...p}
    rows={[
      "........",
      "....#...",
      "....##..",
      "#######.",
      "....##..",
      "....#...",
      "........",
      "........",
    ]}
  />
);
export const MoonIcon = (p: P) => (
  <Pix
    {...p}
    rows={[
      "..###...",
      ".##.....",
      "##......",
      "##......",
      "##......",
      ".##...#.",
      "..####..",
      "........",
    ]}
  />
);

/** Deterministic pixel monogram used as a placeholder company logo. */
export function PixelLogo({ name, hue }: { name: string; hue: number }) {
  let h = 0;
  for (const c of name) {
    h = (h * 31 + (c.codePointAt(0) ?? 0)) >>> 0;
  }
  const cells: [number, number][] = [];
  for (let y = 0; y < 5; y++) {
    for (let x = 0; x < 3; x++) {
      if ((h >> (y * 3 + x)) & 1) {
        cells.push([x, y]);
        if (x < 2) {
          cells.push([4 - x, y]);
        }
      }
    }
  }
  return (
    <svg
      viewBox="-1 -1 7 7"
      shapeRendering="crispEdges"
      className="engraved size-11 shrink-0 rounded-[0.7rem]"
      aria-hidden
    >
      {cells.map(([x, y]) => (
        <rect
          key={`${x}-${y}`}
          x={x}
          y={y}
          width={1}
          height={1}
          fill={`oklch(0.62 0.13 ${hue})`}
        />
      ))}
    </svg>
  );
}
