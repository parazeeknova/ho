// Pixel diagrams for the landing page. Everything is drawn on integer grids
// with crispEdges so it reads as pixel art, not vector illustration.
const T = { fontFamily: "var(--font-pixel)" } as const;

/* ---------- 1. The decision loop ---------- */
const LOOP = [
  "Discover",
  "Normalize",
  "Understand",
  "Rank",
  "Personalize",
  "Apply",
  "Observe",
  "Learn",
];

export function PixelLoop() {
  const cx = 60;
  const cy = 60;
  const r = 42;
  const pts = LOOP.map((_, i) => {
    const a = (i / LOOP.length) * Math.PI * 2 - Math.PI / 2;
    return [
      Math.round(cx + r * Math.cos(a)),
      Math.round(cy + r * Math.sin(a)),
    ] as const;
  });
  // stepped (manhattan) path through the ring, pixel style
  const path = pts
    .map(([x, y], i) => {
      const next = pts[(i + 1) % pts.length];
      if (next === undefined) {
        throw new Error("Missing next point in PixelLoop");
      }
      const [nx, ny] = next;
      return `${i === 0 ? `M${x} ${y}` : ""} H${nx} V${ny}`;
    })
    .join(" ");
  const dots: [number, number][] = [];
  for (const [i, pt] of pts.entries()) {
    const [x, y] = pt;
    const next = pts[(i + 1) % pts.length];
    if (next === undefined) {
      throw new Error("Missing next point in PixelLoop");
    }
    const [nx, ny] = next;
    const sx = Math.sign(nx - x);
    const sy = Math.sign(ny - y);
    for (let k = x; k !== nx; k += sx) {
      if (k % 3 === 0) {
        dots.push([k, y]);
      }
    }
    for (let k = y; k !== ny; k += sy) {
      if (k % 3 === 0) {
        dots.push([nx, k]);
      }
    }
  }

  return (
    <svg
      viewBox="-34 4 188 112"
      shapeRendering="crispEdges"
      className="w-full"
      role="img"
      aria-label="HO decision loop: discover, normalize, understand, rank, personalize, apply, observe, learn, then back to rank"
    >
      {dots.map(([x, y]) => (
        <rect
          key={`${x}-${y}`}
          x={x}
          y={y}
          width={1}
          height={1}
          className="fill-bark/50"
        />
      ))}
      {/* Learn -> Rank feedback chord */}
      {Array.from({ length: 13 }, (_, k) => {
        const l = pts.at(7);
        const rPt = pts.at(3);
        if (l === undefined || rPt === undefined) {
          throw new Error("Missing loop point in PixelLoop");
        }
        const [lx, ly] = l;
        const [rx, ry] = rPt;
        const t = k / 12;
        return (
          <rect
            key={k}
            x={Math.round(lx + (rx - lx) * t)}
            y={Math.round(ly + (ry - ly) * t)}
            width={1}
            height={1}
            className="fill-honey"
          />
        );
      })}
      {/* travelling packet */}
      <rect x={-1.5} y={-1.5} width={3} height={3} className="fill-honey">
        <animateMotion dur="9s" repeatCount="indefinite" path={path} />
      </rect>
      {/* core */}
      <rect x={50} y={52} width={20} height={16} rx={0} className="fill-moss" />
      <rect
        x={51}
        y={53}
        width={18}
        height={1}
        className="fill-background/40"
      />
      <text
        x={60}
        y={63.5}
        textAnchor="middle"
        fontSize={7}
        style={T}
        className="fill-background"
      >
        HO
      </text>
      {pts.map(([x, y], i) => {
        const left = x < cx - 4;
        const right = x > cx + 4;
        let tx = x;
        if (left) {
          tx = x - 5;
        } else if (right) {
          tx = x + 5;
        }
        let ty = y + 1.6;
        if (y < cy - 20) {
          ty = y - 5;
        } else if (y > cy + 20) {
          ty = y + 9;
        }
        let anchor: "end" | "start" | "middle" = "middle";
        if (left) {
          anchor = "end";
        } else if (right) {
          anchor = "start";
        }
        return (
          <g key={LOOP[i]}>
            <rect
              x={x - 3}
              y={y - 3}
              width={6}
              height={6}
              className={i === 3 || i === 7 ? "fill-honey" : "fill-moss"}
            />
            <rect
              x={x - 2}
              y={y - 2}
              width={4}
              height={1}
              className="fill-background/50"
            />
            <text
              x={tx}
              y={ty}
              textAnchor={anchor}
              fontSize={4.2}
              style={T}
              className="fill-foreground"
            >
              {String(i + 1).padStart(2, "0")} {LOOP[i]}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/* ---------- 2. Throughput funnel (log scale, pixel blocks) ---------- */
const STAGES: [string, number, string][] = [
  ["Calibration", 1_000_000, "1M+"],
  ["LTR ranking", 500_000, "500k+"],
  ["Event stream", 250_000, "250k+"],
  ["Dedup checks", 50_000, "50k+"],
  ["URL discovery", 25_000, "25k+"],
  ["Vector search", 15_000, "15k+"],
  ["Jobs fetched", 5000, "5k+"],
  ["Embeddings", 2500, "2.5k+"],
  ["Evidence RAG", 500, "500+"],
  ["Browser forms", 60, "60+"],
  ["LLM drafts", 30, "30"],
  ["Submissions", 30, "30+"],
];

export function ThroughputChart() {
  const cols = 48;
  const max = Math.log10(1_000_000);
  return (
    <div className="space-y-2">
      {STAGES.map(([label, v, txt], i) => {
        const n = Math.max(2, Math.round((Math.log10(v) / max) * cols));
        const slow = i >= 9;
        return (
          <div
            key={label}
            className="grid grid-cols-[6.5rem_1fr_3.5rem] items-center gap-3 sm:grid-cols-[8rem_1fr_4rem]"
          >
            <span className="text-muted-foreground truncate text-xs sm:text-sm">
              {label}
            </span>
            <svg
              viewBox={`0 0 ${cols * 2} 2`}
              shapeRendering="crispEdges"
              className="h-3.5 w-full"
              preserveAspectRatio="none"
              aria-hidden
            >
              {Array.from({ length: cols }, (_, k) => {
                let cls: string;
                if (k >= n) {
                  cls = "fill-foreground/[0.07]";
                } else if (slow) {
                  cls = "fill-honey";
                } else {
                  cls = "fill-moss";
                }
                return (
                  <rect
                    key={k}
                    x={k * 2}
                    y={0}
                    width={1.6}
                    height={2}
                    className={cls}
                  />
                );
              })}
            </svg>
            <span
              className={`font-pixel text-right text-xs sm:text-sm ${slow ? "text-honey" : "text-moss"}`}
            >
              {txt}
            </span>
          </div>
        );
      })}
    </div>
  );
}

/* ---------- 3. Discovery & intelligence pipeline ---------- */
interface Node {
  id: string;
  x: number;
  y: number;
  w: number;
  label: string;
  sub?: string;
  hot?: boolean;
}
const NODES: Node[] = [
  {
    id: "s",
    label: "Sources",
    sub: "ATS · careers · search",
    w: 30,
    x: 2,
    y: 22,
  },
  { id: "f", label: "Frontier", sub: "queues · leases", w: 26, x: 40, y: 22 },
  { id: "p", label: "Parse", sub: "normalize · resolve", w: 26, x: 74, y: 22 },
  { id: "d", label: "Dedup", w: 22, x: 108, y: 22 },
  { id: "g", label: "Neo4j", sub: "company · tech", w: 26, x: 138, y: 6 },
  { id: "v", label: "pgvector", sub: "semantic", w: 26, x: 138, y: 38 },
  { id: "x", label: "Features", w: 24, x: 172, y: 22 },
  { hot: true, id: "r", label: "LTR", sub: "LightGBM", w: 24, x: 204, y: 22 },
  { hot: true, id: "e", label: "EV + bandit", w: 26, x: 236, y: 22 },
];
const EDGES: [string, string][] = [
  ["s", "f"],
  ["f", "p"],
  ["p", "d"],
  ["d", "g"],
  ["d", "v"],
  ["g", "x"],
  ["v", "x"],
  ["x", "r"],
  ["r", "e"],
];
const H = 12;

export function PipelineDiagram() {
  const by = Object.fromEntries(NODES.map((n) => [n.id, n]));
  return (
    <div className="no-scrollbar -mx-2 overflow-x-auto px-2">
      <svg
        viewBox="0 0 264 58"
        shapeRendering="crispEdges"
        className="w-full min-w-[760px]"
        role="img"
        aria-label="Discovery pipeline from sources to ranking"
      >
        {EDGES.map(([a, b]) => {
          const A = by[a];
          const B = by[b];
          if (A === undefined || B === undefined) {
            throw new Error(`Missing node for edge ${a}-${b}`);
          }
          const x1 = A.x + A.w;
          const y1 = A.y + H / 2;
          const x2 = B.x;
          const y2 = B.y + H / 2;
          const mx = Math.round((x1 + x2) / 2);
          const segs: [number, number][] = [];
          for (let x = x1 + 1; x < mx; x += 2) {
            segs.push([x, y1]);
          }
          for (let y = Math.min(y1, y2); y <= Math.max(y1, y2); y += 2) {
            segs.push([mx, y]);
          }
          for (let x = mx; x < x2 - 1; x += 2) {
            segs.push([x, y2]);
          }
          return (
            <g key={a + b}>
              {segs.map(([x, y]) => (
                <rect
                  key={`${x}-${y}`}
                  x={x}
                  y={y - 0.5}
                  width={1}
                  height={1}
                  className="fill-bark/60"
                />
              ))}
              <rect
                x={x2 - 2}
                y={y2 - 1.5}
                width={1}
                height={3}
                className="fill-bark"
              />
            </g>
          );
        })}
        {NODES.map((n) => (
          <g key={n.id}>
            <rect
              x={n.x}
              y={n.y}
              width={n.w}
              height={H}
              className={n.hot ? "fill-honey/25" : "fill-moss/15"}
            />
            <rect
              x={n.x}
              y={n.y}
              width={n.w}
              height={1}
              className={n.hot ? "fill-honey" : "fill-moss"}
            />
            <rect
              x={n.x}
              y={n.y + H - 1}
              width={n.w}
              height={1}
              className="fill-foreground/10"
            />
            <text
              x={n.x + n.w / 2}
              y={n.y + (n.sub ? 5.4 : 7.3)}
              textAnchor="middle"
              fontSize={3.6}
              style={T}
              className="fill-foreground"
            >
              {n.label}
            </text>
            {n.sub && (
              <text
                x={n.x + n.w / 2}
                y={n.y + 9.6}
                textAnchor="middle"
                fontSize={2.5}
                className="fill-muted-foreground"
                style={{ fontFamily: "var(--font-sans)" }}
              >
                {n.sub}
              </text>
            )}
          </g>
        ))}
      </svg>
    </div>
  );
}

/* ---------- 4. Learning curve (stepped pixel line) ---------- */
const CURVE = [
  6, 6, 7, 7, 8, 8, 8, 10, 11, 11, 12, 12, 14, 15, 15, 16, 18, 18, 19, 21, 22,
  22, 23, 25, 26, 26, 27, 28,
];
const PROMOTIONS = [7, 12, 16, 19, 23];

export function LearningCurve() {
  const w = CURVE.length * 4;
  const h = 34;
  return (
    <svg
      viewBox={`-6 -2 ${w + 8} ${h + 8}`}
      shapeRendering="crispEdges"
      className="w-full"
      role="img"
      aria-label="Qualified applications rise as new policies are promoted"
    >
      {[0, 10, 20, 30].map((v) => (
        <g key={v}>
          {Array.from({ length: w / 2 }, (_, k) => (
            <rect
              key={k}
              x={k * 2}
              y={h - v}
              width={1}
              height={0.4}
              className="fill-foreground/10"
            />
          ))}
          <text
            x={-2}
            y={h - v + 1}
            textAnchor="end"
            fontSize={2.6}
            style={T}
            className="fill-muted-foreground"
          >
            {v}
          </text>
        </g>
      ))}
      {CURVE.map((v, i) => {
        const prev = CURVE[i - 1];
        return (
          <g key={i}>
            <rect
              x={i * 4}
              y={h - v}
              width={4}
              height={v}
              className="fill-moss/12"
            />
            <rect
              x={i * 4}
              y={h - v}
              width={4}
              height={1}
              className="fill-moss"
            />
            {i > 0 && prev !== undefined && prev < v && (
              <rect
                x={i * 4}
                y={h - v}
                width={1}
                height={v - prev}
                className="fill-moss"
              />
            )}
          </g>
        );
      })}
      {PROMOTIONS.map((i) => {
        const v = CURVE[i];
        if (v === undefined) {
          return null;
        }
        return (
          <g key={i}>
            <rect
              x={i * 4}
              y={h - v - 4}
              width={2}
              height={2}
              className="fill-honey"
            />
            <rect
              x={i * 4 + 0.5}
              y={h - v - 2}
              width={1}
              height={1}
              className="fill-honey/60"
            />
          </g>
        );
      })}
      <text
        x={0}
        y={h + 5}
        fontSize={2.6}
        style={T}
        className="fill-muted-foreground"
      >
        run 1
      </text>
      <text
        x={w}
        y={h + 5}
        textAnchor="end"
        fontSize={2.6}
        style={T}
        className="fill-muted-foreground"
      >
        run 28
      </text>
    </svg>
  );
}

/* ---------- 5. Fail-closed verification gate ---------- */
export function GateDiagram() {
  const steps = ["Evidence", "Local LLM", "Draft"];
  return (
    <svg
      viewBox="0 0 120 54"
      shapeRendering="crispEdges"
      className="w-full"
      role="img"
      aria-label="Drafts pass a deterministic gate: valid ones submit, invalid ones fail closed"
    >
      {steps.map((s, i) => (
        <g key={s}>
          <rect
            x={2 + i * 24}
            y={21}
            width={20}
            height={10}
            className="fill-moss/15"
          />
          <rect
            x={2 + i * 24}
            y={21}
            width={20}
            height={1}
            className="fill-moss"
          />
          <text
            x={12 + i * 24}
            y={27.6}
            textAnchor="middle"
            fontSize={3.2}
            style={T}
            className="fill-foreground"
          >
            {s}
          </text>
          {Array.from({ length: 1 }, () => (
            <rect
              key="a"
              x={22 + i * 24}
              y={25.5}
              width={2}
              height={1}
              className="fill-bark/60"
            />
          ))}
        </g>
      ))}
      {/* the gate */}
      <rect x={74} y={14} width={4} height={24} className="fill-bark" />
      <rect x={75} y={15} width={2} height={1} className="fill-background/40" />
      <text
        x={76}
        y={11}
        textAnchor="middle"
        fontSize={3}
        style={T}
        className="fill-bark"
      >
        verify
      </text>
      {/* valid branch */}
      {[80, 82, 84, 86].map((x) => (
        <rect key={x} x={x} y={20} width={1} height={1} className="fill-fit" />
      ))}
      <rect x={88} y={15} width={30} height={10} className="fill-fit/20" />
      <rect x={88} y={15} width={30} height={1} className="fill-fit" />
      <text
        x={103}
        y={21.6}
        textAnchor="middle"
        fontSize={3.2}
        style={T}
        className="fill-foreground"
      >
        Submit
      </text>
      {/* invalid branch */}
      {[80, 82, 84, 86].map((x) => (
        <rect
          key={x}
          x={x}
          y={32}
          width={1}
          height={1}
          className="fill-misfit"
        />
      ))}
      <rect x={88} y={27} width={30} height={10} className="fill-misfit/15" />
      <rect x={88} y={27} width={30} height={1} className="fill-misfit" />
      <text
        x={103}
        y={33.6}
        textAnchor="middle"
        fontSize={3.2}
        style={T}
        className="fill-foreground"
      >
        Fail closed
      </text>
      <text
        x={60}
        y={50}
        textAnchor="middle"
        fontSize={2.8}
        className="fill-muted-foreground"
        style={{ fontFamily: "var(--font-sans)" }}
      >
        Better to skip than to send a wrong claim.
      </text>
    </svg>
  );
}
