import { createFileRoute, Link } from "@tanstack/react-router";
import { useEffect, useState } from "react";

import { LearningCurve } from "@/components/pixel-diagrams";
import {
  ArrowIcon,
  LeafIcon,
  MoonIcon,
  PinIcon,
  SparkleIcon,
} from "@/components/pixel-icons";

export const Route = createFileRoute("/profile")({
  component: Profile,
  head: () => ({
    meta: [
      { title: "HO · Your profile" },
      {
        content:
          "Your HO profile: plan usage, application stats, preferences and the experience HO writes from.",
        name: "description",
      },
      { content: "HO · Your profile", property: "og:title" },
      {
        content: "Plan, usage, stats and preferences for your HO job search.",
        property: "og:description",
      },
      { content: "website", property: "og:type" },
      { content: "summary", name: "twitter:card" },
    ],
  }),
});

const USER = {
  email: "aarav@example.com",
  headline: "Backend engineer · 3 yrs · distributed systems",
  initials: "AM",
  joined: "Joined Aug 2026",
  location: "Bengaluru, India",
  name: "Aarav Mehta",
};

const USAGE = [
  { l: "Personalized applications", max: 1000, used: 412 },
  { l: "Fit job postings", max: 3000, used: 1870 },
  { l: "Live review minutes", max: 300, used: 96 },
];

const WEEK = [
  { d: "Mon", v: 14 },
  { d: "Tue", v: 22 },
  { d: "Wed", v: 18 },
  { d: "Thu", v: 27 },
  { d: "Fri", v: 31 },
  { d: "Sat", v: 9 },
  { d: "Sun", v: 12 },
];

const FUNNEL = [
  { l: "Applied", v: 412 },
  { l: "Viewed", v: 186 },
  { l: "Replied", v: 54 },
  { l: "Interview", v: 17 },
  { l: "Offer", v: 2 },
];

const PREFS = [
  {
    k: "Target roles",
    v: ["Backend Engineer", "Platform Engineer", "Data Infra"],
  },
  { k: "Locations", v: ["Remote", "Bengaluru", "Berlin"] },
  { k: "Company stage", v: ["Seed", "Series A", "Series B"] },
  { k: "Work type", v: ["Full-time", "Contract"] },
];

const FACTS = [
  { k: "Expected salary", v: "₹32–40 LPA" },
  { k: "Notice period", v: "30 days" },
  { k: "Work authorization", v: "India · needs visa for EU" },
  { k: "Open to relocate", v: "Yes" },
];

const EXPERIENCE = [
  {
    note: "Cut p99 latency 40% on the payments ledger.",
    org: "Kestrel Pay",
    role: "Backend Engineer",
    when: "2024 – now",
  },
  {
    note: "Built a log processor handling 2M events a minute.",
    org: "Lumen Analytics",
    role: "Software Engineer",
    when: "2022 – 2024",
  },
];

const PROJECTS = [
  {
    d: "Rust event pipeline with exactly-once sinks",
    n: "streamsift",
    tag: "Rust · Kafka",
  },
  {
    d: "Postgres-backed job queue, 8k jobs/s",
    n: "tinyq",
    tag: "Go · Postgres",
  },
];

const SKILLS = [
  "Go",
  "Rust",
  "Postgres",
  "Kafka",
  "Kubernetes",
  "gRPC",
  "Redis",
  "Terraform",
];

function pixelMeterTone(i: number, on: number, cells: number): string {
  if (i >= on) {
    return "bg-muted";
  }
  if (i > cells * 0.8) {
    return "bg-honey";
  }
  return "bg-moss";
}

function PixelMeter({ used, max }: { used: number; max: number }) {
  const cells = 24;
  const on = Math.round((used / max) * cells);
  return (
    <div className="engraved flex gap-[3px] rounded-[0.5rem] p-1.5">
      {Array.from({ length: cells }, (_, i) => (
        <span
          key={i}
          className={`h-3 flex-1 ${pixelMeterTone(i, on, cells)}`}
        />
      ))}
    </div>
  );
}

function WeekChart() {
  const max = 32;
  return (
    <div className="flex h-40 items-end gap-2 sm:gap-3">
      {WEEK.map((w) => {
        const blocks = Math.round((w.v / max) * 10);
        return (
          <div key={w.d} className="flex flex-1 flex-col items-center gap-2">
            <span className="font-pixel text-muted-foreground text-[10px]">
              {w.v}
            </span>
            <div className="flex w-full flex-col-reverse gap-[3px]">
              {Array.from({ length: blocks }, (_, i) => (
                <span
                  key={i}
                  className={`h-2 w-full ${i === blocks - 1 ? "bg-honey" : "bg-moss"}`}
                />
              ))}
            </div>
            <span className="font-pixel text-muted-foreground text-[10px]">
              {w.d}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function Funnel() {
  const [head] = FUNNEL;
  const top = head === undefined ? 1 : head.v;
  return (
    <div className="space-y-3">
      {FUNNEL.map((f, i) => {
        const cells = Math.max(1, Math.round((f.v / top) * 30));
        return (
          <div
            key={f.l}
            className="grid grid-cols-[5.5rem_1fr_3rem] items-center gap-3"
          >
            <span className="font-pixel text-xs">{f.l}</span>
            <div className="flex gap-[2px]">
              {Array.from({ length: cells }, (_, k) => (
                <span
                  key={k}
                  className={`h-3 w-[calc((100%-58px)/30)] ${i >= 3 ? "bg-honey" : "bg-moss"}`}
                />
              ))}
            </div>
            <span className="font-pixel text-muted-foreground text-right text-xs">
              {f.v}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function Card({
  title,
  children,
  className = "",
}: {
  title: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section className={`glass rounded-[1.8rem] p-5 sm:p-7 ${className}`}>
      <h2 className="text-lg sm:text-xl">{title}</h2>
      <div className="mt-5">{children}</div>
    </section>
  );
}

function Profile() {
  const [dusk, setDusk] = useState(false);
  const [notify, setNotify] = useState({
    autoSend: false,
    digest: true,
    replies: true,
  });
  useEffect(() => {
    document.documentElement.classList.toggle("dark", dusk);
  }, [dusk]);

  return (
    <div className="min-h-dvh px-4 pb-16 sm:px-8">
      <header className="mx-auto flex max-w-6xl items-center justify-between py-5 sm:py-6">
        <Link to="/" className="flex items-center gap-2">
          <LeafIcon size={20} className="text-moss" />
          <span className="font-pixel text-lg">HO</span>
        </Link>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setDusk((d) => !d)}
            aria-label={dusk ? "Switch to light" : "Switch to dark"}
            className={`glass grid size-10 place-items-center rounded-full transition-colors ${dusk ? "text-honey" : "text-muted-foreground hover:text-foreground"}`}
          >
            <MoonIcon size={16} />
          </button>
          <Link
            to="/app"
            className="btn-honey group flex h-10 items-center gap-2 rounded-full px-5 text-sm font-bold"
          >
            Back to assistant
            <ArrowIcon
              size={12}
              className="transition-transform duration-200 group-hover:translate-x-1"
            />
          </Link>
        </div>
      </header>

      <main className="mx-auto max-w-6xl space-y-5">
        {/* Identity + plan */}
        <div className="grid gap-5 lg:grid-cols-[1.3fr_1fr]">
          <section className="glass flex flex-col gap-6 rounded-[1.8rem] p-6 sm:flex-row sm:items-center sm:p-8">
            <span className="engraved font-pixel text-bark grid size-24 shrink-0 place-items-center rounded-[1.4rem] text-3xl">
              {USER.initials}
            </span>
            <div className="min-w-0 flex-1">
              <h1 className="text-2xl sm:text-3xl">{USER.name}</h1>
              <p className="text-muted-foreground mt-1">{USER.headline}</p>
              <div className="text-muted-foreground mt-3 flex flex-wrap gap-x-4 gap-y-1 text-sm">
                <span className="flex items-center gap-1.5">
                  <PinIcon size={12} />
                  {USER.location}
                </span>
                <span>{USER.email}</span>
                <span>{USER.joined}</span>
              </div>
              <div className="mt-4 flex flex-wrap gap-2">
                <span className="gem gem-fit font-pixel px-3 py-1 text-[11px]">
                  Profile 92% complete
                </span>
                <span className="glass rounded-full px-3 py-1 text-xs font-semibold">
                  resume_aarav.pdf
                </span>
              </div>
            </div>
            <button className="engraved h-10 shrink-0 rounded-[0.9rem] px-4 text-sm font-bold">
              Edit profile
            </button>
          </section>

          <div className="pixel-frame is-pop">
            <div className="pixel-inner flex h-full flex-col p-6">
              <div className="flex items-center justify-between">
                <h2 className="text-xl">Pro plan</h2>
                <span className="gem gem-fit font-pixel px-3 py-1 text-[11px]">
                  Active
                </span>
              </div>
              <p className="mt-3">
                <span className="font-pixel text-3xl">$22</span>
                <span className="text-muted-foreground">
                  {" "}
                  a month · renews Nov 6
                </span>
              </p>
              <div className="mt-auto flex gap-2 pt-6">
                <button className="btn-honey h-10 flex-1 rounded-[0.9rem] text-sm font-bold">
                  Upgrade to Max
                </button>
                <button className="engraved h-10 flex-1 rounded-[0.9rem] text-sm font-bold">
                  Manage billing
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Usage */}
        <Card title="This month's usage">
          <div className="grid gap-6 md:grid-cols-3">
            {USAGE.map((u) => (
              <div key={u.l}>
                <div className="flex items-baseline justify-between gap-2">
                  <span className="text-sm font-bold">{u.l}</span>
                  <span className="font-pixel text-muted-foreground text-xs">
                    {u.used.toLocaleString()} / {u.max.toLocaleString()}
                  </span>
                </div>
                <div className="mt-2">
                  <PixelMeter used={u.used} max={u.max} />
                </div>
              </div>
            ))}
          </div>
        </Card>

        {/* Stats */}
        <div className="grid gap-5 sm:grid-cols-4">
          {[
            { l: "Applications sent", n: "412" },
            { l: "Reply rate", n: "13%" },
            { l: "Interviews", n: "17" },
            { l: "Offers", n: "2" },
          ].map((s) => (
            <div key={s.l} className="glass rounded-[1.4rem] p-5">
              <p className="font-pixel text-moss text-3xl">{s.n}</p>
              <p className="text-muted-foreground mt-2 text-sm">{s.l}</p>
            </div>
          ))}
        </div>

        <div className="grid gap-5 lg:grid-cols-2">
          <Card title="Applications this week">
            <WeekChart />
          </Card>
          <Card title="Where they land">
            <Funnel />
          </Card>
        </div>

        <Card title="HO is getting sharper for you">
          <LearningCurve />
          <p className="text-muted-foreground mt-3 text-sm">
            Qualified applications per run, tuned on your replies and
            rejections.
          </p>
        </Card>

        {/* Preferences */}
        <div className="grid gap-5 lg:grid-cols-[1.3fr_1fr]">
          <Card title="What you're looking for">
            <div className="space-y-4">
              {PREFS.map((p) => (
                <div key={p.k}>
                  <p className="text-muted-foreground text-xs font-bold tracking-wide uppercase">
                    {p.k}
                  </p>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {p.v.map((v) => (
                      <span
                        key={v}
                        className="engraved rounded-full px-3 py-1.5 text-sm font-semibold"
                      >
                        {v}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </Card>
          <Card title="Details">
            <dl className="divide-border divide-y">
              {FACTS.map((f) => (
                <div
                  key={f.k}
                  className="flex justify-between gap-4 py-3 text-sm"
                >
                  <dt className="text-muted-foreground">{f.k}</dt>
                  <dd className="text-right font-bold">{f.v}</dd>
                </div>
              ))}
            </dl>
          </Card>
        </div>

        {/* Experience */}
        <div className="grid gap-5 lg:grid-cols-2">
          <Card title="Experience HO writes from">
            <ol className="space-y-4">
              {EXPERIENCE.map((e) => (
                <li key={e.org} className="engraved rounded-[1.1rem] p-4">
                  <div className="flex flex-wrap items-baseline justify-between gap-2">
                    <p className="font-bold">
                      {e.role} · {e.org}
                    </p>
                    <span className="font-pixel text-muted-foreground text-xs">
                      {e.when}
                    </span>
                  </div>
                  <p className="mt-2 flex gap-2 text-sm">
                    <SparkleIcon
                      size={12}
                      className="text-honey mt-1 shrink-0"
                    />
                    {e.note}
                  </p>
                </li>
              ))}
            </ol>
          </Card>
          <Card title="Projects and skills">
            <div className="space-y-3">
              {PROJECTS.map((p) => (
                <div
                  key={p.n}
                  className="engraved flex items-center justify-between gap-3 rounded-[1.1rem] p-4"
                >
                  <div>
                    <p className="font-pixel text-sm">{p.n}</p>
                    <p className="text-muted-foreground mt-1 text-sm">{p.d}</p>
                  </div>
                  <span className="text-moss shrink-0 text-xs font-semibold">
                    {p.tag}
                  </span>
                </div>
              ))}
            </div>
            <div className="mt-5 flex flex-wrap gap-2">
              {SKILLS.map((s) => (
                <span
                  key={s}
                  className="glass font-pixel rounded-full px-3 py-1 text-[11px]"
                >
                  {s}
                </span>
              ))}
            </div>
          </Card>
        </div>

        {/* Settings */}
        <Card title="Settings">
          <div className="grid gap-3 md:grid-cols-3">
            {(
              [
                ["digest", "Daily match digest", "A short list every morning."],
                [
                  "replies",
                  "Reply alerts",
                  "Ping me when a company writes back.",
                ],
                [
                  "autoSend",
                  "Auto-send verified drafts",
                  "Skip my review for strong fits.",
                ],
              ] as const
            ).map(([k, t, d]) => (
              <button
                key={k}
                onClick={() => setNotify((n) => ({ ...n, [k]: !n[k] }))}
                aria-pressed={notify[k]}
                className="engraved flex items-center justify-between gap-4 rounded-[1.1rem] p-4 text-left"
              >
                <span>
                  <span className="block text-sm font-bold">{t}</span>
                  <span className="text-muted-foreground mt-1 block text-xs">
                    {d}
                  </span>
                </span>
                <span
                  className={`flex h-6 w-11 shrink-0 items-center p-1 transition-colors ${notify[k] ? "bg-moss justify-end" : "bg-muted justify-start"}`}
                >
                  <span className="bg-background size-4" />
                </span>
              </button>
            ))}
          </div>
          <div className="mt-5 flex flex-wrap items-center justify-between gap-3 text-sm">
            <span className="text-muted-foreground">
              Local agents: Claude Code, pi, Codex ·{" "}
              <span className="text-honey">soon</span>
            </span>
            <div className="flex gap-2">
              <button className="engraved h-10 rounded-[0.9rem] px-4 font-bold">
                Export my data
              </button>
              <button className="engraved text-destructive h-10 rounded-[0.9rem] px-4 font-bold">
                Sign out
              </button>
            </div>
          </div>
        </Card>
      </main>
    </div>
  );
}
