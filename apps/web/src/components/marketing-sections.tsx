import { Link } from "@tanstack/react-router";
import { useState } from "react";

import { JOBS } from "@/lib/jobs-data";

import { JobCard } from "./jobs-tab";
import { LearningCurve } from "./pixel-diagrams";
import {
  ArrowIcon,
  LeafIcon,
  PixelLogo,
  SearchIcon,
  SendIcon,
  SparkleIcon,
} from "./pixel-icons";

const SOURCES = [
  { h: 240, n: "LinkedIn" },
  { h: 45, n: "Y Combinator" },
  { h: 30, n: "Workday" },
  { h: 150, n: "Greenhouse" },
  { h: 200, n: "Lever" },
  { h: 280, n: "Ashby" },
];

const STEPS = [
  {
    d: "Drop in your resume and a few real projects. That's the only form you'll fill.",
    t: "Tell HO who you are",
  },
  {
    d: "HO sweeps career sites and boards, dedupes everything and ranks each role against you.",
    t: "It goes hunting",
  },
  {
    d: "Every match says apply or skip, with one honest line on why.",
    t: "You get a short list",
  },
  {
    d: "Grounded in your actual work, verified, then sent. You review live if you like.",
    t: "Applications go out",
  },
  {
    d: "Replies, rejections and silence all feed back, so next week's picks are sharper.",
    t: "It learns",
  },
];

const PLANS = [
  {
    items: [
      "400 personalized applications a month",
      "1,500 average fit job postings a month",
    ],
    m: 12,
    name: "Starter",
    y: 8,
  },
  {
    items: [
      "1,000 personalized applications a month",
      "3,000 average fit job postings a month",
      "300 minutes of live review a month",
    ],
    m: 22,
    name: "Pro",
    pop: true,
    y: 16,
  },
  {
    items: [
      "3,000 applications a month",
      "6,000 job postings a month",
      "600 minutes of live review a month",
    ],
    m: 99,
    name: "Max",
    y: 99,
  },
];

export function MarketingSections() {
  const [yearly, setYearly] = useState(false);
  return (
    <>
      {/* Sources */}
      <section className="mx-auto max-w-6xl pb-20">
        <div className="glass rounded-[1.8rem] p-6 sm:p-10">
          <p className="text-muted-foreground text-center text-sm">
            Jobs from company career sites and job boards, including
          </p>
          <div className="mt-6 flex flex-wrap items-center justify-center gap-3">
            {SOURCES.map((s) => (
              <span
                key={s.n}
                className="engraved flex items-center gap-2.5 rounded-full py-1.5 pr-4 pl-1.5"
              >
                <span className="[&>svg]:size-8">
                  <PixelLogo name={s.n} hue={s.h} />
                </span>
                <span className="font-pixel text-sm">{s.n}</span>
              </span>
            ))}
            <span className="font-pixel text-moss text-sm">+50 more</span>
          </div>
        </div>
      </section>

      {/* Statement */}
      <section className="mx-auto max-w-6xl pb-20 text-center">
        <h2 className="mx-auto max-w-3xl text-3xl leading-tight sm:text-5xl">
          Your evenings deserve better than copy‑pasting cover letters.
        </h2>
        <p className="text-muted-foreground mx-auto mt-5 max-w-lg text-lg">
          Hand HO the next hundred applications. Keep your energy for the
          conversations that matter.
        </p>
      </section>

      {/* Pricing */}
      <section id="pricing" className="mx-auto max-w-6xl pb-20">
        <div className="text-center">
          <h2 className="text-2xl sm:text-4xl">Plans and pricing</h2>
          <p className="text-muted-foreground mt-3">
            Start with a 3-day free trial on any plan. Change or cancel anytime.
          </p>
          <div className="engraved mx-auto mt-6 inline-flex rounded-full p-1">
            {[false, true].map((y) => (
              <button
                key={String(y)}
                onClick={() => setYearly(y)}
                className={`rounded-full px-5 py-2 text-sm font-bold transition-colors ${yearly === y ? "btn-honey" : "text-muted-foreground hover:text-foreground"}`}
              >
                {y ? "Yearly" : "Monthly"}
              </button>
            ))}
          </div>
        </div>
        <div className="mt-8 grid gap-5 md:grid-cols-3">
          {PLANS.map((p) => (
            <div
              key={p.name}
              className={`pixel-frame flex ${p.pop ? "is-pop" : ""}`}
            >
              <div className="pixel-inner relative flex w-full flex-col overflow-hidden p-6">
                {[0, 1, 2].map((k) => (
                  <LeafIcon
                    key={k}
                    size={10 + k * 2}
                    className={`leaf-sway pointer-events-none absolute top-0 ${p.pop ? "text-honey" : "text-moss"}`}
                    style={{
                      animationDelay: `${k * 2}s`,
                      right: `${12 + k * 22}%`,
                    }}
                  />
                ))}
                <div className="flex items-center justify-between">
                  <h3 className="text-xl">{p.name}</h3>
                  {p.pop && (
                    <span className="gem gem-fit font-pixel px-3 py-1 text-[11px]">
                      Most popular
                    </span>
                  )}
                </div>
                <p className="mt-5">
                  <span className="font-pixel text-4xl">
                    ${yearly ? p.y : p.m}
                  </span>
                  <span className="text-muted-foreground"> a month</span>
                </p>
                <ul className="mt-6 flex flex-1 flex-col gap-3 text-sm">
                  {p.items.map((it) => (
                    <li key={it} className="flex gap-2.5">
                      <SparkleIcon
                        size={12}
                        className="text-moss mt-1 shrink-0"
                      />
                      {it}
                    </li>
                  ))}
                </ul>
                <Link
                  to="/app"
                  className={`group mt-7 flex h-11 items-center justify-center gap-2 rounded-[0.9rem] font-bold ${p.pop ? "btn-honey" : "engraved"}`}
                >
                  Start free trial
                  <ArrowIcon
                    size={12}
                    className="transition-transform duration-200 group-hover:translate-x-1"
                  />
                </Link>
              </div>
            </div>
          ))}
        </div>
        <p className="text-muted-foreground mt-6 text-center text-sm">
          Your plan starts with a 3-day free trial. During the trial you get 10
          applications and 50 job postings. Cancel anytime.
        </p>
      </section>
    </>
  );
}

export function HowItWorks() {
  return (
    <section id="how" className="mx-auto max-w-6xl pb-20">
      <h2 className="text-2xl sm:text-4xl">How it works</h2>
      <p className="text-muted-foreground mt-3 max-w-md">
        Five minutes of setup. Then it just keeps going.
      </p>
      <div className="mt-10 flex flex-col gap-6">
        {STEPS.map((s, i) => (
          <div
            key={s.t}
            className="glass grid items-center gap-6 rounded-[1.8rem] p-5 sm:p-8 lg:grid-cols-2 lg:gap-12"
          >
            <div className={i % 2 ? "lg:order-2" : ""}>
              <span className="font-pixel text-honey text-3xl">0{i + 1}</span>
              <h3 className="mt-3 text-xl sm:text-2xl">{s.t}</h3>
              <p className="text-muted-foreground mt-3 max-w-sm leading-relaxed">
                {s.d}
              </p>
            </div>
            <div className="engraved rounded-[1.4rem] p-4 sm:p-6">
              <StepDemo i={i} />
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function Bot({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-end gap-2.5">
      <LeafIcon size={16} className="text-moss mb-3 shrink-0" />
      <div className="bubble-bot rounded-[1.25rem] rounded-bl-md px-4 py-3 text-sm leading-relaxed">
        {children}
      </div>
    </div>
  );
}
function Me({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex justify-end">
      <div className="bubble-user rounded-[1.25rem] rounded-br-md px-4 py-3 text-sm leading-relaxed">
        {children}
      </div>
    </div>
  );
}

function StepDemo({ i }: { i: number }) {
  if (i === 0) {
    return (
      <div className="space-y-3">
        <Me>
          Here's my resume. I built a log processor that handles 2M events a
          minute.
        </Me>
        <Bot>Nice. I'll use that as proof for backend and data roles.</Bot>
        <div className="glass flex items-center gap-2 rounded-[1.2rem] p-2">
          <span className="engraved text-muted-foreground flex h-10 flex-1 items-center rounded-[0.8rem] px-3 text-sm">
            Add a project link
          </span>
          <span className="btn-honey grid size-10 place-items-center rounded-[0.8rem]">
            <SendIcon size={14} />
          </span>
        </div>
      </div>
    );
  }
  if (i === 1) {
    return (
      <div className="space-y-3">
        <div className="glass flex items-center gap-3 rounded-[1.2rem] p-2">
          <span className="engraved text-muted-foreground flex h-10 flex-1 items-center gap-2 rounded-[0.8rem] px-3 text-sm">
            <SearchIcon size={14} /> backend · remote · seed to series B
          </span>
        </div>
        <div className="flex flex-wrap gap-2">
          {["Greenhouse 412", "Lever 188", "Ashby 96", "Career pages 731"].map(
            (c) => (
              <span
                key={c}
                className="glass rounded-full px-3 py-1.5 text-xs font-semibold"
              >
                {c}
              </span>
            )
          )}
        </div>
        <p className="font-pixel text-muted-foreground text-xs">
          1,427 found · 1,102 after dedup · 38 ranked fits
        </p>
      </div>
    );
  }
  if (i === 2) {
    const [job] = JOBS;
    if (job === undefined) {
      return null;
    }
    return <JobCard job={job} i={0} />;
  }
  if (i === 3) {
    return (
      <div className="space-y-3">
        <Bot>
          Draft for Fernwood Labs is ready. I cited your log processor and the
          40% latency cut.
        </Bot>
        <Me>Looks good, send it.</Me>
        <Bot>Verified and sent. I'll tell you when they reply.</Bot>
      </div>
    );
  }
  return (
    <div>
      <LearningCurve />
      <p className="text-muted-foreground mt-3 text-xs">
        Each run, HO keeps what got replies and drops what didn't.
      </p>
    </div>
  );
}
