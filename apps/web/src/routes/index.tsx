import { createFileRoute, Link } from "@tanstack/react-router";
import { useEffect, useState } from "react";

import { JobCard } from "@/components/jobs-tab";
import { HowItWorks, MarketingSections } from "@/components/marketing-sections";
import {
  GateDiagram,
  LearningCurve,
  PipelineDiagram,
  PixelLoop,
  ThroughputChart,
} from "@/components/pixel-diagrams";
import {
  ArrowIcon,
  LeafIcon,
  MoonIcon,
  SparkleIcon,
} from "@/components/pixel-icons";
import { PipelineMascot } from "@/components/pixel-mascot";
import { PixelSword } from "@/components/pixel-sword";
import { JOBS } from "@/lib/jobs-data";

const TITLE = "HO · A job search engine that learns";
const DESC =
  "HO discovers jobs across the web, ranks them against real experience, writes grounded applications and learns from every outcome.";

export const Route = createFileRoute("/")({
  component: Landing,
  head: () => ({
    meta: [
      { title: TITLE },
      { content: DESC, name: "description" },
      { content: TITLE, property: "og:title" },
      { content: DESC, property: "og:description" },
      { content: "website", property: "og:type" },
      { content: "summary_large_image", name: "twitter:card" },
    ],
  }),
});

const BENEFITS = [
  {
    d: "HO watches ATS boards, career pages and startup lists around the clock, so you see jobs hours after they go live.",
    n: "24/7",
    t: "Never miss a fresh role",
  },
  {
    d: "Every match comes with a plain apply or skip, and the reason why. No more guessing if you're qualified.",
    n: "1 line",
    t: "Honest advice, not noise",
  },
  {
    d: "Applications are written from your real projects and results, then checked before anything is sent.",
    n: "0",
    t: "Generic cover letters",
  },
];

const FOOTER: { h: string; links: { t: string; href: string }[] }[] = [
  {
    h: "Product",
    links: [
      { href: "/app", t: "Open assistant" },
      { href: "#how", t: "How it works" },
      { href: "#pricing", t: "Pricing" },
    ],
  },
  {
    h: "Open source",
    links: [
      { href: "https://github.com/parazeeknova/ho", t: "GitHub" },
      {
        href: "https://github.com/parazeeknova/ho/issues",
        t: "Report an issue",
      },
    ],
  },
  {
    h: "People",
    links: [
      { href: "https://przknv.cc", t: "Author · pzk" },
      { href: "https://itssingularity.com", t: "Singularity Works" },
    ],
  },
];

function Landing() {
  const [heroJob] = JOBS;
  const [dusk, setDusk] = useState(false);
  useEffect(() => {
    document.documentElement.classList.toggle("dark", dusk);
  }, [dusk]);

  return (
    <div className="min-h-dvh px-4 sm:px-8">
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
            Open assistant
            <ArrowIcon
              size={12}
              className="transition-transform duration-200 group-hover:translate-x-1"
            />
          </Link>
        </div>
      </header>

      {/* Hero */}
      <section className="mx-auto grid max-w-6xl items-center gap-12 pt-10 pb-24 sm:pt-16 lg:grid-cols-[1.1fr_1fr] lg:gap-16">
        <div>
          <h1 className="text-4xl leading-[1.15] sm:text-6xl">
            A job search
            <br />
            that learns.
          </h1>
          <p className="text-muted-foreground mt-6 max-w-md text-lg leading-relaxed">
            HO, the Hyperdimensional Orchestrator, scans the web for fresh
            roles, ranks them against your real experience and tells you plainly
            what to apply for.
          </p>
          <div className="mt-9 flex flex-wrap items-center gap-5">
            <Link
              to="/app"
              className="btn-honey group flex h-12 items-center gap-2 rounded-2xl px-6 font-bold"
            >
              Meet your assistant
              <ArrowIcon
                size={14}
                className="transition-transform duration-200 group-hover:translate-x-1"
              />
            </Link>
            <span className="text-muted-foreground flex items-center gap-2 text-sm">
              <SparkleIcon size={12} className="text-honey" />
              12 new matches today
            </span>
          </div>
        </div>
        <div className="relative mx-auto w-full max-w-md">
          <div className="bubble-bot mb-4 ml-6 rounded-[1.25rem] rounded-bl-md px-4 py-3 text-[15px] leading-relaxed">
            Fernwood Labs is a strong fit. Their event pipeline looks a lot like
            your log processor.
          </div>
          {heroJob === undefined ? null : <JobCard job={heroJob} i={0} />}
        </div>
      </section>

      {/* Why */}
      <section className="mx-auto max-w-6xl pb-24">
        <h2 className="max-w-xl text-2xl leading-snug sm:text-4xl">
          Stop scrolling job boards. Start getting replies.
        </h2>
        <div className="mt-10 grid gap-x-10 gap-y-8 sm:grid-cols-3">
          {BENEFITS.map((b) => (
            <div key={b.t}>
              <p className="font-pixel text-moss text-3xl">{b.n}</p>
              <h3 className="mt-3 text-lg">{b.t}</h3>
              <p className="text-muted-foreground mt-2 leading-relaxed">
                {b.d}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* Loop */}
      <section className="mx-auto grid max-w-6xl items-center gap-10 pb-20 lg:grid-cols-[0.8fr_1.2fr]">
        <div>
          <h2 className="text-2xl sm:text-4xl">One loop, always running</h2>
          <p className="text-muted-foreground mt-4 max-w-sm leading-relaxed">
            Most tools search and stop. HO treats every application as an
            experiment, watches what happens, and quietly gets better at picking
            the next one.
          </p>
        </div>
        <div className="glass rounded-[1.8rem] p-4 sm:p-8">
          <PixelLoop />
        </div>
      </section>

      {/* Pipeline */}
      <section className="mx-auto max-w-6xl pb-20">
        <div className="glass rounded-[1.8rem] p-5 sm:p-8">
          <div className="flex items-center gap-4 sm:gap-6">
            <PipelineMascot />
            <div className="min-w-0">
              <h2 className="text-xl sm:text-2xl">Discovery to decision</h2>
              <p className="text-muted-foreground mt-1 max-w-md text-sm leading-relaxed">
                Ranking is kept apart from discovery, so new sources earn trust.
              </p>
            </div>
          </div>
          <div className="engraved mt-6 rounded-[1.2rem] p-4 sm:p-6">
            <PipelineDiagram />
          </div>
        </div>
      </section>

      <HowItWorks />

      {/* Charts */}
      <section className="mx-auto max-w-6xl pb-8">
        <h2 className="max-w-2xl text-2xl leading-snug sm:text-4xl">
          Fast where it should be. Careful where it counts.
        </h2>
        <p className="text-muted-foreground mt-4 max-w-xl leading-relaxed">
          HO reads thousands of postings a minute so you don't have to, then
          slows right down before anything goes out with your name on it.
        </p>
      </section>
      <section className="mx-auto grid max-w-6xl gap-5 pb-20 lg:grid-cols-[1.25fr_1fr]">
        <div className="glass rounded-[1.8rem] p-5 sm:p-8">
          <h2 className="text-xl sm:text-2xl">Peak throughput / min</h2>
          <p className="text-muted-foreground mt-2 text-sm">
            Log scale. Fast at finding, deliberately slow at applying.
          </p>
          <div className="mt-6">
            <ThroughputChart />
          </div>
          <p className="mt-5 font-bold">
            Fresh roles land in your list while other applicants are still
            refreshing.
          </p>
          <p className="text-muted-foreground mt-2 text-xs">
            i7 12th Gen · 32 GB DDR5 · RTX 3060 · 700 Mbps
          </p>
          <div className="pixel-frame mt-6">
            <div className="pixel-inner p-4 sm:p-5">
              <p className="font-bold">
                Run the whole pipeline locally, with your own agents.
              </p>
              <div className="mt-3 flex flex-wrap items-center gap-2">
                {["Claude Code", "pi", "Codex"].map((a) => (
                  <span
                    key={a}
                    className="glass font-pixel rounded-full px-3 py-1 text-[11px]"
                  >
                    {a} <span className="text-honey">· soon</span>
                  </span>
                ))}
              </div>
              <p className="text-muted-foreground mt-3 text-xs">
                Support for all local agents is coming soon!!
              </p>
            </div>
          </div>
        </div>
        <div className="flex flex-col gap-5">
          <div className="glass rounded-[1.8rem] p-5 sm:p-8">
            <h2 className="text-xl sm:text-2xl">It gets sharper</h2>
            <p className="text-muted-foreground mt-2 text-sm">
              Qualified applications per run.{" "}
              <span className="text-honey">■</span> new policy promoted.
            </p>
            <div className="mt-5">
              <LearningCurve />
            </div>
            <p className="mt-4 text-sm font-bold">
              Week four picks beat week one. Every outcome teaches it.
            </p>
          </div>
          <div className="glass rounded-[1.8rem] p-5 sm:p-8">
            <h2 className="text-xl sm:text-2xl">Fails closed</h2>
            <div className="mt-4">
              <GateDiagram />
            </div>
            <p className="mt-4 text-sm font-bold">
              If a claim can't be backed by your real work, it never gets sent.
              No embarrassing applications.
            </p>
          </div>
        </div>
      </section>

      <MarketingSections />

      <section className="mx-auto max-w-6xl pb-16">
        <div className="glass rounded-[1.8rem] px-6 py-14 text-center sm:py-20">
          <h2 className="text-3xl sm:text-5xl">
            Discover. Measure. Adapt. Repeat.
          </h2>
          <p className="text-muted-foreground mx-auto mt-5 max-w-md">
            Let HO do the searching, sorting and second-guessing. You just show
            up for the interviews.
          </p>
          <Link
            to="/app"
            className="btn-honey group mx-auto mt-9 inline-flex h-12 items-center gap-2 rounded-2xl px-6 font-bold"
          >
            Open the assistant
            <ArrowIcon
              size={14}
              className="transition-transform duration-200 group-hover:translate-x-1"
            />
          </Link>
        </div>
      </section>

      <footer className="mx-auto max-w-6xl pb-10">
        <div className="glass grid gap-8 rounded-[1.8rem] p-6 sm:grid-cols-2 sm:p-10 lg:grid-cols-[1.4fr_1fr_1fr_1fr_auto]">
          <div>
            <Link to="/" className="flex items-center gap-2">
              <LeafIcon size={20} className="text-moss" />
              <span className="font-pixel text-lg">HO</span>
            </Link>
            <p className="text-muted-foreground mt-3 max-w-xs text-sm leading-relaxed">
              The Hyperdimensional Orchestrator. A job search that learns. Open
              source.
            </p>
          </div>
          {FOOTER.map((col) => (
            <div key={col.h}>
              <p className="font-pixel text-sm">{col.h}</p>
              <ul className="text-muted-foreground mt-3 space-y-2 text-sm">
                {col.links.map((l) => (
                  <li key={l.t}>
                    {l.href === "/app" ? (
                      <Link
                        to="/app"
                        className="hover:text-foreground transition-colors"
                      >
                        {l.t}
                      </Link>
                    ) : (
                      <a
                        href={l.href}
                        {...(l.href.startsWith("http")
                          ? { rel: "noreferrer", target: "_blank" }
                          : {})}
                        className="hover:text-foreground transition-colors"
                      >
                        {l.t}
                      </a>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          ))}
          <div className="flex items-center justify-center sm:col-span-2 lg:col-span-1">
            <PixelSword />
          </div>
        </div>
      </footer>
    </div>
  );
}
