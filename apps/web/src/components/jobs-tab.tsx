import { useMemo, useState } from "react";

import { CATEGORIES, JOBS } from "@/lib/jobs-data";
import type { Job } from "@/lib/jobs-data";

import {
  ArrowIcon,
  ClockIcon,
  PinIcon,
  PixelLogo,
  SearchIcon,
  SparkleIcon,
} from "./pixel-icons";

export function JobCard({ job, i }: { job: Job; i: number }) {
  return (
    <article
      className="glass card-lift rise flex flex-col gap-4 rounded-[1.4rem] p-5"
      style={{ animationDelay: `${Math.min(i, 8) * 50}ms` }}
    >
      <div className="flex items-center gap-3">
        <PixelLogo name={job.company} hue={job.hue} />
        <div className="min-w-0">
          <p className="truncate font-bold">{job.company}</p>
          <p className="text-muted-foreground text-xs">Found on {job.source}</p>
        </div>
        <span
          className={`gem font-pixel ml-auto shrink-0 px-3 py-1 text-[11px] ${job.fit ? "gem-fit" : "gem-misfit"}`}
        >
          {job.fit ? "Good fit" : "Bad fit"}
        </span>
      </div>

      <div>
        <h3 className="text-[17px] leading-snug">{job.title}</h3>
        <div className="text-muted-foreground mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[13px]">
          <span className="flex items-center gap-1.5">
            <ClockIcon size={12} />
            {job.posted}
          </span>
          <span className="flex items-center gap-1.5">
            <PinIcon size={12} />
            {job.location}
          </span>
          <span>{job.type}</span>
        </div>
        <p className="text-foreground/80 mt-3 line-clamp-2 text-sm leading-relaxed">
          {job.description}
        </p>
      </div>

      <div className="engraved flex gap-2.5 rounded-[0.9rem] p-3 text-[13px] leading-relaxed">
        <SparkleIcon
          size={12}
          className={`mt-1 shrink-0 ${job.fit ? "text-honey" : "text-misfit"}`}
        />
        <p>{job.advice}</p>
      </div>

      <a
        href={job.url}
        target="_blank"
        rel="noreferrer"
        className="btn-honey group mt-auto flex h-11 items-center justify-center gap-2 rounded-[0.9rem] font-bold"
      >
        Apply
        <ArrowIcon
          size={14}
          className="transition-transform duration-200 group-hover:translate-x-1"
        />
      </a>
    </article>
  );
}

export function JobsTab() {
  const [q, setQ] = useState("");
  const [cat, setCat] = useState("All");

  const list = useMemo(() => {
    const s = q.toLowerCase();
    return JOBS.filter(
      (j) =>
        (cat === "All" || j.tags.includes(cat)) &&
        (!s ||
          `${j.title} ${j.company} ${j.location} ${j.description}`
            .toLowerCase()
            .includes(s))
    );
  }, [q, cat]);

  return (
    <div className="tab-in mx-auto w-full max-w-6xl pb-12">
      <label className="glass flex items-center gap-3 rounded-[1.4rem] p-2">
        <span className="engraved focus-within:ring-ring flex h-12 flex-1 items-center gap-3 rounded-[1rem] px-4 focus-within:ring-2">
          <SearchIcon size={16} className="text-muted-foreground" />
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search roles, companies, places"
            className="placeholder:text-muted-foreground h-full flex-1 bg-transparent text-[15px] outline-none"
          />
        </span>
      </label>

      <div className="no-scrollbar -mx-4 mt-4 flex gap-2 overflow-x-auto px-4 pb-1">
        {CATEGORIES.map((c) => (
          <button
            key={c}
            onClick={() => setCat(c)}
            className={`shrink-0 rounded-full px-4 py-2 text-sm font-semibold transition-colors ${
              cat === c
                ? "btn-honey"
                : "glass text-foreground/80 hover:text-foreground"
            }`}
          >
            {c}
          </button>
        ))}
      </div>

      <p className="text-muted-foreground mt-6 px-1 text-sm">
        {list.length} {list.length === 1 ? "match" : "matches"} ·{" "}
        {list.filter((j) => j.fit).length} good fits
      </p>

      {list.length ? (
        <div className="mt-3 grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {list.map((j, i) => (
            <JobCard key={j.id} job={j} i={i} />
          ))}
        </div>
      ) : (
        <div className="glass mt-3 rounded-[1.4rem] p-10 text-center">
          <h3>Nothing here yet</h3>
          <p className="text-muted-foreground mt-2 text-sm">
            Try another word or category.
          </p>
        </div>
      )}
    </div>
  );
}
