import { useEffect, useRef, useState } from "react";

import { ArrowIcon, LeafIcon, SendIcon } from "./pixel-icons";

interface Msg {
  id: number;
  from: "me" | "bot";
  text: string;
  cta?: boolean;
}

const SEED: Msg[] = [
  { from: "me", id: 1, text: "Morning! Anything new for backend internships?" },
  {
    from: "bot",
    id: 2,
    text: "Morning, Aarav. I found 12 new matches since yesterday. 5 are strong fits for you.",
  },
  {
    cta: true,
    from: "bot",
    id: 3,
    text: "Two stand out. Fernwood Labs wants a Rust intern for their event pipeline, which is close to your log processor project. Copperleaf AI needs help with GPU scheduling and is fully remote.",
  },
  { from: "me", id: 4, text: "Nice. Why did the Lumen Grove one get flagged?" },
  {
    from: "bot",
    id: 5,
    text: "It asks for 8+ years leading ML teams. Worth revisiting in a few years, but not today.",
  },
];

const CHIPS = [
  "Show remote roles",
  "Why was this a bad fit?",
  "Find me startups",
];

const REPLIES: Record<string, string> = {
  "Find me startups":
    "Fernwood Labs, Copperleaf AI and Hollow Pine are early-stage teams hiring interns right now. All three are good fits.",
  "Show remote roles":
    "4 of today's matches are remote. Fernwood Labs, Copperleaf AI and Oakline are good fits. Thistle is remote too, but leans frontend.",
  "Why was this a bad fit?":
    "Usually it comes down to years of experience or location. Kestrel Systems, for example, needs UK work rights and 4+ years on-call.",
};

export function ChatTab({ onOpenJobs }: { onOpenJobs: () => void }) {
  const [msgs, setMsgs] = useState<Msg[]>(SEED);
  const [typing, setTyping] = useState(false);
  const [draft, setDraft] = useState("");
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [msgs, typing]);

  const send = (text: string) => {
    const t = text.trim();
    if (!t || typing) {
      return;
    }
    setMsgs((m) => [...m, { from: "me", id: Date.now(), text: t }]);
    setDraft("");
    setTyping(true);
    setTimeout(() => {
      setTyping(false);
      setMsgs((m) => [
        ...m,
        {
          cta: true,
          from: "bot",
          id: Date.now() + 1,
          text:
            REPLIES[t] ??
            "Got it. I'll keep an eye out and refresh your matches as new roles appear.",
        },
      ]);
    }, 1400);
  };

  return (
    <div className="tab-in mx-auto flex h-full w-full max-w-3xl flex-col">
      <div className="no-scrollbar flex-1 space-y-4 overflow-y-auto px-1 pt-2 pb-6">
        {msgs.map((m) => (
          <div
            key={m.id}
            className={`rise flex ${m.from === "me" ? "justify-end" : "justify-start"}`}
          >
            <div className="flex max-w-[85%] items-end gap-2.5 sm:max-w-[75%]">
              {m.from === "bot" && (
                <LeafIcon size={18} className="text-moss mb-3 shrink-0" />
              )}
              <div
                className={`rounded-[1.25rem] px-4 py-3 text-[15px] leading-relaxed ${
                  m.from === "me"
                    ? "bubble-user rounded-br-md"
                    : "bubble-bot rounded-bl-md"
                }`}
              >
                {m.text}
                {m.cta && (
                  <button
                    onClick={onOpenJobs}
                    className="group text-accent-foreground mt-2.5 flex items-center gap-1.5 text-sm font-bold"
                  >
                    Open in Jobs
                    <ArrowIcon
                      size={14}
                      className="transition-transform duration-200 group-hover:translate-x-1"
                    />
                  </button>
                )}
              </div>
            </div>
          </div>
        ))}
        {typing && (
          <div
            className="flex items-end gap-2.5"
            aria-label="Assistant is typing"
          >
            <LeafIcon size={18} className="text-moss mb-3" />
            <div className="bubble-bot flex gap-1.5 rounded-[1.25rem] rounded-bl-md px-4 py-4">
              {[0, 1, 2].map((i) => (
                <span
                  key={i}
                  className="typing-dot bg-accent-foreground block size-1.5"
                  style={{ animationDelay: `${i * 0.15}s` }}
                />
              ))}
            </div>
          </div>
        )}
        <div ref={endRef} />
      </div>

      <div className="space-y-3 pb-4 sm:pb-6">
        <div className="no-scrollbar -mx-1 flex gap-2 overflow-x-auto px-1">
          {CHIPS.map((c) => (
            <button
              key={c}
              onClick={() => send(c)}
              className="glass text-foreground/85 hover:text-foreground shrink-0 rounded-full px-4 py-2 text-sm font-semibold transition-colors"
            >
              {c}
            </button>
          ))}
        </div>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            send(draft);
          }}
          className="glass flex items-center gap-2 rounded-[1.4rem] p-2"
        >
          <input
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder="Ask about roles, fit, or companies"
            className="engraved placeholder:text-muted-foreground focus-visible:ring-ring h-12 flex-1 rounded-[1rem] bg-transparent px-4 text-[15px] outline-none focus-visible:ring-2"
          />
          <button
            type="submit"
            aria-label="Send"
            disabled={!draft.trim() || typing}
            className="btn-honey grid size-12 place-items-center rounded-[1rem] disabled:opacity-60"
          >
            <SendIcon size={18} />
          </button>
        </form>
      </div>
    </div>
  );
}
