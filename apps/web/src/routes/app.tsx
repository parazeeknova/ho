import { createFileRoute, Link } from "@tanstack/react-router";
import { useEffect, useState } from "react";

import { ChatTab } from "@/components/chat-tab";
import { JobsTab } from "@/components/jobs-tab";
import { LeafIcon, MoonIcon } from "@/components/pixel-icons";

export const Route = createFileRoute("/app")({
  component: Index,
  head: () => ({
    meta: [
      { title: "HO · Assistant" },
      {
        content:
          "Chat with your assistant and browse fresh jobs matched to your profile every day.",
        name: "description",
      },
      { content: "HO · Assistant", property: "og:title" },
      {
        content:
          "Fresh jobs matched to you, with honest advice on what to apply for.",
        property: "og:description",
      },
      { content: "website", property: "og:type" },
      { content: "summary_large_image", name: "twitter:card" },
    ],
  }),
});

function Index() {
  const [tab, setTab] = useState<"chat" | "jobs">("chat");
  const [dusk, setDusk] = useState(false);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dusk);
  }, [dusk]);

  return (
    <div className="relative h-dvh overflow-hidden">
      <div className="relative flex h-full flex-col px-4 sm:px-8">
        <header className="mx-auto grid w-full max-w-6xl grid-cols-[1fr_auto_1fr] items-center gap-3 py-4 sm:py-6">
          <Link to="/" className="flex items-center gap-2">
            <LeafIcon size={20} className="text-moss" />
            <span className="font-pixel hidden text-lg sm:inline">HO</span>
          </Link>

          <nav
            className="glass relative grid grid-cols-2 rounded-full p-1.5"
            aria-label="Sections"
          >
            <span
              aria-hidden
              className="engraved absolute inset-y-1.5 left-1.5 w-[calc(50%-0.375rem)] rounded-full transition-transform duration-300 ease-[cubic-bezier(.3,.8,.3,1)]"
              style={{
                transform: tab === "jobs" ? "translateX(100%)" : "none",
              }}
            />
            {(["chat", "jobs"] as const).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                aria-current={tab === t}
                className={`font-pixel relative z-10 w-20 rounded-full py-2 text-sm capitalize transition-colors sm:w-28 ${
                  tab === t
                    ? "text-foreground"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                {t}
              </button>
            ))}
          </nav>

          <div className="flex items-center justify-end gap-2">
            <button
              onClick={() => setDusk((d) => !d)}
              aria-label={
                dusk ? "Switch to day forest" : "Switch to twilight forest"
              }
              className={`glass grid size-10 place-items-center rounded-full transition-colors ${dusk ? "text-honey" : "text-muted-foreground hover:text-foreground"}`}
            >
              <MoonIcon size={16} />
            </button>
            <Link
              to="/profile"
              className="glass hover:text-foreground flex items-center gap-2 rounded-full py-1 pr-1 pl-1 transition-colors sm:pr-3"
            >
              <span className="engraved font-pixel text-bark grid size-8 place-items-center rounded-full text-xs">
                AM
              </span>
              <span className="hidden text-sm font-bold sm:inline">Aarav</span>
            </Link>
          </div>
        </header>

        <main className="min-h-0 flex-1">
          {tab === "chat" ? (
            <ChatTab onOpenJobs={() => setTab("jobs")} />
          ) : (
            <div className="no-scrollbar h-full overflow-y-auto">
              <JobsTab />
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
