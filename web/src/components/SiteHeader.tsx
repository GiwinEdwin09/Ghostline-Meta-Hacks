import Link from "next/link";

import { ButtonLink, GhostMark } from "@/components/ui";

const LINKS = [
  { href: "/examples", label: "Examples" },
  { href: "/benchmark", label: "Benchmark" },
  { href: "/about", label: "About" },
];

export function SiteHeader() {
  const demo = process.env.NEXT_PUBLIC_USE_MOCKS === "1";
  return (
    <header className="sticky top-0 z-30 border-b border-line bg-background/90 backdrop-blur supports-[backdrop-filter]:bg-background/75">
      <div className="mx-auto flex h-14 w-full max-w-6xl items-center gap-3 px-4">
        <Link href="/" className="flex min-h-11 items-center gap-2 font-semibold tracking-tight">
          <GhostMark className="size-6 text-ghost" />
          Ghostline
        </Link>
        {demo ? (
          <span className="whitespace-nowrap rounded-full border border-amber-400/40 bg-amber-400/10 px-2 py-0.5 text-[11px] font-medium text-amber-300">
            Demo<span className="hidden sm:inline"> data</span>
          </span>
        ) : null}
        <nav aria-label="Main" className="ml-auto hidden items-center gap-1 sm:flex">
          {LINKS.map((l) => (
            <Link
              key={l.href}
              href={l.href}
              className="inline-flex min-h-11 items-center rounded-lg px-3 text-sm text-muted hover:bg-surface-2 hover:text-foreground"
            >
              {l.label}
            </Link>
          ))}
        </nav>
        <ButtonLink href="/new" tone="primary" className="ml-auto whitespace-nowrap px-3 sm:ml-2 sm:px-4">
          New<span className="hidden sm:inline"> experiment</span>
        </ButtonLink>
        <details className="group relative sm:hidden">
          <summary
            aria-label="Menu"
            className="flex size-11 cursor-pointer list-none items-center justify-center rounded-lg text-muted hover:bg-surface-2 [&::-webkit-details-marker]:hidden"
          >
            <svg viewBox="0 0 24 24" className="size-5" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
              <path d="M4 7h16M4 12h16M4 17h16" strokeLinecap="round" />
            </svg>
          </summary>
          <div className="absolute right-0 top-12 w-48 rounded-xl border border-line-strong bg-surface p-1.5 shadow-xl">
            {LINKS.map((l) => (
              <Link
                key={l.href}
                href={l.href}
                className="flex min-h-11 items-center rounded-lg px-3 text-sm hover:bg-surface-2"
              >
                {l.label}
              </Link>
            ))}
          </div>
        </details>
      </div>
    </header>
  );
}

export function SiteFooter() {
  return (
    <footer className="mt-16 border-t border-line">
      <div className="mx-auto flex w-full max-w-6xl flex-wrap items-center justify-between gap-3 px-4 py-6 text-sm text-muted">
        <p>Ghostline. Built for the Meta AI Developer Hackathon.</p>
        <p className="flex gap-4">
          <Link href="/about" className="underline-offset-4 hover:underline">
            How it works
          </Link>
          <Link href="/benchmark" className="underline-offset-4 hover:underline">
            Benchmark
          </Link>
        </p>
      </div>
    </footer>
  );
}
