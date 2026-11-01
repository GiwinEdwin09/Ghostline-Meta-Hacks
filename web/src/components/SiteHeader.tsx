import Link from "next/link";
import { Suspense } from "react";

import { MainNav, MainNavView, MobileMenu, MobileMenuView } from "@/components/MainNav";
import { ButtonLink, LogoMark } from "@/components/ui";
import { NAV_LINKS } from "@/lib/nav";

export function SiteHeader() {
  const demo = process.env.NEXT_PUBLIC_USE_MOCKS === "1";
  return (
    <header className="sticky top-0 z-30 border-b border-line bg-background/85 backdrop-blur-md supports-[backdrop-filter]:bg-background/70">
      <div className="mx-auto flex h-14 w-full max-w-6xl items-center gap-3 px-4">
        <Link href="/" className="-ml-1 flex min-h-11 items-center gap-2 rounded-md px-1" aria-label="Ghostline, home">
          <LogoMark className="size-7 text-foreground" />
          <span className="font-display text-[1.45rem] leading-none tracking-tight">Ghostline</span>
        </Link>
        {demo ? (
          <span className="whitespace-nowrap rounded-[4px] border border-amber-400/40 bg-amber-400/[0.08] px-1.5 py-1 font-mono text-[0.625rem] font-medium uppercase leading-none tracking-[0.08em] text-amber-300">
            Demo<span className="hidden sm:inline"> data</span>
          </span>
        ) : null}
        {/* The active link reads the pathname, which suspends on /e/[id] while prerendering. */}
        <Suspense fallback={<MainNavView pathname={null} />}>
          <MainNav />
        </Suspense>
        <ButtonLink href="/new" tone="primary" className="ml-auto whitespace-nowrap px-3 sm:ml-1 sm:px-4">
          <span aria-hidden className="text-base leading-none">+</span>
          <span>
            New<span className="hidden sm:inline"> experiment</span>
          </span>
        </ButtonLink>
        <Suspense fallback={<MobileMenuView pathname={null} />}>
          <MobileMenu />
        </Suspense>
      </div>
    </header>
  );
}

export function SiteFooter() {
  return (
    <footer className="mt-20 border-t border-line">
      <div className="mx-auto grid w-full max-w-6xl gap-6 px-4 py-8 sm:grid-cols-[1fr_auto] sm:items-end">
        <div>
          <Link href="/" className="inline-flex min-h-11 items-center gap-2" aria-label="Ghostline, home">
            <LogoMark className="size-6 text-foreground" />
            <span className="font-display text-xl leading-none">Ghostline</span>
          </Link>
          <p className="mt-1 max-w-sm text-sm text-muted">A physics lab in your phone, with an AI lab partner that shows its work. Built for the Meta AI Developer Hackathon.</p>
        </div>
        <nav aria-label="Footer" className="flex flex-wrap gap-x-5 text-sm text-muted">
          {[...NAV_LINKS, { href: "/new", label: "New experiment" }].map((l) => (
            <Link key={l.href} href={l.href} className="inline-flex min-h-11 items-center underline-offset-4 hover:text-foreground hover:underline">
              {l.label}
            </Link>
          ))}
        </nav>
      </div>
      <div className="border-t border-line">
        <p className="eyebrow mx-auto w-full max-w-6xl px-4 py-3 normal-case tracking-[0.04em]">
          x(t) = x₀ + v₀t + ½gt² &nbsp;·&nbsp; every number on this site comes from the harness, not the model
        </p>
      </div>
    </footer>
  );
}
