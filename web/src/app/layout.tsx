import type { Metadata, Viewport } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans, Instrument_Serif } from "next/font/google";

import { PasscodeProvider } from "@/components/PasscodeProvider";
import { SiteFooter, SiteHeader } from "@/components/SiteHeader";
import "./globals.css";

const plexSans = IBM_Plex_Sans({
  variable: "--font-plex-sans",
  subsets: ["latin"],
});

const plexMono = IBM_Plex_Mono({
  variable: "--font-plex-mono",
  subsets: ["latin"],
  weight: ["400", "500"],
});

const instrumentSerif = Instrument_Serif({
  variable: "--font-instrument-serif",
  subsets: ["latin"],
  weight: "400",
  style: ["normal", "italic"],
});

export const metadata: Metadata = {
  title: { default: "Ghostline", template: "%s · Ghostline" },
  description: "A physics lab in your phone, with an AI lab partner that shows its work.",
};

export const viewport: Viewport = {
  themeColor: "#0b0e13",
  colorScheme: "dark",
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${plexSans.variable} ${plexMono.variable} ${instrumentSerif.variable} h-full antialiased`}>
      <body className="flex min-h-full flex-col">
        <PasscodeProvider>
          <SiteHeader />
          <div className="flex flex-1 flex-col">{children}</div>
          <SiteFooter />
        </PasscodeProvider>
      </body>
    </html>
  );
}
