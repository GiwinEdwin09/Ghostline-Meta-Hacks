import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";

import { PasscodeProvider } from "@/components/PasscodeProvider";
import { SiteFooter, SiteHeader } from "@/components/SiteHeader";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: { default: "Ghostline", template: "%s · Ghostline" },
  description: "A physics lab in your phone, with an AI lab partner that shows its work.",
};

export const viewport: Viewport = {
  themeColor: "#07090c",
  colorScheme: "dark",
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}>
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
