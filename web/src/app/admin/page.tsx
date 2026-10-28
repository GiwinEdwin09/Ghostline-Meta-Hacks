import type { Metadata } from "next";

import { AdminView } from "@/components/AdminView";

export const metadata: Metadata = { title: "Admin", robots: { index: false } };

export default function AdminPage() {
  return (
    <main className="mx-auto w-full max-w-4xl px-4 py-8 sm:py-12">
      <h1 className="mb-6 text-3xl font-semibold tracking-tight">Admin</h1>
      <AdminView />
    </main>
  );
}
