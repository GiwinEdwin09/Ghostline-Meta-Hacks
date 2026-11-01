import type { Metadata } from "next";

import { AdminView } from "@/components/AdminView";
import { PageHeader } from "@/components/ui";

export const metadata: Metadata = { title: "Admin", robots: { index: false } };

export default function AdminPage() {
  return (
    <main className="flex-1">
      <PageHeader eyebrow="§ Operations" title="Admin" />
      <div className="mx-auto w-full max-w-4xl px-4 pt-8 sm:pt-10">
        <AdminView />
      </div>
    </main>
  );
}
