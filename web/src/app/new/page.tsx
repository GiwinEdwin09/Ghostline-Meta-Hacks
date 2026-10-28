import type { Metadata } from "next";

import { Wizard } from "@/components/wizard/Wizard";

export const metadata: Metadata = { title: "New experiment" };

export default function NewExperimentPage() {
  return <Wizard />;
}
