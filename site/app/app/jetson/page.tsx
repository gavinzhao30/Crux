import type { Metadata } from "next";

import { PageHeader } from "@/components/app/page-header";
import { JetsonRun } from "@/components/jetson-run";

export const metadata: Metadata = { title: "Jetson run" };

export default function JetsonRunPage() {
  return (
    <div className="space-y-10">
      <PageHeader eyebrow="Live on hardware" title="One letter, read by a camera">
        The same engine on an NVIDIA Jetson. It chooses which part of a DNA strip to read, reads it through the
        camera, and settles which haemoglobin variant it carries. Then someone swaps the strip.
      </PageHeader>
      <JetsonRun />
    </div>
  );
}
