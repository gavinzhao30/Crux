"use client";

import { ArrowUpRightIcon, FileUpIcon } from "lucide-react";
import { useRouter } from "next/navigation";
import { useRef, useState } from "react";

import { ProcessingPanel } from "@/components/app/processing-panel";
import { cn } from "@/lib/utils";

const MAX_BYTES = 20 * 1024 * 1024;
type Phase =
  | { kind: "idle" }
  | { kind: "uploading" | "reading" | "opening"; name: string; percent: number }
  | { kind: "error"; message: string };

export function UploadDropzone({ className }: { className?: string }) {
  const router = useRouter();
  const input = useRef<HTMLInputElement>(null);
  const inFlight = useRef(false);
  const [over, setOver] = useState(false);
  const [phase, setPhase] = useState<Phase>({ kind: "idle" });
  const busy = phase.kind !== "idle" && phase.kind !== "error";

  function upload(file: File | undefined) {
    if (!file || inFlight.current) return;
    if (file.type !== "application/pdf" && !file.name.toLowerCase().endsWith(".pdf")) {
      setPhase({ kind: "error", message: "Choose a PDF to continue." });
      return;
    }
    if (file.size > MAX_BYTES || file.size === 0) {
      setPhase({ kind: "error", message: file.size === 0 ? "That PDF is empty." : "Choose a PDF smaller than 20 MB." });
      return;
    }
    inFlight.current = true;
    setPhase({ kind: "uploading", name: file.name, percent: 0 });
    const body = new FormData();
    body.set("file", file);
    const request = new XMLHttpRequest();
    const fail = (message: string) => {
      inFlight.current = false;
      setPhase({ kind: "error", message });
    };
    request.open("POST", "/api/papers");
    request.timeout = 180_000;
    request.upload.onprogress = (event) => {
      if (event.lengthComputable) setPhase({ kind: "uploading", name: file.name, percent: Math.round(event.loaded / event.total * 100) });
    };
    request.upload.onload = () => setPhase({ kind: "reading", name: file.name, percent: 100 });
    request.onerror = () => fail("The upload was interrupted. Check your connection and try again.");
    request.ontimeout = () => fail("Reading this PDF took too long. Try uploading it again.");
    request.onabort = () => fail("The upload was interrupted. Choose your PDF to try again.");
    request.onload = () => {
      let data: { id?: string; error?: string } = {};
      try { data = JSON.parse(request.responseText); } catch { /* Handle an unreadable response below. */ }
      if (request.status < 200 || request.status >= 300 || !data.id) {
        fail(data.error || "We could not read that PDF. Please try again.");
        return;
      }
      setPhase({ kind: "opening", name: file.name, percent: 100 });
      router.push(`/app/papers/${data.id}`);
      router.refresh();
    };
    request.send(body);
  }

  return (
    <div className={className}>
      {busy ? (
        <ProcessingPanel
          title={phase.kind === "uploading" ? "Your paper is on its way." : phase.kind === "reading" ? "Reading your PDF." : "Your workspace is ready."}
          description={phase.kind === "uploading" ? `Uploading ${phase.name}` : phase.kind === "reading" ? `Extracting readable text from ${phase.name}. Keep this page open.` : "Opening your paper. Next, extract its scientific claims."}
          steps={["Upload PDF", "Read document", "Open workspace"]}
          active={phase.kind === "uploading" ? 0 : phase.kind === "reading" ? 1 : 2}
          percent={phase.kind === "uploading" ? phase.percent : undefined}
        />
      ) : (
        <button
          type="button"
          aria-describedby="upload-status"
          onClick={() => input.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setOver(true); }}
          onDragLeave={() => setOver(false)}
          onDrop={(e) => { e.preventDefault(); setOver(false); upload(e.dataTransfer.files[0]); }}
          className={cn("group flex min-h-64 w-full flex-col items-center justify-center gap-4 rounded-2xl border border-dashed bg-card px-6 py-10 text-center transition-colors hover:border-brand/50 hover:bg-brand/5 focus-visible:ring-3 focus-visible:ring-ring/40 focus-visible:outline-none", over && "border-brand bg-brand/5")}
        >
          <span className="flex size-14 items-center justify-center rounded-2xl border bg-background text-brand"><FileUpIcon className="size-6" aria-hidden="true" /></span>
          <span className="font-display text-2xl">Drop your paper here</span>
          <span className="inline-flex items-center gap-1.5 text-sm font-medium">or choose a PDF <ArrowUpRightIcon className="size-4" aria-hidden="true" /></span>
          <span className="text-xs text-muted-foreground">Text-based PDF · up to 20 MB · saved privately</span>
        </button>
      )}
      <input ref={input} type="file" accept="application/pdf,.pdf" className="sr-only" tabIndex={-1} aria-hidden="true" disabled={busy}
        onChange={(e) => { upload(e.target.files?.[0]); e.target.value = ""; }} />
      <p id="upload-status" role="alert" className="mt-3 min-h-5 text-sm text-destructive">{phase.kind === "error" ? phase.message : ""}</p>
    </div>
  );
}
