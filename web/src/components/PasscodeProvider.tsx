"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";

import { registerPasscodePrompter } from "@/lib/passcode";
import { Button, Dialog, cn, inputClass } from "@/components/ui";

/**
 * Asks for the write-routes passcode when the API needs one (build-plan Parts 21, 24).
 * The value is stored by `promptForPasscode` in localStorage, inside try/catch.
 */
export function PasscodeProvider({ children }: { children: ReactNode }) {
  const [request, setRequest] = useState<{ reason: "missing" | "rejected" } | null>(null);
  const [value, setValue] = useState("");
  const resolver = useRef<((v: string | null) => void) | null>(null);

  useEffect(() => {
    registerPasscodePrompter(
      (reason) =>
        new Promise<string | null>((resolve) => {
          resolver.current?.(null);
          resolver.current = resolve;
          setValue("");
          setRequest({ reason });
        }),
    );
    return () => registerPasscodePrompter(null);
  }, []);

  const finish = (result: string | null) => {
    resolver.current?.(result);
    resolver.current = null;
    setRequest(null);
  };

  return (
    <>
      {children}
      <Dialog open={request !== null} onClose={() => finish(null)} title="Passcode needed">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            finish(value.trim() || null);
          }}
        >
          <p className="text-sm leading-relaxed text-fg-2">
            {request?.reason === "rejected"
              ? "That passcode didn't work. Check the submission notes and try again."
              : "Starting an analysis costs real compute, so it needs the passcode from the submission notes. Reading the examples never does."}
          </p>
          <label className="mt-4 block text-sm font-medium" htmlFor="passcode">
            Passcode
          </label>
          <input
            id="passcode"
            type="password"
            autoComplete="off"
            autoCapitalize="none"
            autoFocus
            value={value}
            onChange={(e) => setValue(e.target.value)}
            aria-invalid={request?.reason === "rejected" && value === "" ? true : undefined}
            className={cn(inputClass, "mt-1.5 h-11 w-full font-mono")}
          />
          <div className="mt-5 flex justify-end gap-2">
            <Button tone="ghost" onClick={() => finish(null)}>
              Cancel
            </Button>
            <Button tone="primary" type="submit" disabled={!value.trim()}>
              Continue
            </Button>
          </div>
        </form>
      </Dialog>
    </>
  );
}
