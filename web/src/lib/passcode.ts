/**
 * The write-routes passcode (build-plan Parts 21, 24 and 28). It lives in localStorage and
 * every read and write is wrapped in try/catch, because Safari private mode and blocked
 * storage both throw. When a write is refused with 401 the API client asks for a new one
 * through whatever prompter the <PasscodeProvider> registered.
 */

const KEY = "ghostline.passcode";
let memory: string | null = null;

export function getPasscode(): string | null {
  try {
    return window.localStorage.getItem(KEY);
  } catch {
    return null;
  }
}

export function setPasscode(value: string): void {
  try {
    window.localStorage.setItem(KEY, value);
  } catch {
    /* kept in memory only */
    memory = value;
  }
}

export function clearPasscode(): void {
  memory = null;
  try {
    window.localStorage.removeItem(KEY);
  } catch {
    /* ignore */
  }
}

/** Passcode to send: localStorage, or the in-memory copy when storage is unavailable. */
export function currentPasscode(): string | null {
  return getPasscode() ?? memory;
}

export type PasscodePrompter = (reason: "missing" | "rejected") => Promise<string | null>;

let prompter: PasscodePrompter | null = null;

export function registerPasscodePrompter(fn: PasscodePrompter | null): void {
  prompter = fn;
}

/** Ask the user for the passcode (resolves to null if they cancel). */
export async function promptForPasscode(reason: "missing" | "rejected"): Promise<string | null> {
  if (!prompter) return null;
  const value = await prompter(reason);
  if (value) setPasscode(value);
  return value;
}
