/** An HTTP error from the Ghostline API. `status` is what the upload retry logic looks at. */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly detail?: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/** The request never got an answer: offline, DNS, CORS, the server is asleep. */
export class NetworkError extends Error {
  constructor(message = "Could not reach the Ghostline server.") {
    super(message);
    this.name = "NetworkError";
  }
}

export interface FriendlyError {
  /** What happened. */
  title: string;
  /** What to do about it. */
  hint: string;
}

/** Turn any thrown value into a message that says what happened and what to do. */
export function describeError(err: unknown): FriendlyError {
  if (err instanceof NetworkError || (err instanceof TypeError && /fetch|network|load failed/i.test(err.message))) {
    return {
      title: "Can't reach Ghostline.",
      hint: "Check your connection and try again. If you are on cellular, a Wi-Fi network is more reliable.",
    };
  }
  if (err instanceof ApiError) {
    switch (err.status) {
      case 401:
      case 403:
        return { title: "That passcode didn't work.", hint: "Enter the passcode from the submission notes and try again." };
      case 404:
        return { title: "We couldn't find that.", hint: "It may have been deleted. Go back to the examples and open another." };
      case 409:
        return { title: "Something is already running for this experiment.", hint: "Wait for it to finish, then try again." };
      case 413:
        return { title: "That clip is too large.", hint: "Clips can be up to 1 GiB. Trim it in Photos or film a shorter one." };
      case 422:
        return { title: "Ghostline couldn't use that input.", hint: err.message };
      case 429:
        return { title: "Ghostline is busy or the daily limit was reached.", hint: "Examples stay available. Try a new analysis again later." };
      default:
        if (err.status >= 500) {
          return {
            title: "The server had a problem.",
            hint: "Try again in a moment. If it keeps failing, the examples still work.",
          };
        }
    }
    return { title: "That didn't work.", hint: err.message };
  }
  if (err instanceof Error && err.message) return { title: "Something went wrong.", hint: err.message };
  return { title: "Something went wrong.", hint: "Try again." };
}
