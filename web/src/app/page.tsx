import { ApiStatus } from "@/components/ApiStatus";

export default function Home() {
  return (
    <main className="flex flex-1 flex-col items-center justify-center gap-6 bg-zinc-950 px-4 text-center text-zinc-100">
      <h1 className="text-4xl font-semibold tracking-tight">Ghostline</h1>
      <p className="max-w-md text-lg text-zinc-400">
        Film anything that moves. An AI physicist finds the law behind the motion and proves it
        with a ghost that predicts where the object goes next.
      </p>
      <ApiStatus />
    </main>
  );
}
