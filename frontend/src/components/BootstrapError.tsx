interface BootstrapErrorProps {
  title: string;
  message: string;
}

export function BootstrapError({ title, message }: BootstrapErrorProps) {
  return (
    <div
      className="mx-auto flex min-h-[60vh] max-w-xl flex-col items-center justify-center gap-3 p-8 text-center text-sm"
      role="alert"
    >
      <h1 className="text-lg font-semibold">{title}</h1>
      <p className="text-muted-foreground">{message}</p>
      <button
        type="button"
        className="rounded-md bg-primary px-4 py-2 font-medium text-primary-foreground transition hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        onClick={() => window.location.reload()}
      >
        Try again
      </button>
    </div>
  );
}
