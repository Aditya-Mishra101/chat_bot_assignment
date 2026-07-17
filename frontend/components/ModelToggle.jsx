"use client";

const OPTIONS = [
  { value: "api", label: "API" },
  { value: "ollama", label: "Ollama" },
];

export default function ModelToggle({ value, onChange, disabled }) {
  return (
    <div
      role="radiogroup"
      aria-label="LLM backend"
      className="inline-flex rounded-lg border border-neutral-200 bg-neutral-100 p-1 text-sm"
    >
      {OPTIONS.map((opt) => {
        const isActive = value === opt.value;
        return (
          <button
            key={opt.value}
            type="button"
            role="radio"
            aria-checked={isActive}
            disabled={disabled}
            onClick={() => onChange(opt.value)}
            className={`rounded-md px-3 py-1.5 font-medium transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-neutral-500 disabled:cursor-not-allowed disabled:opacity-50 ${
              isActive
                ? "bg-white text-neutral-900 shadow-sm"
                : "text-neutral-500 hover:text-neutral-800"
            }`}
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}
