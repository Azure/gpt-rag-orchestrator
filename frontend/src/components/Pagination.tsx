import { ChevronLeft, ChevronRight } from "lucide-react";

interface PaginationProps {
  page: number;
  pageSize: number;
  hasMore: boolean;
  onChange: (page: number) => void;
}

export function Pagination({ page, pageSize, hasMore, onChange }: PaginationProps) {
  const start = (page - 1) * pageSize + 1;
  const end = start + pageSize - 1;
  return (
    <div className="flex items-center justify-between px-2 py-3">
      <span className="text-sm text-muted-foreground">
        Showing {start}–{end}
      </span>
      <div className="flex items-center gap-1">
        <button
          disabled={page <= 1}
          onClick={() => onChange(page - 1)}
          className="rounded-md p-1 hover:bg-accent disabled:opacity-40"
          aria-label="Previous page"
        >
          <ChevronLeft className="h-4 w-4" />
        </button>
        <button
          disabled={!hasMore}
          onClick={() => onChange(page + 1)}
          className="rounded-md p-1 hover:bg-accent disabled:opacity-40"
          aria-label="Next page"
        >
          <ChevronRight className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}
