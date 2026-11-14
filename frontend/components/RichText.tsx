/**
 * Minimal markdown-subset renderer for assistant replies.
 *
 * Covers the constrained markup the backend emits in chat replies:
 * `**bold**` inline spans, `- ` / `• ` bullet lines, and paragraphs
 * separated by blank lines. Everything renders through React text
 * nodes, so there is no HTML injection surface.
 */

import { useMemo, type ReactNode } from 'react';

type Block = { kind: 'para'; text: string } | { kind: 'list'; items: string[] };

const LIST_LINE = /^(?:[-•])\s+/;
const BOLD = /\*\*([^*]+)\*\*/;

function parseBlocks(source: string): Block[] {
  const blocks: Block[] = [];
  let para: string[] = [];
  let list: string[] = [];

  const flushPara = () => {
    if (para.length > 0) {
      blocks.push({ kind: 'para', text: para.join('\n') });
      para = [];
    }
  };
  const flushList = () => {
    if (list.length > 0) {
      blocks.push({ kind: 'list', items: list });
      list = [];
    }
  };

  for (const rawLine of source.split('\n')) {
    const line = rawLine.trimEnd();
    if (LIST_LINE.test(line.trimStart())) {
      flushPara();
      list.push(line.trimStart().replace(LIST_LINE, ''));
    } else if (line.trim() === '') {
      flushPara();
      flushList();
    } else {
      flushList();
      para.push(line);
    }
  }
  flushPara();
  flushList();
  return blocks;
}

function renderInline(text: string, keyPrefix: string): ReactNode[] {
  // split() with a capturing group interleaves plain and bold parts
  // at even/odd indices.
  return text.split(BOLD).map((part, index) =>
    index % 2 === 1 ? <strong key={`${keyPrefix}-b${index}`}>{part}</strong> : part
  );
}

export function RichText({ text, className }: { text: string; className?: string }) {
  const blocks = useMemo(() => parseBlocks(text), [text]);
  return (
    <div className={className ?? 'text-sm'}>
      {blocks.map((block, index) =>
        block.kind === 'list' ? (
          <ul
            key={`l${index}`}
            className="my-1 list-disc space-y-1 pl-5 first:mt-0 last:mb-0"
          >
            {block.items.map((item, itemIndex) => (
              <li key={`l${index}-${itemIndex}`}>
                {renderInline(item, `i${index}-${itemIndex}`)}
              </li>
            ))}
          </ul>
        ) : (
          <p key={`p${index}`} className="my-1 whitespace-pre-wrap first:mt-0 last:mb-0">
            {renderInline(block.text, `p${index}`)}
          </p>
        )
      )}
    </div>
  );
}
