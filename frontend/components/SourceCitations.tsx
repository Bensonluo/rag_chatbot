import type { SourceCitation } from '@/lib/api';

export function SourceCitations({
  citations,
  isEn,
}: {
  citations: SourceCitation[];
  isEn: boolean;
}) {
  if (!citations.length) return null;

  return (
    <details className="mt-3 border-t border-border/60 pt-2 text-xs">
      <summary className="cursor-pointer rounded py-1 font-medium text-foreground focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2">
        {isEn ? 'References' : '参考资料'} ({citations.length})
      </summary>
      <p className="mt-1 text-muted-foreground">
        {isEn
          ? 'Snapshots of the references retrieved for this answer. They do not verify every statement.'
          : '本次回答使用的参考资料快照，不代表每句话都已逐条核验。'}
      </p>
      <ol className="mt-2 space-y-3">
        {citations.map((citation, index) => {
          const title = citation.title || (citation.source_type === 'faq'
            ? (isEn ? 'Curated FAQ' : '已审核常见问题')
            : citation.source_type === 'graph'
              ? (isEn ? 'Knowledge graph reference' : '知识图谱参考')
              : (isEn ? 'Document reference' : '文档参考'));
          const page = citation.page_start;
          const lastPage = citation.page_end;
          return (
            <li key={index} className="rounded-md border border-border/60 bg-background/60 p-3">
              <p className="break-words font-medium">{index + 1}. {title}</p>
              <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-muted-foreground">
                {page != null && page > 0 && (
                  <span>
                    {isEn ? 'Page' : '页码'} {page}
                    {lastPage != null && lastPage > page ? `–${lastPage}` : ''}
                  </span>
                )}
                {citation.version_id ? (
                  <span className="break-all" title={citation.version_id}>
                    {isEn ? 'Version' : '版本'} {citation.version_id.startsWith('sha256:')
                      ? citation.version_id.slice(0, 19)
                      : citation.version_id.slice(0, 8)}
                  </span>
                ) : citation.source_type === 'graph' && citation.source_version_ids.length > 0 ? (
                  <span>{isEn ? 'Graph synthesis' : '图谱综合资料'}</span>
                ) : (
                  <span>{isEn ? 'Version not recorded' : '未记录版本'}</span>
                )}
              </div>
              {citation.section_path.length > 0 && (
                <p className="mt-1 break-words text-muted-foreground">{citation.section_path.join(' / ')}</p>
              )}
              {citation.source_type === 'graph' && citation.source_version_ids.length > 0 && (
                <p className="mt-1 break-all text-muted-foreground">
                  {isEn ? 'Graph source versions: ' : '图谱来源版本：'}
                  {citation.source_version_ids.join(', ')}
                </p>
              )}
              <blockquote className="mt-2 whitespace-pre-wrap break-words border-l-2 border-primary/30 pl-3 leading-relaxed">
                {citation.excerpt || (isEn ? 'No excerpt available.' : '暂无可用片段。')}
              </blockquote>
            </li>
          );
        })}
      </ol>
    </details>
  );
}
