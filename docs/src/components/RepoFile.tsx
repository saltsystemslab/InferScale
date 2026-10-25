import type { ReactNode } from 'react'
import { repoPathUrl } from '../site'
import { inlineCode, link } from './styles'

/** A link to a repository path on GitHub; without children the path itself is shown as code. */
export function RepoFile({ path, children }: { path: string; children?: ReactNode }) {
  return (
    <a href={repoPathUrl(path)} className={link}>
      {children ?? <code className={inlineCode}>{path}</code>}
    </a>
  )
}
