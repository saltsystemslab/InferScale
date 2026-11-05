import type { ReactNode } from 'react'
import { newTab, repoPathUrl } from '../site'
import { inlineCode, link } from './styles'

/** A link to a repository path on GitHub; without children the path itself is shown as code. */
export function RepoFile({ path, children }: { path: string; children?: ReactNode }) {
  return (
    <a href={repoPathUrl(path)} {...newTab} className={link}>
      {children ?? <code className={inlineCode}>{path}</code>}
    </a>
  )
}
