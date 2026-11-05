import { repoPathUrl } from '../site'
import { link } from './styles'

export function Footer() {
  return (
    <footer className="mt-24 border-t border-line pt-8 text-sm text-ink-muted">
      <p>
        InferScale is released under the{' '}
        <a href={repoPathUrl('LICENSE')} className={link} target="_blank" rel="noopener noreferrer">
          BSD 3-Clause License
        </a>
        .
      </p>
    </footer>
  )
}
