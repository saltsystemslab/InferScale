import type { ReactNode } from 'react'
import { inlineCode } from './styles'

/** Names and their descriptions, side by side on wide screens and stacked on narrow ones. */
export function Definitions({ children }: { children: ReactNode }) {
  return <dl className="my-6 divide-y divide-line border-y border-line">{children}</dl>
}

type DefinitionProps = {
  /** The name being described, set in monospace. */
  term: string
  /** The default value, shown as code below the description. */
  default?: string
  /** Marks a setting that has no default. */
  required?: boolean
  children: ReactNode
}

export function Definition({
  term,
  default: defaultValue,
  required = false,
  children,
}: DefinitionProps) {
  return (
    <div className="py-3 sm:grid sm:grid-cols-[13rem_minmax(0,1fr)] sm:gap-x-6">
      <dt className="font-mono text-[13px] leading-6 font-medium break-words text-ink">{term}</dt>
      <dd className="mt-1 text-[15px] leading-6 sm:mt-0">
        {children}
        {required && <span className="mt-1 block text-sm text-ink-muted">Required</span>}
        {defaultValue !== undefined && (
          <span className="mt-1 block text-sm text-ink-muted">
            Default <code className={inlineCode}>{defaultValue}</code>
          </span>
        )}
      </dd>
    </div>
  )
}
