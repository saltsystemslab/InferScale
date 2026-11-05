import { createContext, useContext, useEffect, useRef, useState, type ComponentProps } from 'react'
import { cx } from '../cx'
import { CheckIcon, CopyIcon } from './Icons'
import { inlineCode } from './styles'

const InCodeBlock = createContext(false)

/** Inline code; inside a code block the highlighted code renders untouched. */
export function Code({ className, ...props }: ComponentProps<'code'>) {
  const inBlock = useContext(InCodeBlock)
  return <code className={inBlock ? className : cx(inlineCode, className)} {...props} />
}

/** A fenced code block, already highlighted by Shiki at build time, with a copy button. */
export function Pre({ className, children, ...props }: ComponentProps<'pre'>) {
  const ref = useRef<HTMLPreElement>(null)
  return (
    <div className="group relative my-6">
      <pre
        ref={ref}
        className={cx(
          className,
          'overflow-x-auto rounded-lg border border-line bg-surface py-3.5 pr-14 pl-4 font-mono text-[13px] leading-6',
        )}
        {...props}
      >
        <InCodeBlock value>{children}</InCodeBlock>
      </pre>
      <CopyButton getText={() => ref.current?.textContent ?? ''} />
    </div>
  )
}

function CopyButton({ getText }: { getText: () => string }) {
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    if (!copied) return
    const timeout = setTimeout(() => setCopied(false), 2000)
    return () => clearTimeout(timeout)
  }, [copied])

  const copy = () => {
    navigator.clipboard.writeText(getText()).then(
      () => setCopied(true),
      // Without clipboard access the reader can still select the code by hand.
      () => {},
    )
  }

  return (
    <>
      <button
        type="button"
        onClick={copy}
        aria-label="Copy code"
        className="absolute top-2.5 right-2.5 grid size-8 place-items-center rounded-md border border-line bg-canvas text-ink-muted opacity-0 transition group-hover:opacity-100 hover:text-ink focus-visible:opacity-100 focus-visible:outline-2 focus-visible:outline-accent [@media(hover:none)]:opacity-100"
      >
        {copied ? <CheckIcon className="size-4 text-accent" /> : <CopyIcon className="size-4" />}
      </button>
      <output className="sr-only">{copied ? 'Copied to clipboard' : ''}</output>
    </>
  )
}
