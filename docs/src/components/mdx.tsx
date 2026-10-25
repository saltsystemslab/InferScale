import type { MDXComponents } from 'mdx/types'
import { cx } from '../cx'
import { Code, Pre } from './CodeBlock'
import { Heading } from './Heading'
import { link, table, tableCell, tableFrame, tableHeadCell } from './styles'

/** Tailwind styling for the elements that Markdown produces. */
export const mdxComponents: MDXComponents = {
  h2: (props) => (
    <Heading
      as="h2"
      className="mt-20 border-t border-line pt-10 text-2xl font-semibold tracking-tight text-ink"
      {...props}
    />
  ),
  h3: (props) => (
    <Heading
      as="h3"
      className="mt-12 text-lg font-semibold tracking-tight text-ink [h2+&]:mt-8"
      {...props}
    />
  ),
  h4: ({ children, ...props }) => (
    <h4 className="mt-10 font-semibold text-ink [h3+&]:mt-6" {...props}>
      {children}
    </h4>
  ),
  p: (props) => <p className="mt-4 leading-7" {...props} />,
  a: ({ children, ...props }) => (
    <a className={link} {...props}>
      {children}
    </a>
  ),
  strong: (props) => <strong className="font-semibold text-ink" {...props} />,
  ul: (props) => <ul className="mt-4 list-disc space-y-2 pl-6 marker:text-ink-muted" {...props} />,
  ol: (props) => (
    <ol className="mt-4 list-decimal space-y-2 pl-6 marker:text-ink-muted" {...props} />
  ),
  li: (props) => <li className="pl-1 leading-7" {...props} />,
  blockquote: (props) => (
    <blockquote className="mt-6 border-l-2 border-accent pl-4 text-ink-muted" {...props} />
  ),
  hr: (props) => <hr className="my-12 border-line" {...props} />,
  code: Code,
  pre: Pre,
  table: (props) => (
    <div className={tableFrame}>
      <table className={table} {...props} />
    </div>
  ),
  th: ({ className, ...props }) => <th className={cx(tableHeadCell, className)} {...props} />,
  td: ({ className, ...props }) => <td className={cx(tableCell, className)} {...props} />,
}
