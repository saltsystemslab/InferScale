import type { ComponentProps } from 'react'

type HeadingProps = ComponentProps<'h2'> & { as: 'h2' | 'h3' }

/** A heading that links to itself; the "#" is CSS-generated so it stays out of the nav text. */
export function Heading({ as: Tag, id, className, children }: HeadingProps) {
  return (
    <Tag id={id} className={className}>
      <a
        href={`#${id}`}
        className="after:ml-2 after:text-ink-muted after:opacity-0 after:transition-opacity after:content-['#'] hover:after:opacity-100 focus-visible:after:opacity-100"
      >
        {children}
      </a>
    </Tag>
  )
}
