import type { ReactNode } from 'react'
import { cx } from '../cx'

type FigureProps = {
  src: string
  alt: string
  width: number
  height: number
  caption?: ReactNode
  /** Mount the image on a white plate, for figures drawn on white like the paper's. */
  plate?: boolean
  /**
   * The text column by default; "wide" extends past it on wide screens for figures
   * with small print, and "narrow" keeps tall figures from filling the screen.
   */
  size?: 'narrow' | 'wide'
}

/** An image that opens at full resolution when clicked. */
export function Figure({ src, alt, width, height, caption, plate = false, size }: FigureProps) {
  return (
    <figure
      className={cx(
        'my-8',
        size === 'wide' && 'xl:-mx-12',
        size === 'narrow' && 'mx-auto max-w-lg',
      )}
    >
      <a
        href={src}
        className={cx(
          'block cursor-zoom-in overflow-hidden rounded-xl border border-line',
          plate && 'bg-white p-3 sm:p-5 dark:brightness-90',
        )}
      >
        <img
          src={src}
          alt={alt}
          width={width}
          height={height}
          loading="lazy"
          decoding="async"
          className="h-auto w-full"
        />
      </a>
      {caption && (
        <figcaption
          className={cx('mt-3 text-sm leading-6 text-ink-muted', size === 'wide' && 'xl:px-12')}
        >
          {caption}
        </figcaption>
      )}
    </figure>
  )
}
