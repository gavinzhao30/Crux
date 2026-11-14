import { Component, type ErrorInfo, type ReactNode } from 'react';

/**
 * Keeps one panel's render error from blanking the whole screen during a
 * live demo. Retries whenever `resetKey` changes (the next event).
 */
export class Boundary extends Component<
  { name: string; resetKey: number; children: ReactNode },
  { failedAt: number | null }
> {
  state = { failedAt: null as number | null };

  static getDerivedStateFromError(): { failedAt: number } {
    return { failedAt: -1 };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error(`[${this.props.name}] render failed`, error, info.componentStack);
    this.setState({ failedAt: this.props.resetKey });
  }

  componentDidUpdate(prev: { resetKey: number }): void {
    if (this.state.failedAt !== null && this.state.failedAt !== -1 && prev.resetKey !== this.props.resetKey) {
      this.setState({ failedAt: null });
    }
  }

  render(): ReactNode {
    if (this.state.failedAt !== null) {
      return (
        <section className="flex min-h-0 items-center justify-center rounded-[0.9rem] border border-line bg-panel text-[0.9rem] text-mute">
          {this.props.name} is waiting for the next event
        </section>
      );
    }
    return this.props.children;
  }
}
