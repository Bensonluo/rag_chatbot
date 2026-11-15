/** Chat island containment: a render-time crash inside the live chat must
 * degrade to a bilingual fallback inside the panel — never blank the whole
 * storefront, which is React's behavior for uncaught errors (it unmounts
 * the entire tree). The fallback's retry must remount the island. */
import { act, StrictMode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { LanguageProvider } from '@/lib/i18n/context';
import { ChatErrorBoundary } from '@/components/ChatErrorBoundary';

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

let container: HTMLDivElement;
let root: Root | undefined;

beforeEach(() => {
  localStorage.clear();
  localStorage.setItem('portfolio-locale', 'en');
  container = document.createElement('div');
  document.body.appendChild(container);
});

afterEach(() => {
  act(() => root?.unmount());
  container.remove();
  root = undefined;
});

function ThrowingIsland(): never {
  throw new Error('island render crashed');
}

test('a crashing chat island degrades to the fallback note, not a blank page', () => {
  act(() => {
    root = createRoot(container);
    root.render(
      <StrictMode>
        <LanguageProvider>
          <ChatErrorBoundary>
            <ThrowingIsland />
          </ChatErrorBoundary>
          <p>Storefront content outside the island</p>
        </LanguageProvider>
      </StrictMode>
    );
  });

  // The rest of the page survives; the island shows its bilingual note.
  expect(container.textContent).toContain('Storefront content outside the island');
  expect(container.textContent).toContain('The live chat hit an error');
  expect((container.querySelector('[role="alert"] button') as HTMLButtonElement)
    .textContent).toBe('Try again');
});

test('the fallback retry remounts the island instead of stranding the user', () => {
  // The fault is modeled as a stable condition, not a throw-once counter:
  // React's concurrent renderer re-attempts a failed render synchronously,
  // so a first-call-only crash can be absorbed without the boundary.
  let broken = true;
  let renders = 0;
  function Island() {
    renders += 1;
    if (broken) throw new Error('island fault persists until retried');
    return <p>Chat recovered</p>;
  }

  act(() => {
    root = createRoot(container);
    root.render(
      <StrictMode>
        <LanguageProvider>
          <ChatErrorBoundary>
            <Island />
          </ChatErrorBoundary>
        </LanguageProvider>
      </StrictMode>
    );
  });
  expect(container.textContent).toContain('The live chat hit an error');
  expect(renders).toBeGreaterThanOrEqual(1);

  broken = false;
  act(() => {
    (container.querySelector('[role="alert"] button') as HTMLButtonElement).click();
  });
  expect(renders).toBeGreaterThan(1);
  expect(container.textContent).toContain('Chat recovered');
  expect(container.textContent).not.toContain('The live chat hit an error');
});

test('the fallback follows the visitor locale', () => {
  localStorage.setItem('portfolio-locale', 'zh');
  act(() => {
    root = createRoot(container);
    root.render(
      <StrictMode>
        <LanguageProvider>
          <ChatErrorBoundary>
            <ThrowingIsland />
          </ChatErrorBoundary>
        </LanguageProvider>
      </StrictMode>
    );
  });
  expect(container.textContent).toContain('在线客服暂时不可用');
  expect((container.querySelector('[role="alert"] button') as HTMLButtonElement)
    .textContent).toBe('重试');
});
