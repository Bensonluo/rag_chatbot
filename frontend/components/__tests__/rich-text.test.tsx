/** RichText renders the assistant reply markdown subset without an HTML surface. */
import { act, StrictMode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { RichText } from '@/components/RichText';

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

let root: Root | undefined;
let container: HTMLDivElement;

function render(text: string): void {
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
  act(() => {
    root!.render(
      <StrictMode>
        <RichText text={text} />
      </StrictMode>
    );
  });
}

beforeEach(() => {
  root = undefined;
});

afterEach(() => {
  if (root) {
    act(() => root!.unmount());
  }
  container.remove();
});

describe('RichText', () => {
  it('renders **bold** spans as <strong> without the asterisks', () => {
    render('My pick: **Hengrui Nova N0001** at **¥212**.');
    const strongs = Array.from(container.querySelectorAll('strong'));
    expect(strongs.map((node) => node.textContent)).toEqual(['Hengrui Nova N0001', '¥212']);
    expect(container.textContent).toContain('My pick: Hengrui Nova N0001 at ¥212.');
    expect(container.textContent).not.toContain('**');
  });

  it('groups consecutive - / • lines into a single list', () => {
    render('Options:\n- **Nova N0001** — ¥212\n- Vertex V0003 — ¥240\n• Third option');
    const lists = container.querySelectorAll('ul');
    expect(lists).toHaveLength(1);
    const items = Array.from(lists[0].querySelectorAll('li'));
    expect(items).toHaveLength(3);
    expect(items[0].textContent).toBe('Nova N0001 — ¥212');
    expect(items[0].querySelector('strong')).not.toBeNull();
    expect(items[2].textContent).toBe('Third option');
  });

  it('splits paragraphs on blank lines and preserves inner line breaks', () => {
    render('Line one\nLine two\n\nSecond paragraph');
    const paras = container.querySelectorAll('p');
    expect(paras).toHaveLength(2);
    expect(paras[0].textContent).toBe('Line one\nLine two');
    expect(paras[1].textContent).toBe('Second paragraph');
  });

  it('leaves a stray unbalanced ** as literal text', () => {
    render('Price **¥212 and rising');
    expect(container.querySelector('strong')).toBeNull();
    expect(container.textContent).toContain('Price **¥212 and rising');
  });

  it('never interprets HTML-looking content as markup', () => {
    render('Try <img src=x onerror=alert(1)> and <script>alert(2)</script>');
    expect(container.querySelector('img')).toBeNull();
    expect(container.querySelector('script')).toBeNull();
    expect(container.textContent).toContain('<img src=x onerror=alert(1)>');
  });

  it('renders an empty string as an empty container', () => {
    render('');
    expect(container.textContent).toBe('');
    expect(container.querySelectorAll('p, ul')).toHaveLength(0);
  });
});
