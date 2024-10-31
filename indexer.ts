import { readdir } from "node:fs/promises";
import bm25 from 'wink-bm25-text-search';
import nlp from 'wink-nlp-utils';
import * as he from 'he';

function scrapeHtmlContent(rewriter: HTMLRewriter) {
  const contents: string[] = [];
  let i = 0;
  let skip = false;

  rewriter.on('body *', {
    element(element) {
      if (element.tagName === 'style' || element.tagName === 'script') {
        skip = true;
      }
      element.onEndTag((endTag) => {
        skip = false;
      })
    }
  })

  rewriter.on('body', {
    text(text) {
      if (skip) return;

      if (contents[i]) {
        contents[i] += text.text
      } else {
        contents[i] = text.text
      }
      if (text.lastInTextNode) {
        contents[i] = he.decode(contents[i].trim())
        if (contents[i].length > 0) {
          i++
        }
      }
    }
  });

  return () => contents.join(' ');
}

function extractPageTitle(rewriter: HTMLRewriter) {
  let title = '';

  rewriter.on('head > title', {
    text(text) {
      title += text.text
    }
  });

  return () => he.decode(title.trim());
}

const bm25Engine = bm25();

bm25Engine.defineConfig({ fldWeights: { title: 1, content: 1 } });

const pipe = [
  nlp.string.lowerCase,
  nlp.string.tokenize0,
  nlp.tokens.removeWords,
  nlp.tokens.stem,
  nlp.tokens.propagateNegations
];

bm25Engine.definePrepTasks(pipe);

const files = await readdir('webpages');

console.log(`Found ${files.length} webpages`)

const docs: Record<string, { url: string; title: string; }> = {};

let i = 1;
for (const path of files) {
  const { url, content: html } = await Bun.file(`webpages/${path}`).json();

  const rewriter = new HTMLRewriter();

  const titleRef = extractPageTitle(rewriter);
  const contentRef = scrapeHtmlContent(rewriter);

  rewriter.transform(html);

  const urlHash = String(Bun.hash(url));
  const title = titleRef();
  const content = contentRef();

  docs[urlHash] = { url, title };

  bm25Engine.addDoc({
    url: url,
    title,
    content,
  }, urlHash);

  console.write(`Indexed page ${i++} of ${files.length}\r`)
}

console.write('\n')

bm25Engine.consolidate();

await Bun.write("docs.json", JSON.stringify(docs));
await Bun.write("index.json", bm25Engine.exportJSON());
