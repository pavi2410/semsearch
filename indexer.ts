import { readdir } from "node:fs/promises";
import bm25 from 'wink-bm25-text-search';
import nlp from 'wink-nlp-utils';

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
        contents[i] = contents[i].trim()
        if (contents[i].length > 0) {
          i++
        }
      }
    }
  });

  return contents;
}

function extractPageTitle(rewriter: HTMLRewriter) {
  let title: [string] = [''];

  rewriter.on('head > title', {
    text(text) {
      title[0] += text.text
    }
  });

  return title;
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

const docs: Record<string, { url: string; title: string; }> = {};

for (const path of files) {
  const { url, content: html } = await Bun.file(`webpages/${path}`).json();

  const rewriter = new HTMLRewriter();

  const titleRef = extractPageTitle(rewriter);
  const contents = scrapeHtmlContent(rewriter);

  rewriter.transform(html);

  const urlHash = String(Bun.hash(url));
  const title = nlp.string.trim(titleRef[0]);

  docs[urlHash] = { url, title };

  bm25Engine.addDoc({
    url: url,
    title,
    content: contents.join(' '),
  }, urlHash);
}

bm25Engine.consolidate();

await Bun.write("docs.json", JSON.stringify(docs));
await Bun.write("index.json", bm25Engine.exportJSON());
