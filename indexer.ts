import { TfIdf, TreebankWordTokenizer } from 'natural';
import { readdir } from "node:fs/promises";

function scrapeHtmlContent(html: string) {
  const contents: string[] = [];
  let i = 0;
  const rewriter = new HTMLRewriter();

  rewriter.on('*', {
    text(text) {
      if (text.text.trim() === '') {
        return
      }
      if (contents[i]) {
        contents[i] += text.text
      } else {
        contents[i] = text.text
      }
      if (text.lastInTextNode) {
        i++
      }
    }
  });
  
  rewriter.transform(new Response(html));

  return contents;
}

function extractPageTitle(html: string) {
  let title: string = '';

  new HTMLRewriter().on('title', {
    text(text) {
      title += text.text
    }
  }).transform(new Response(html));

  return title;
}

const tokenizer = new TreebankWordTokenizer();
const tfidf = new TfIdf();

const files = await readdir('webpages');

const docs: Array<{ url: string; title: string; }> = [];

for (const path of files) {
  const { url, content: html } = await Bun.file(`webpages/${path}`).json();

  const title = extractPageTitle(html);

  docs.push({ url, title });

  const contents = scrapeHtmlContent(html);

  const tokens = contents.flatMap(word => tokenizer.tokenize(word.toLowerCase()))

  tfidf.addDocument(tokens);
}

await Bun.write("docs.json", JSON.stringify(docs));
await Bun.write("tfidf.index.json", JSON.stringify(tfidf));
