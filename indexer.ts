import { TfIdf, TreebankWordTokenizer } from 'natural';
import { readdir } from "node:fs/promises";

function scrapeHtmlContent(rewriter: HTMLRewriter) {
  const contents: string[] = [];
  let i = 0;

  rewriter.on('*', {
    text(text) {
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

  return contents;
}

function extractPageTitle(rewriter: HTMLRewriter) {
  let title: [string] = [''];

  rewriter.on('title', {
    text(text) {
      title[0] += text.text
    }
  });

  return title;
}

const tokenizer = new TreebankWordTokenizer();
const tfidf = new TfIdf();

const files = await readdir('webpages');

const docs: Array<{ url: string; title: string; }> = [];

for (const path of files) {
  const { url, content: html } = await Bun.file(`webpages/${path}`).json();

  const rewriter = new HTMLRewriter();

  const titleRef = extractPageTitle(rewriter);
  const contents = scrapeHtmlContent(rewriter);

  rewriter.transform(new Response(html));

  docs.push({ url, title: titleRef[0] });

  const tokens = contents.flatMap(word => tokenizer.tokenize(word.toLowerCase()))

  tfidf.addDocument(tokens);
}

await Bun.write("docs.json", JSON.stringify(docs));
await Bun.write("index.json", JSON.stringify(tfidf));
