import { TfIdf, WordTokenizer } from 'natural';
import { readdir } from "node:fs/promises";

class ContentScraper {
  contents: string[];
  i: number;
  constructor() {
    this.contents = []
    this.i = 0
  }
  text(text: HTMLRewriterTypes.Text) {
    if (text.text.trim() === '') {
      return
    }
    if (this.contents[this.i]) {
      this.contents[this.i] += text.text
    } else {
      this.contents[this.i] = text.text
    }
    if (text.lastInTextNode) {
      console.log(this.contents[this.i])
      this.i++
    }
  }
}

function scrapeHtmlContent(html: string) {
  const scraper = new ContentScraper();

  new HTMLRewriter().on('*', scraper).transform(new Response(html));

  return scraper.contents;
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

const tokenizer = new WordTokenizer();
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
