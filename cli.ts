import { LevenshteinDistanceSearch, TfIdf, TreebankWordTokenizer } from "natural";
import { styleText } from "node:util";
import docsList from "./docs.json";

const indexJson = await Bun.file("./tfidf.index.json").json();
const tfidf = new TfIdf(indexJson);
const tokenizer = new TreebankWordTokenizer();

const searchQuery = process.argv.slice(2).join(' ')

console.log("Semsearch CLI\n")
console.log(`Search results for "${styleText('bold', searchQuery)}"`);
const tokens = tokenizer.tokenize(searchQuery.toLowerCase());

const results = new Map<number, number>();

const start = performance.now();
tfidf.tfidfs(tokens, (docIdx, score) => {
  if (score > 0) {
    results.set(docIdx, score);
  }
});
const end = performance.now();
console.log(styleText('dim', `Found ${results.size} results in ${(end - start).toFixed(2)}ms\n`));

const sortedResults = Array.from(results.entries()).sort((a, b) => b[1] - a[1]);

for (const [docIdx, score] of sortedResults.slice(0, 10)) {
  const doc = docsList[docIdx];

  console.log(`${styleText('bold', addHighlights(doc.title, tokens))} (${score.toFixed(2)})
↳ ${styleText('dim', styleText('underline', doc.url))}\n`);
}

function highlightSpan(text: string, start: number, end: number) {
  return text.slice(0, start) + styleText('bgYellow', text.slice(start, end)) + text.slice(end);
}

type Span = [start: number, end: number];
function addHighlights(text: string, tokens: string[]) {
  const lowercasedText = text.toLowerCase();
  
  const spans: Span[] = [];
  for (const token of tokens) {
    const ld = LevenshteinDistanceSearch(token, lowercasedText);
    if (ld.distance < token.length / 2) {
      spans.push([ld.offset, ld.offset + ld.substring.length]);
    }
  }

  spans.sort((a, b) => (a[0] === b[0]) ? (a[1] - b[1]) : (a[0] - b[0]));

  const mergedSpans: Span[] = [];
  for (const span of spans) {
    if (mergedSpans.length === 0) {
      mergedSpans.push(span);
    } else {
      const lastSpan = mergedSpans[mergedSpans.length - 1];
      if (span[0] <= lastSpan[1]) {
        lastSpan[1] = Math.max(lastSpan[1], span[1]);
      } else {
        mergedSpans.push(span);
      }
    }
  }

  let m = 0;
  for (const [start, end] of mergedSpans) {
    text = highlightSpan(text, m + start, m + end);
    m += 10; // ANSI escape code length offset added on each highlight
  }

  return text
}
