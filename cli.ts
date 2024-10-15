import { LevenshteinDistanceSearch } from "natural";
import { styleText } from "node:util";
import docsList from "./docs.json";
import { search } from "./search";

const searchQuery = process.argv.slice(2).join(' ')

console.log("Semsearch CLI\n")
console.log(`Search results for "${styleText('bold', searchQuery)}"`);

const { queryTokens, queryTime, results } = search(searchQuery);

console.log(styleText('dim', `Found ${results.length} results in ${queryTime.toFixed(2)}ms\n`));

for (const [docIdx, score] of results.slice(0, 10)) {
  const doc = docsList[docIdx];

  const title = doc.title.trim() === '' ? styleText('italic', 'Untitled page') : styleText('bold', addHighlights(doc.title, queryTokens));

  const hostname = new URL(doc.url).hostname;
  const start = doc.url.indexOf(hostname);
  const end = start + hostname.length;
  const url = styleText('dim', styleText('underline', highlightSpan(doc.url, start, end, 'italic')));

  const scoreDisplay = styleText('dim', `(${score.toFixed(2)})`);

  console.log(`${title} ${scoreDisplay}\n↳ ${url}\n`);
}

function highlightSpan(text: string, start: number, end: number, format: Parameters<typeof styleText>[0] = 'bgYellow') {
  return text.slice(0, start) + styleText(format, text.slice(start, end)) + text.slice(end);
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
