import { TfIdf, TreebankWordTokenizer } from "natural";

const indexJson = await Bun.file("./index.json").json();
const tfidf = new TfIdf(indexJson);
const tokenizer = new TreebankWordTokenizer();

type SearchResult = {
  queryTokens: string[];
  queryTime: number;
  results: [docIdx: number, score: number][];
}

export function search(query: string): SearchResult {
  const tokens = tokenizer.tokenize(query.toLowerCase());

  const results = new Map<number, number>();

  const start = performance.now();
  tfidf.tfidfs(tokens, (docIdx, score) => {
    if (score > 0) {
      results.set(docIdx, score);
    }
  });
  const end = performance.now();

  const sortedResults = Array.from(results.entries()).sort((a, b) => b[1] - a[1]);

  return {
    queryTokens: tokens,
    queryTime: end - start,
    results: sortedResults,
  }
}