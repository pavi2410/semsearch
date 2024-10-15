import bm25 from 'wink-bm25-text-search';
import nlp from 'wink-nlp-utils';

const indexJson = await Bun.file("./index.json").text();

const bm25Engine = bm25();
bm25Engine.importJSON(indexJson);

const pipe = [
  nlp.string.lowerCase,
  nlp.string.tokenize0,
  nlp.tokens.removeWords,
  nlp.tokens.stem,
  nlp.tokens.propagateNegations
];

bm25Engine.definePrepTasks(pipe);

type SearchResult = {
  queryTime: number;
  results: [docIdx: string, score: number][];
}

export function search(query: string): SearchResult {
  const start = performance.now();
  const results = bm25Engine.search(query)
  const end = performance.now();

  return {
    queryTime: end - start,
    results,
  }
}