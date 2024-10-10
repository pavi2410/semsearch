import { TfIdf, WordTokenizer } from "natural";
import docsList from "./docs.json";

const indexJson = await Bun.file("./index.json").json();
const tfidf = new TfIdf(indexJson);
const tokenizer = new WordTokenizer();

const searchQuery = process.argv.slice(2).join(' ')

console.log("Semsearch CLI")
console.log(`Search results for "${searchQuery}"`);
const tokens = tokenizer.tokenize(searchQuery);

const results = new Map<number, number>();

const start = performance.now();
tfidf.tfidfs(tokens, (docIdx, score) => {
  if (score > 0) {
    results.set(docIdx, score);
  }
});
const end = performance.now();
console.log(`Found ${results.size} results in ${(end - start).toFixed(2)}ms`);

const sortedResults = Array.from(results.entries()).sort((a, b) => b[1] - a[1]);

for (const [docIdx, score] of sortedResults.slice(0, 10)) {
  const doc = docsList[docIdx];
  console.log(`${doc} (${score.toFixed(2)})`);
}
