import { TfIdf, WordTokenizer } from "natural";
import { styleText } from "node:util";
import docsList from "./docs.json";

const indexJson = await Bun.file("./index.json").json();
const tfidf = new TfIdf(indexJson);
const tokenizer = new WordTokenizer();

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
  console.log(`${styleText('bold', doc.title)} (${score.toFixed(2)})
↳ ${styleText('dim', styleText('underline', doc.url))}\n`);
}
