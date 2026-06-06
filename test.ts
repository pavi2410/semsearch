import { Database } from 'bun:sqlite';
import { readdir } from "node:fs/promises";

const db = new Database('./data/semsearch.db');
db.exec("PRAGMA journal_mode=WAL;");

db.exec(`CREATE TABLE IF NOT EXISTS webpages (
    url TEXT NOT NULL UNIQUE,
    content TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);`)

db.exec(`CREATE VIRTUAL TABLE IF NOT EXISTS webpages_fts USING fts5(
    url,
    title,
    contents,
    tokenize = 'porter'
);`)

db.exec(`CREATE VIRTUAL TABLE IF NOT EXISTS webpages_fts_trigram USING fts5(
    url,
    title,
    contents,
    tokenize = 'trigram'
);`)

const files = await readdir('webpages');

const stmt = db.query(`INSERT INTO webpages (url, content) VALUES (?, ?)`);
for (const path of files) {
    const { url, content: html } = await Bun.file(`webpages/${path}`).json();
    stmt.run(url, html.trim());
}
stmt.finalize();

////////////////////////////////////////

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

const indexQuery = db.query(`INSERT INTO webpages_fts (url, title, contents) VALUES (?, ?, ?)`);
const indexTriQuery = db.query(`INSERT INTO webpages_fts_trigram (url, title, contents) VALUES (?, ?, ?)`);

const pages = db.query(`SELECT url, content FROM webpages`);
for (const { url, content: html } of pages.iterate()) {
  const rewriter = new HTMLRewriter();

  const titleRef = extractPageTitle(rewriter);
  const contents = scrapeHtmlContent(rewriter);

  rewriter.transform(html);

  const title = titleRef[0].trim();

  indexQuery.run(url, title, contents.join(' '));
  indexTriQuery.run(url, title, contents.join(' '));
}
indexQuery.finalize();

////////////////////////////////////////

const searchString = process.argv.slice(2).join(' ')
console.log(`Search results for "${searchString}"`);


const searchQuery = db.query(`SELECT url, title, rank FROM webpages_fts WHERE contents MATCH ? ORDER BY rank LIMIT 5`);

const results = searchQuery.all(searchString);

for (const row of results) {
    console.log(`${row.title} ${row.rank}\n↳ ${row.url}\n`);
}

console.log('-------------------------')

{
    const searchQuery = db.query(`SELECT url, title, rank FROM webpages_fts_trigram WHERE contents MATCH ? ORDER BY rank LIMIT 5`);

    const results = searchQuery.all(searchString);

    for (const row of results) {
        console.log(`${row.title} ${row.rank}\n↳ ${row.url}\n`);
    }
}