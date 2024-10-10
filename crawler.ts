async function fetchWebPage(url: string): Promise<string> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
  }
  return await response.text();
}

async function crawl(startUrl: string): Promise<void> {
  const queue: string[] = [startUrl];
  const visited: Set<string> = new Set();

  while (queue.length > 0) {
    const url = queue.shift()!;
    if (visited.has(url)) continue;

    try {
      const html = await fetchWebPage(url);
      const filename = Bun.hash(url);
      await Bun.write(`webpages/${new URL(url).hostname}_${filename}.json`, JSON.stringify({
        url,
        content: html,
      }));
      console.log(`Saved ${url} to ${filename}`);

      visited.add(url);

      const links = extractLinks(html, url);
      for (const link of links) {
        if (!visited.has(link) && link.startsWith(startUrl)) {
          queue.push(link);
        }
      }
    } catch (error) {
      console.error(`Error crawling ${url}`);
    }
  }
}

function extractLinks(html: string, baseUrl: string): string[] {
  const links: string[] = [];
  const rewriter = new HTMLRewriter();

  rewriter.on('a', {
    element(el) {
      const href = el.getAttribute('href');
      if (href && !href.startsWith('#')) {
        const absoluteUrl = new URL(href, baseUrl).toString();
        links.push(absoluteUrl);
      }
    }
  });

  rewriter.transform(new Response(html));

  return links;
}


const urls = [
  'https://en.wikipedia.org',
  'https://news.ycombinator.com',
];

console.time('crawl')
await Promise.allSettled(urls.map(url => crawl(url)))
console.timeEnd('crawl')

