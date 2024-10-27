async function fetchWebPage(url: string): Promise<string> {
  const response = await fetch(url, {
    headers: {
      'Accept': 'text/html',
      'Accept-Language': 'en',
      'User-Agent': 'Googlebot',
    }
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
  }
  return await response.text();
}

async function crawl(startUrls: string[]): Promise<void> {
  const queue: string[] = startUrls;
  const visited: Set<string> = new Set();

  while (queue.length > 0) {
    const url = queue.shift()!;
    if (visited.has(url)) continue;

    try {
      const urlHash = Bun.hash(url);
      const file = Bun.file(`webpages/${new URL(url).hostname}_${urlHash}.json`);

      let html: string;
      try {
        const { lastFetchedAt = 0, content } = await file.json();
        if (Date.now() - lastFetchedAt > 1000 * 60 * 60 * 24) {
          throw new Error('File too old');
        }
        html = content;
        console.log(`Skip fetching ${url}`);
      } catch (error) {
        // File does not exist or is stale; so fetch it
        html = await fetchWebPage(url);
        await Bun.write(file, JSON.stringify({
          url,
          lastFetchedAt: Date.now(),
          content: html,
        }));
        console.log(`Saved ${url} to ${urlHash}`);
      }

      visited.add(url);

      const links = extractLinks(html, url);
      for (const link of links) {
        if (!visited.has(link)) {
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
      if (href && href.startsWith('http') && !href.startsWith('#')) {
        const absoluteUrl = new URL(href, baseUrl).toString();
        links.push(absoluteUrl);
      }
    }
  });

  rewriter.transform(new Response(html));

  return links;
}


const startUrls = [
  'https://en.wikipedia.org',
  'https://news.ycombinator.com',
  'https://www.reddit.com',
  'https://pavi2410.me',
];

console.time('crawl')
await crawl(startUrls)
console.timeEnd('crawl')
