import {reddit, redis} from '@devvit/web/server'

/**
 * Real retail-sentiment posts for a ticker, pulled from Reddit via Devvit's
 * server-side reddit client. Ported conceptually from the dashboard's old
 * PRAW-based fetcher, adapted to the real @devvit/reddit@0.14.0 API surface
 * -- there is no `reddit.search()` method in this version (confirmed via
 * `grep -rn "search(" node_modules/@devvit/reddit/*.d.ts`, zero matches).
 * Only getNewPosts/getHotPosts/getTopPosts/getBestPosts exist, so this pulls
 * recent posts per-subreddit and filters by a whole-word ticker match
 * client-side, same idea as a server-side search without one being
 * available.
 */

export type Mention = {
  id: string
  subreddit: string
  title: string
  body: string | undefined
  score: number
  numComments: number
  createdAt: number
  permalink: string
}

const SUBS = ['wallstreetbets', 'stocks', 'investing']
const POSTS_PER_SUB = 50
const CACHE_TTL_SECONDS = 300

function tickerPattern(symbol: string): RegExp {
  // Whole-word match, optional leading '$', so "F" doesn't match every post
  // containing the letter.
  return new RegExp(`(?:^|[^A-Za-z0-9])\\$?${symbol}(?:[^A-Za-z0-9]|$)`, 'i')
}

/**
 * Fetches recent posts from SUBS and filters to ones mentioning `ticker`.
 * Cached in Redis for CACHE_TTL_SECONDS -- the Devvit server process is
 * short-lived per request, so without this every call would re-hit Reddit
 * for all three subreddits.
 */
export async function getRedditSentiment(
  ticker: string,
  postsPerSub = POSTS_PER_SUB,
): Promise<Mention[]> {
  const symbol = ticker.trim().toUpperCase()
  if (!/^[A-Z]{1,5}$/.test(symbol)) throw Error(`invalid ticker: ${ticker}`)

  console.log(`getRedditSentiment: called for ${symbol}`)
  const cacheKey = `sentiment:${symbol}`
  const cached = await redis.get(cacheKey)
  if (cached != null) {
    const parsed = JSON.parse(cached) as Mention[]
    console.log(
      `getRedditSentiment: redis cache hit for ${symbol}, ${parsed.length} mentions`,
    )
    return parsed
  }

  const pattern = tickerPattern(symbol)
  const mentions: Mention[] = []

  // Each subreddit queried separately with its own try/catch -- the
  // installed reddit client has no multireddit ("sub1+sub2") syntax, and
  // one subreddit erroring (e.g. this app not having read access there --
  // see the playtest verification note in the README) shouldn't drop the
  // other two.
  for (const subredditName of SUBS) {
    try {
      const listing = reddit.getNewPosts({subredditName, limit: postsPerSub})
      const posts = await listing.all()
      let matched = 0
      for (const post of posts) {
        const haystack = `${post.title} ${(post.body ?? '').slice(0, 2000)}`
        if (!pattern.test(haystack)) continue
        matched++
        mentions.push({
          id: post.id,
          subreddit: subredditName,
          title: post.title,
          body: post.body,
          score: post.score,
          numComments: post.numberOfComments,
          createdAt: post.createdAt.getTime(),
          permalink: `https://reddit.com${post.permalink}`,
        })
      }
      console.log(
        `getRedditSentiment: r/${subredditName} fetched ${posts.length} posts, ${matched} matched ${symbol}`,
      )
    } catch (err) {
      console.error(`getRedditSentiment: r/${subredditName} failed:`, err)
    }
  }

  mentions.sort((a, b) => b.createdAt - a.createdAt)
  await redis.set(cacheKey, JSON.stringify(mentions), {
    expiration: new Date(Date.now() + CACHE_TTL_SECONDS * 1000),
  })
  console.log(`getRedditSentiment: done for ${symbol}, ${mentions.length} total mentions`)
  return mentions
}

export type SubredditAccessReport = {
  subreddit: string
  ok: boolean
  postCount: number
  samplePostTitle: string | null
  error: string | null
}

/**
 * Answers the one real open question from before this was wired up: can
 * this app read r/wallstreetbets, r/stocks, r/investing via getNewPosts()
 * without being installed in any of them? Hits each subreddit directly
 * (no cache, no filtering) and reports success/failure per sub. Called
 * from the DebugSentiment route below during `devvit playtest` --
 * reachable from the webview with no app token needed, unlike the real
 * /external/sentiment path.
 */
export async function debugSubredditAccess(): Promise<SubredditAccessReport[]> {
  console.log('debugSubredditAccess: starting probe of', SUBS.join(', '))
  const reports: SubredditAccessReport[] = []
  for (const subredditName of SUBS) {
    try {
      const posts = await reddit.getNewPosts({subredditName, limit: 5}).all()
      console.log(
        `debugSubredditAccess: r/${subredditName} ok, ${posts.length} posts`,
      )
      reports.push({
        subreddit: subredditName,
        ok: true,
        postCount: posts.length,
        samplePostTitle: posts[0]?.title ?? null,
        error: null,
      })
    } catch (err) {
      console.error(`debugSubredditAccess: r/${subredditName} failed:`, err)
      reports.push({
        subreddit: subredditName,
        ok: false,
        postCount: 0,
        samplePostTitle: null,
        error: err instanceof Error ? err.message : String(err),
      })
    }
  }
  console.log('debugSubredditAccess: done', JSON.stringify(reports))
  return reports
}
