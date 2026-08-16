/** Generic error detail for all responses. */
export type ErrorRsp = {error: string; status: number}

/** The current counter state for this post. */
export type GetCounterRsp = {count: number}

/** Increment the post counter by a signed amount. */
export type IncCounterReq = {amount: number}
export type IncCounterRsp = {count: number}

/** GET /external/sentiment?ticker=XXX -- called by the external Streamlit
 * dashboard (a "global"-scope managed app token call, see devvit.json's
 * server.externalEndpoints.sentiment). Not reachable from the webview
 * client -- this is the outside-in path. */
export type GetSentimentRsp = {
  mentions: {
    id: string
    subreddit: string
    title: string
    body: string | undefined
    score: number
    numComments: number
    createdAt: number
    permalink: string
  }[]
}

export type Endpoint = (typeof Endpoint)[keyof typeof Endpoint]
export const Endpoint = {
  GetCounter: 'api/counter',
  IncCounter: 'api/counter/inc',
  OnAppInstall: 'internal/on/app/install',
  OnMenuNewPost: 'internal/on/menu/new-post',
  GetSentiment: 'external/sentiment',
  /** GET /api/debug-sentiment?ticker=XXX -- same getRedditSentiment() call
   * as GetSentiment, but reachable from the webview client during
   * `devvit playtest` with no app token needed (like GetCounter). Exists
   * purely to answer "can this app read wallstreetbets/stocks/investing
   * without being installed there" before bothering with publish + a
   * managed app token. Remove once that's confirmed and the real
   * external-endpoint path is verified end to end. */
  DebugSentiment: 'api/debug-sentiment',
} as const

export type DebugSentimentRsp = {
  reports: {
    subreddit: string
    ok: boolean
    postCount: number
    samplePostTitle: string | null
    error: string | null
  }[]
}

export const EndpointMethod = {
  [Endpoint.GetCounter]: 'GET',
  [Endpoint.IncCounter]: 'POST',
  [Endpoint.OnAppInstall]: 'POST',
  [Endpoint.OnMenuNewPost]: 'POST',
  [Endpoint.GetSentiment]: 'GET',
  [Endpoint.DebugSentiment]: 'GET',
} as const satisfies {[endpoint: string]: 'GET' | 'POST'}
