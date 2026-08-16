import {once} from 'node:events'
import type {IncomingMessage, ServerResponse} from 'node:http'
import {context, reddit} from '@devvit/web/server'
import type {
  PartialJsonValue,
  TriggerResponse,
  UiResponse,
} from '@devvit/web/shared'
import {
  type DebugSentimentRsp,
  Endpoint,
  EndpointMethod,
  type ErrorRsp,
  type GetCounterRsp,
  type GetSentimentRsp,
  type IncCounterReq,
  type IncCounterRsp,
} from '../shared/api.ts'
import {dbGetCounter, dbIncCounter} from './db.ts'
import {debugSubredditAccess, getRedditSentiment} from './sentiment.ts'

type AnyRsp =
  | GetCounterRsp
  | IncCounterRsp
  | GetSentimentRsp
  | DebugSentimentRsp
  | UiResponse
  | TriggerResponse
  | ErrorRsp

export async function onReq(
  reqMsg: IncomingMessage,
  rspMsg: ServerResponse,
): Promise<void> {
  try {
    await route(reqMsg, rspMsg)
  } catch (err) {
    const msg = `server error; ${err instanceof Error ? err.stack : err}`
    console.error(msg)
    writeJson<ErrorRsp>(500, {error: msg, status: 500}, rspMsg)
  }
}

async function route(
  reqMsg: IncomingMessage,
  rspMsg: ServerResponse,
): Promise<void> {
  // reqMsg.url carries the query string too (e.g.
  // "external/sentiment?ticker=STX") -- split it off before matching
  // against the Endpoint enum, which only holds bare paths.
  const url = new URL(reqMsg.url ?? '/', 'http://internal')
  const endpoint = url.pathname.slice(1) as Endpoint
  const method = EndpointMethod[endpoint]

  let rsp: AnyRsp
  if (method !== reqMsg.method) {
    rsp = {error: 'not found', status: 404}
  } else {
    switch (endpoint) {
      case Endpoint.GetCounter:
        rsp = await routeGetCounter()
        break
      case Endpoint.IncCounter:
        rsp = await routeInc(reqMsg)
        break
      case Endpoint.OnMenuNewPost:
        rsp = await routeMenuNewPost()
        break
      case Endpoint.OnAppInstall:
        rsp = await routeAppInstall()
        break
      case Endpoint.GetSentiment:
        rsp = await routeGetSentiment(url)
        break
      case Endpoint.DebugSentiment:
        rsp = await routeDebugSentiment()
        break
      default:
        endpoint satisfies never
        rsp = {error: 'not found', status: 404}
        break
    }
  }

  writeJson<PartialJsonValue>('status' in rsp ? rsp.status : 200, rsp, rspMsg)
}

async function routeGetCounter(): Promise<GetCounterRsp> {
  const t3 = context.postId
  if (!t3) throw Error('no t3')
  return {count: await dbGetCounter(t3)}
}

async function routeInc(reqMsg: IncomingMessage): Promise<IncCounterRsp> {
  const t3 = context.postId
  if (!t3) throw Error('no t3')
  const req = await readJson<IncCounterReq>(reqMsg)
  return {count: await dbIncCounter(t3, req.amount)}
}

async function routeMenuNewPost(): Promise<UiResponse> {
  const post = await reddit.submitCustomPost({title: context.appSlug})
  return {
    showToast: {text: `Post ${post.id} created.`, appearance: 'success'},
    navigateTo: post.url,
  }
}

async function routeAppInstall(): Promise<TriggerResponse> {
  await reddit.submitCustomPost({title: context.appSlug})
  return {}
}

/** GET /external/sentiment?ticker=XXX -- see devvit.json's
 * server.externalEndpoints.sentiment (scopes: ["global"]) for the managed
 * app token that authorizes this from outside callers, and sentiment.ts
 * for the actual reddit fetch + filter logic. */
async function routeGetSentiment(
  url: URL,
): Promise<GetSentimentRsp | ErrorRsp> {
  const ticker = url.searchParams.get('ticker')
  if (!ticker) return {error: 'missing ticker query param', status: 400}
  try {
    const mentions = await getRedditSentiment(ticker)
    return {mentions}
  } catch (err) {
    return {
      error: err instanceof Error ? err.message : String(err),
      status: 400,
    }
  }
}

/** GET /api/debug-sentiment -- reachable from the webview with no app
 * token (temporary, see the DebugSentiment endpoint doc comment in
 * shared/api.ts). Reports per-subreddit read access so we can confirm
 * getNewPosts() works on r/wallstreetbets etc. before touching publish. */
async function routeDebugSentiment(): Promise<DebugSentimentRsp> {
  return {reports: await debugSubredditAccess()}
}

async function readJson<T>(reqMsg: IncomingMessage): Promise<T> {
  const chunks: Uint8Array[] = []
  reqMsg.on('data', chunk => chunks.push(chunk))
  await once(reqMsg, 'end')
  return JSON.parse(`${Buffer.concat(chunks)}`)
}

function writeJson<T extends PartialJsonValue>(
  status: number,
  json: Readonly<T>,
  rsp: ServerResponse,
): void {
  const body = JSON.stringify(json)
  const len = Buffer.byteLength(body)
  rsp.writeHead(status, {
    'Content-Length': len,
    'Content-Type': 'application/json',
  })
  rsp.end(body)
}
