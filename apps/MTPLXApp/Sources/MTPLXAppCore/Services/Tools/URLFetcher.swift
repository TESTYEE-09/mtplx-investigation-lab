import Foundation
import os

// MARK: - URLFetcher
//
// Port of Aphanes V2's `NativeWebSearchService.fetchURL`. Plain HTTP
// GET with a real browser User-Agent, HTML-aware extraction of title +
// readable text, and a 4000-character cap on the returned content so
// fetched pages don't blow out the model's context window when injected
// as tool results. No JS rendering, no headless browser — if a page is
// JS-only, we return whatever HTML the server initially shipped.

public struct URLFetchRequest: Sendable, Hashable {
    public var url: URL
    /// Optional override of the 4000-character cap. Leave nil to use
    /// the default; pass a larger value at your own risk (the model's
    /// context budget eats it).
    public var maxCharacters: Int?

    public init(url: URL, maxCharacters: Int? = nil) {
        self.url = url
        self.maxCharacters = maxCharacters
    }
}

public struct URLFetchResult: Sendable, Hashable {
    public var url: URL
    public var title: String?
    public var content: String

    public init(url: URL, title: String?, content: String) {
        self.url = url
        self.title = title
        self.content = content
    }
}

public actor URLFetchCache {
    public static let shared = URLFetchCache()
    private var pages: [String: URLFetchResult] = [:]

    public init() {}

    public func cached(for key: String) -> URLFetchResult? {
        pages[key]
    }

    public func store(_ result: URLFetchResult, for key: String) {
        pages[key] = result
    }

    public func clear() {
        pages.removeAll()
    }
}

public final class URLFetcher: @unchecked Sendable {
    public static let defaultMaxCharacters = 4000

    private let transport: any WebTransport
    private let cache: URLFetchCache
    private static let log = Logger(subsystem: "com.mtplx.app", category: "URLFetch")

    public init(
        transport: any WebTransport = URLSessionWebTransport(),
        cache: URLFetchCache = .shared
    ) {
        self.transport = transport
        self.cache = cache
    }

    public func fetch(_ request: URLFetchRequest) async throws -> URLFetchResult {
        let cap = request.maxCharacters ?? Self.defaultMaxCharacters
        let cacheKey = "\(cap)|\(request.url.absoluteString)"
        if let cached = await cache.cached(for: cacheKey) {
            return cached
        }

        var urlRequest = URLRequest(url: request.url)
        urlRequest.setValue(WebSearchService.browserUserAgent, forHTTPHeaderField: "User-Agent")
        urlRequest.setValue("text/html,application/xhtml+xml", forHTTPHeaderField: "Accept")
        urlRequest.setValue("en-US,en;q=0.9", forHTTPHeaderField: "Accept-Language")

        let (data, response) = try await transport.data(for: urlRequest)
        try WebSearchService.validate(response: response)

        let contentType = (response as? HTTPURLResponse)?
            .value(forHTTPHeaderField: "Content-Type")?
            .lowercased() ?? ""
        let body = String(decoding: data, as: UTF8.self)

        let result: URLFetchResult
        if contentType.contains("html") || body.contains("<html") || body.contains("<body") {
            result = URLFetchResult(
                url: request.url,
                title: WebHTML.extractHTMLTitle(from: body),
                content: Self.cap(WebHTML.extractReadableContent(from: body), at: cap)
            )
        } else {
            result = URLFetchResult(
                url: request.url,
                title: request.url.host,
                content: Self.cap(WebHTML.normalizeWhitespace(body), at: cap)
            )
        }
        await cache.store(result, for: cacheKey)
        return result
    }

    private static func cap(_ text: String, at limit: Int) -> String {
        guard text.count > limit else { return text }
        let end = text.index(text.startIndex, offsetBy: limit)
        return String(text[..<end]) + "…"
    }
}
